#!/usr/bin/env python3
"""Replay Step-2 MyoLeg loopable references with gait events; show graphs after playback.

INPUT: gait_with_landmarks/SXXX/TrialXX/stepYY_loopable.npz
       and the adjacent stepYY_loopable_manifest.json / _events.csv.

This is purely kinematic replay: sets qpos/qvel and calls mj_forward, NEVER mj_step.
The pelvis keeps advancing by the saved stride displacement at every cycle.
Use --plots-at-end for an explicitly offline plot mode: only the MuJoCo
viewer runs during replay. After the requested cycles end (or the user closes
the viewer), joint and landmark plots are saved and shown. No live plotting.

python replay_loopable_gait_with_events.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --cycles 10 --speed 0.5 --reference "gait_with_landmarks\\S060\\Trial01\\step01_loopable.npz" --plots-at-end --all-cycles --noise-deg 1 --noise-seed 42 --noise-cutoff-hz 20

Requires: numpy, matplotlib, mujoco, glfw (usually installed with mujoco).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


JOINTS = (
    ("Right hip flexion", "hip_flexion_r"),
    ("Left hip flexion", "hip_flexion_l"),
    ("Right knee", "knee_angle_r"),
    ("Left knee", "knee_angle_l"),
    ("Right ankle", "ankle_angle_r"),
    ("Left ankle", "ankle_angle_l"),
)


@dataclass(frozen=True)
class Event:
    phase: float  # [0,1); right HS at 0 recurs at every stride boundary
    side: str     # right / left
    kind: str     # HS / TO
    provenance: str  # measured / corroborated / boundary / assumed / candidate
    detail: str

    @property
    def label(self) -> str:
        return f"{self.side.title()} {self.kind} ({self.provenance})"

    @property
    def color(self) -> str:
        if self.provenance in ("measured", "corroborated"):
            return "tab:green"
        if self.provenance == "candidate":
            return "tab:pink"
        return "tab:orange"


def _periodic_velocity(x: np.ndarray, t: np.ndarray, T: float) -> np.ndarray:
    """Derivative of a periodic signal sampled on [0, T] with x[-1] == x[0]."""
    xc, tc = x[:-1], t[:-1]
    xe = np.concatenate([xc[-1:], xc, xc[:1]])
    te = np.concatenate([[tc[-1] - T], tc, [tc[0] + T]])
    v = np.gradient(xe, te)[1:-1]
    return np.append(v, v[0])


def _smooth_periodic_noise(m: int, T: float, cutoff_hz: float, std: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Low-pass white noise on a closed cycle of m samples, scaled to the given std.

    Filtering is done in the frequency domain on the periodic signal, so the
    result wraps seamlessly. Frequencies are multiples of 1/T; the mean is removed.
    """
    spec = np.fft.rfft(rng.normal(0.0, 1.0, m))
    freqs = np.fft.rfftfreq(m, d=T / m)
    spec[freqs > cutoff_hz] = 0.0
    sig = np.fft.irfft(spec, n=m)
    sig -= sig.mean()
    sig *= std / sig.std()
    return sig


class Reference:
    def __init__(self, path: Path):
        self.path = path
        with np.load(path, allow_pickle=False) as f:
            self.qpos = np.asarray(f["qpos"], dtype=float).copy()
            self.qvel = np.asarray(f["qvel"], dtype=float).copy()
            self.t = np.asarray(f["time_since_start_s"], dtype=float).copy()
            self.T = float(f["stride_duration_s"])
            self.displacement = np.asarray(f["root_displacement_xyz_m"], dtype=float).copy()
            self.seam = float(f["seam_fraction"]) if "seam_fraction" in f else None
        self.noise = None          # set by add_joint_noise
        self._cycle_cache = {}     # cycle index -> (qpos, qvel) with that step's noise
        if (self.qpos.ndim != 2 or self.qvel.ndim != 2 or
                len(self.t) != len(self.qpos) or len(self.t) != len(self.qvel) or
                self.T <= 0 or len(self.t) < 3 or
                not np.isclose(self.t[0], 0) or not np.isclose(self.t[-1], self.T) or
                np.any(np.diff(self.t) <= 0)):
            raise ValueError("Invalid Step-2 qpos/qvel/time arrays")
        if self.displacement.shape != (3,):
            raise ValueError("Expected root_displacement_xyz_m shape (3,)")
        if not all(np.all(np.isfinite(a)) for a in (self.qpos, self.qvel, self.t, self.displacement)):
            raise ValueError("Non-finite reference data")
        if not np.allclose(self.qpos[-1, :2] - self.qpos[0, :2], self.displacement[:2], atol=1e-5):
            raise ValueError("Root displacement inconsistent with reference endpoints")
        if not np.allclose(self.qvel[0], self.qvel[-1], atol=1e-3):
            print("WARNING: saved qvel differs between the two right-HS endpoints")

    def add_joint_noise(self, qidx, vidx, std_rad: float, seed: int, cutoff_hz: float = 0.0):
        """Give every step its own joint-angle perturbation (step-to-step variability).

        Step k draws independent noise from default_rng([seed, k]). cutoff_hz == 0
        gives white noise; cutoff_hz > 0 gives low-pass noise. The nominal arrays
        are left untouched; use cycle_arrays(k) to get step k's perturbed values.
        """
        self.noise = {"qidx": list(qidx), "vidx": list(vidx), "std": std_rad,
                      "seed": seed, "cutoff": cutoff_hz}
        self._cycle_cache = {}

    def cycle_arrays(self, k: int):
        """qpos/qvel for step k. Each step's noise is zero at its first and last frame,
        so consecutive steps join without a jump at heel strike."""
        if self.noise is None:
            return self.qpos, self.qvel
        if k in self._cycle_cache:
            return self._cycle_cache[k]
        nz = self.noise
        n = len(self.t)
        rng = np.random.default_rng([nz["seed"], k])
        qp = self.qpos.copy()
        qv = self.qvel.copy()
        for q_i, v_i in zip(nz["qidx"], nz["vidx"]):
            if nz["cutoff"] > 0:
                p = _smooth_periodic_noise(n - 1, self.T, nz["cutoff"], nz["std"], rng)
            else:
                p = rng.normal(0.0, nz["std"], n - 1)
            p = np.append(p, p[0])
            p = p - p[0]
            qp[:, q_i] += p
            qv[:, v_i] = _periodic_velocity(qp[:, q_i], self.t, self.T)
        self._cycle_cache[k] = (qp, qv)
        return qp, qv

    def sample(self, unwrapped_time_s: float):
        """Interpolate one cycle and accumulate root travel over whole cycles.

        Use the first frame at exact t=k*T, with k*displacement added.
        This avoids an extra frame or a root-position reset at each wrap.
        """
        cycles = math.floor(unwrapped_time_s / self.T)
        local = unwrapped_time_s - cycles * self.T
        if local >= self.T - 1e-10:
            cycles += 1
            local = 0.0
        i = int(np.searchsorted(self.t, local, side="right") - 1)
        i = min(max(0, i), len(self.t) - 2)
        a = float((local - self.t[i]) / (self.t[i + 1] - self.t[i]))
        qp, qv = self.cycle_arrays(cycles)
        q = (1 - a) * qp[i] + a * qp[i + 1]
        v = (1 - a) * qv[i] + a * qv[i + 1]
        # Free-root quaternion is [w,x,y,z], not an ordinary vector.
        q0, q1 = qp[i, 3:7], qp[i + 1, 3:7]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        quat = (1 - a) * q0 + a * q1
        q[3:7] = quat / np.linalg.norm(quat)
        q[:3] += cycles * self.displacement
        return q, v, local / self.T


def _raw_candidates(ref: Reference) -> list[dict]:
    """Read the source CSV for corroboration, not as automatically valid events."""
    csv_path = ref.path.with_name(ref.path.stem + "_events.csv")
    if not csv_path.is_file():
        return []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        try:
            # Exporter's normalized_phase refers to this very stride.
            phase = float(row["normalized_phase"])
            side = (row.get("foot_side") or "unassigned").strip().lower()
            kind = (row.get("event") or "").strip().lower()
            kind = "HS" if kind in ("hs_candidate", "hs", "on") else (
                "TO" if kind in ("to_candidate", "to", "off") else None)
            if kind and 0 <= phase <= 1 and side in ("left", "right"):
                result.append({"phase": phase, "side": side, "kind": kind,
                               "detail": row.get("force_channel", "")})
        except (KeyError, ValueError, TypeError):
            continue
    return result


def get_events(ref: Reference, show_raw_crossings: bool = False) -> list[Event]:
    """Selected annotations are from the original manifest, not guessed from kinematics."""
    path = ref.path.with_name(ref.path.stem + "_manifest.json")
    man = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    em = man.get("events", {})
    raw = _raw_candidates(ref)
    boundary = em.get("right_hs_boundary_sources", [])
    corroborated = any("corrob" in str(b).lower() for b in boundary)
    corroborated |= any(x["side"] == "right" and x["kind"] == "HS" and
                       min(x["phase"], 1 - x["phase"]) * ref.T <= 0.03 for x in raw)
    events = [Event(0.0, "right", "HS", "corroborated" if corroborated else "boundary",
                    "dataset right-HS stride boundary")]

    left_phase = em.get("left_hs_phase_in_right_stride")
    if left_phase is None and em.get("left_hs_time_s") is not None and man.get("t0_s") is not None:
        left_phase = (float(em["left_hs_time_s"]) - float(man["t0_s"])) / ref.T
    if left_phase is not None:
        source = str(em.get("left_hs_source", "unknown"))
        provenance = ("assumed" if source.upper().startswith("ASSUMED") else
                      "measured" if source.lower().startswith("observed") else "boundary")
        if 0 < float(left_phase) < 1:
            events.append(Event(float(left_phase), "left", "HS", provenance, source))
    else:
        print("WARNING: selected left-HS not found in companion manifest; not inventing one")

    for side, key, source_key in (("left", "left_toe_off_phase", "left_toe_off_source"),
                                  ("right", "right_toe_off_phase", "right_toe_off_source")):
        phase = em.get(key)
        if phase is None:
            continue
        source = str(em.get(source_key, "provenance_unknown"))
        provenance = ("assumed" if source.upper().startswith("ASSUMED") else
                      "measured" if source.lower().startswith("observed") else "boundary")
        if 0 <= float(phase) < 1:
            events.append(Event(float(phase), side, "TO", provenance, source))

    if show_raw_crossings:
        for r in raw:
            # Keep diagnostics visually distinct; never present these as selected.
            if r["kind"] == "HS" and any(e.kind == "HS" and e.side == r["side"] and
                    abs(e.phase - r["phase"]) * ref.T < 0.03 for e in events):
                continue
            events.append(Event(float(r["phase"] % 1.0), r["side"], r["kind"],
                                "candidate", str(r["detail"])))
    events.sort(key=lambda e: (e.phase, e.side, e.kind))
    return events


def model_joint_indices(model, mujoco):
    result = []
    for title, name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"MyoLeg joint {name!r} not found. Check XML joint names.")
        qidx = int(model.jnt_qposadr[jid])
        vidx = int(model.jnt_dofadr[jid])
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected a hinge joint for {name}")
        result.append((title, qidx, vidx))
    return result


def cycle_phase(ref: Reference, n_cycles: int) -> np.ndarray:
    """Stride phase on an x-axis spanning n_cycles (cycle k covers [k, k+1])."""
    base = ref.t / ref.T
    return np.concatenate([k + base for k in range(n_cycles)])


def repeat_cycles(values: np.ndarray, n_cycles: int) -> np.ndarray:
    """Tile one cycle's samples n_cycles times (works for 1-D and 2-D arrays)."""
    return np.concatenate([values] * n_cycles, axis=0)


def event_marks(e: Event, n_cycles: int) -> list[float]:
    """x-positions for an event across all plotted cycles.

    Right HS at phase 0 is marked at both edges of every cycle.
    """
    offsets = ((0.0, 1.0) if e.phase == 0.0 and e.kind == "HS" and e.side == "right"
               else (e.phase,))
    return [k + o for k in range(n_cycles) for o in offsets]


def stacked_joint(ref: Reference, qid: int, vid: int, n_cycles: int):
    """Angle and velocity (degrees) for one joint over n_cycles, each step's own values."""
    angles, vels = [], []
    for c in range(n_cycles):
        cq, cv = ref.cycle_arrays(c)
        angles.append(np.rad2deg(cq[:, qid]))
        vels.append(np.rad2deg(cv[:, vid]))
    return np.concatenate(angles), np.concatenate(vels)


def graph_window(ref: Reference, joints, events, output: Path, n_cycles: int = 1):
    """Create an angle/velocity dashboard after playback (no live updates)."""
    import matplotlib.pyplot as plt
    phase = cycle_phase(ref, n_cycles)
    fig, axes = plt.subplots(len(joints), 2, figsize=(14, 13), sharex=True,
                             constrained_layout=True)
    event_seen = set()
    for row, (label, qid, vid) in enumerate(joints):
        for col, values in enumerate(stacked_joint(ref, qid, vid, n_cycles)):
            ax = axes[row, col]
            ax.plot(phase, values, lw=1.5, color="tab:blue")
            ax.set_ylabel(("Angle (°)" if col == 0 else "Velocity (°/s)"), fontsize=9)
            ax.set_title(f"{label} — {'angle' if col == 0 else 'angular velocity'}", fontsize=10)
            ax.grid(alpha=0.2)
            for e in events:
                for point in event_marks(e, n_cycles):
                    ax.axvline(point, color=e.color, lw=1, ls="--", alpha=0.75)
                event_seen.add((e.label, e.color))
            ax.set_xlim(0, n_cycles)
    xlabel = ("Stride phase (right HS = 0/1)" if n_cycles == 1
              else f"Stride phase over {n_cycles} cycles (right HS = integers)")
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=color, lw=2, ls="--", label=label)
              for label, color in sorted(event_seen)]
    if legend:
        fig.legend(handles=legend, loc="upper center", ncol=min(5, len(legend)),
                   bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.suptitle(f"{ref.path.stem} — joint kinematics & gait events"
                 + (f"  (seam {ref.seam:.3f})" if ref.seam is not None else ""),
                 fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Saved graph: {output}")
    return fig


def load_stored_sites(ref: Reference):
    """Load stored pelvis-relative positions, quaternions and velocities."""
    with np.load(ref.path, allow_pickle=False) as f:
        required = ("site_names", "site_pos_rel", "site_quat_rel",
                    "site_linvel_rel", "site_angvel_rel")
        if any(k not in f for k in required):
            print("NOTE: complete stored site kinematics unavailable")
            return None
        names = [str(x) for x in f["site_names"]]
        pos = np.asarray(f["site_pos_rel"], dtype=float)
        quat = np.asarray(f["site_quat_rel"], dtype=float)
        lin = np.asarray(f["site_linvel_rel"], dtype=float)
        ang = np.asarray(f["site_angvel_rel"], dtype=float)
    shape = (len(ref.t), len(names), 3)
    if pos.shape != shape or lin.shape != shape or ang.shape != shape or quat.shape != shape[:2] + (4,):
        raise ValueError("Invalid stored site array dimensions")
    return names, pos, quat, lin, ang


def quaternion_rpy_degrees(quats):
    """Convert stored MuJoCo wxyz quaternions to XYZ roll/pitch/yaw (degrees)."""
    q = np.asarray(quats, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.rad2deg(np.stack((roll, pitch, yaw), axis=-1))


def landmark_arrays(ref: Reference, model, mujoco, stored, noisy: bool, n_cycles: int = 1):
    """Pelvis-frame landmark positions/velocities from MuJoCo FK on the current qpos/qvel.

    This follows the joints actually replayed (including any --noise-deg), so the
    plots match the markers in the viewer. Positions are in the pelvis body frame;
    velocities are the linear velocity of each site relative to the pelvis origin,
    also in the pelvis body frame. When the joints are not noised, the result is
    compared with the stored arrays as a convention check.
    """
    if stored is None:
        print("NOTE: site_* data not present; skipping pelvis-relative landmark plot")
        return None
    names, stored_pos, stored_quat, stored_vel, stored_ang = stored
    n_frames = len(ref.t)
    total = n_cycles * n_frames
    positions = np.zeros((total, len(names), 3))
    velocities = np.zeros_like(positions)
    orientations = np.full_like(positions, np.nan)  # roll, pitch, yaw in degrees
    angular_velocities = np.full_like(positions, np.nan)  # pelvis-frame rad/s
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rows = []
    for j, name in enumerate(names):
        if name == "pelvis":
            continue
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            print(f"WARNING: site {name!r} not in model; plotting its stored values instead")
            positions[:, j] = np.tile(stored_pos[:, j], (n_cycles, 1))
            velocities[:, j] = np.tile(stored_vel[:, j], (n_cycles, 1))
            continue
        rows.append((j, sid))

    data = mujoco.MjData(model)
    for c in range(n_cycles):
        cq, cv = ref.cycle_arrays(c)
        for k in range(n_frames):
            r = c * n_frames + k
            data.qpos[:] = cq[k]
            data.qvel[:] = cv[k]
            mujoco.mj_forward(model, data)
            rot = data.xmat[pelvis].reshape(3, 3)
            origin = data.xpos[pelvis].copy()
            pel_v = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, pel_v, 1)
            for j, sid in rows:
                positions[r, j] = rot.T @ (data.site_xpos[sid] - origin)
                sv = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, sv, 1)
                velocities[r, j] = rot.T @ (sv[3:] - pel_v[3:])
                relative_rot = rot.T @ data.site_xmat[sid].reshape(3, 3)
                # Intrinsic XYZ (roll/pitch/yaw), equivalent to Rz(yaw) Ry(pitch) Rx(roll).
                pitch = np.arcsin(np.clip(-relative_rot[2, 0], -1.0, 1.0))
                roll = np.arctan2(relative_rot[2, 1], relative_rot[2, 2])
                yaw = np.arctan2(relative_rot[1, 0], relative_rot[0, 0])
                orientations[r, j] = np.rad2deg([roll, pitch, yaw])
                angular_velocities[r, j] = rot.T @ (sv[:3] - pel_v[:3])

    if not noisy and rows:
        # Convention check on the nominal (unperturbed) first step only.
        idx = [j for j, _ in rows]
        dp = np.abs(positions[:n_frames, idx] - stored_pos[:, idx]).max()
        dv = np.abs(velocities[:n_frames, idx] - stored_vel[:, idx]).max()
        status = "OK" if max(dp, dv) < 1e-3 else "MISMATCH (check frame convention)"
        print(f"Landmark check vs stored arrays: max |dpos|={dp:.2e} m, max |dvel|={dv:.2e} m/s  [{status}]")
    return names, positions, velocities, orientations, angular_velocities


def save_site_graphs(ref: Reference, events, output: Path, names, positions, velocities,
                     n_cycles: int = 1):
    """Save pelvis-relative site trajectories (computed from the model, see landmark_arrays).

    These are in the pelvis BODY frame, so x/y/z are body-frame axes rather
    than automatically world forward/lateral/vertical directions.
    """
    if positions.shape != velocities.shape or positions.shape != (n_cycles * len(ref.t), len(names), 3):
        print("WARNING: malformed site position/velocity arrays; site plot not generated")
        return
    import matplotlib.pyplot as plt
    # The first entry is the zero-relative pelvis body origin, not a site.
    indices = [i for i, name in enumerate(names) if name != "pelvis"]
    if not indices:
        return
    fig, axs = plt.subplots(len(indices), 2, figsize=(15, 2.15 * len(indices)),
                            sharex=True, constrained_layout=True)
    axs = np.atleast_2d(axs)
    phase = cycle_phase(ref, n_cycles)
    for row, i in enumerate(indices):
        for col, values in enumerate((positions[:, i, :], velocities[:, i, :])):
            ax = axs[row, col]
            for dim, axis in enumerate("xyz"):
                ax.plot(phase, values[:, dim], lw=1.15, label=axis)
            for e in events:
                for point in event_marks(e, n_cycles):
                    ax.axvline(point, color=e.color, ls="--", lw=0.8, alpha=0.5)
            ax.set_title(f"{names[i]} — {'position' if col == 0 else 'linear velocity'}", fontsize=10)
            ax.set_ylabel("Pelvis-relative m" if col == 0 else "Pelvis-relative m/s", fontsize=9)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, n_cycles)
            if row == 0:
                ax.legend(loc="best", fontsize=8, ncol=3)
    xlabel = ("Stride phase (right HS = 0/1)" if n_cycles == 1
              else f"Stride phase over {n_cycles} cycles (right HS = integers)")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f"{ref.path.stem} — pelvis-frame landmark positions and velocities", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved site graph: {output}")
    return fig



def save_site_orientation_graphs(ref: Reference, events, output: Path, names,
                                 orientations, angular_velocities, n_cycles: int = 1):
    """Plot site orientation and angular velocity relative to the pelvis frame."""
    import matplotlib.pyplot as plt
    indices = [i for i, name in enumerate(names)
               if name != "pelvis" and np.isfinite(orientations[:, i]).any()]
    if not indices:
        print("NOTE: no model sites with valid orientations; skipping orientation plots")
        return None
    fig, axs = plt.subplots(len(indices), 2, figsize=(15, 2.4 * len(indices)),
                            sharex=True, constrained_layout=True)
    axs = np.atleast_2d(axs)
    phase = cycle_phase(ref, n_cycles)
    for row, i in enumerate(indices):
        # Unwrap Euler angle branches for continuous visualizations.
        angles = np.rad2deg(np.unwrap(np.deg2rad(orientations[:, i, :]), axis=0))
        for col, values in enumerate((angles, angular_velocities[:, i, :])):
            ax = axs[row, col]
            for dim, axis in enumerate(("roll", "pitch", "yaw") if col == 0 else ("wx", "wy", "wz")):
                ax.plot(phase, values[:, dim], lw=1.15, label=axis)
            for e in events:
                for point in event_marks(e, n_cycles):
                    ax.axvline(point, color=e.color, ls="--", lw=0.8, alpha=0.5)
            ax.set_title(f"{names[i]} — {'orientation' if col == 0 else 'angular velocity'}", fontsize=10)
            ax.set_ylabel("Pelvis-relative degrees" if col == 0 else "Pelvis-relative rad/s", fontsize=9)
            ax.grid(alpha=0.2)
            ax.set_xlim(0, n_cycles)
            if row == 0:
                ax.legend(loc="best", fontsize=8, ncol=3)
    xlabel = ("Stride phase (right HS = 0/1)" if n_cycles == 1
              else f"Stride phase over {n_cycles} cycles (right HS = integers)")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    fig.suptitle(f"{ref.path.stem} — pelvis-frame site orientations and angular velocities", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved site orientation graph: {output}")
    return fig


def visual_marker_sites(model, mujoco):
    """Preserve original RCAL/LCAL/RTOE/LTOE event markers where available."""
    names = {
        ("right", "HS"): ("RCAL", "heel_r", "RHEE"),
        ("left", "HS"): ("LCAL", "heel_l", "LHEE"),
        ("right", "TO"): ("RTOE", "toe_r"),
        ("left", "TO"): ("LTOE", "toe_l"),
    }
    ids = {}
    for key, options in names.items():
        ids[key] = next((sid for n in options
                         if (sid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)) >= 0), -1)
    missing = [f"{side} {kind}" for (side, kind), sid in ids.items() if sid < 0]
    if missing:
        print("WARNING: event spheres unavailable for " + ", ".join(missing)
              + "; event lines and logs still work")
    return ids


def draw_event_spheres(viewer, model, data, mujoco, event_sites, events, phase, held, flash_s, T):
    viewer.user_scn.ngeom = 0
    for e in events:
        dist = abs((phase - e.phase + 0.5) % 1 - 0.5) * T
        if dist > flash_s and e != held:
            continue
        sid = event_sites[(e.side, e.kind)]
        if sid < 0 or viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            continue
        pos = np.array(data.site_xpos[sid], dtype=float, copy=True)
        pos[2] += 0.12
        rgb = ({"measured": [0.07, 0.9, 0.35], "corroborated": [0.07, 0.9, 0.35],
                "candidate": [0.95, 0.18, 0.5]}.get(e.provenance, [1.0, 0.65, 0.1]))
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.array([0.065, 0, 0]), pos, np.eye(3).ravel(),
                           np.array([*rgb, 0.9], dtype=float))
        viewer.user_scn.ngeom += 1


def next_event_after(current_s, T, events):
    """Return the next selected event in unwrapped physical time."""
    cycle = math.floor(current_s / T)
    candidates = [(k * T + e.phase * T, e)
                  for k in (cycle, cycle + 1) for e in events
                  if k * T + e.phase * T > current_s + 1e-8]
    return min(candidates, key=lambda z: z[0]) if candidates else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True, help="Step-2 *_loopable.npz")
    ap.add_argument("--model", type=Path, help="MyoLeg XML or MJB; required for replay/plots")
    ap.add_argument("--output", type=Path, default=Path("gait_replay_plots"), help="PNG output folder")
    ap.add_argument("--cycles", type=int, default=3, help="Replay cycles; 0 = continuous")
    ap.add_argument("--speed", type=float, default=0.5, help="Playback speed multiplier")
    ap.add_argument("--hold", type=float, default=0.5, help="Wall-time pause at selected events, seconds")
    ap.add_argument("--flash", type=float, default=0.07, help="Event marker visibility in seconds")
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--all-cycles", action="store_true",
                    help="Graphs show every replayed cycle (--cycles) instead of one stride")
    ap.add_argument("--noise-deg", type=float, default=0.0,
                    help="Std of white noise added to joint angles, degrees (default 0 = off)")
    ap.add_argument("--noise-seed", type=int, default=0, help="RNG seed for --noise-deg")
    ap.add_argument("--noise-cutoff-hz", type=float, default=0.0,
                    help="0 = white noise (default); >0 = low-pass noise with this cutoff in Hz "
                         "(must be >= 1/stride duration)")
    plot_mode = ap.add_mutually_exclusive_group()
    plot_mode.add_argument("--plots-at-end", dest="plot_mode", action="store_const",
                           const="end", default="end",
                           help="Do not create/update Matplotlib during replay; save and display plots when playback ends (default)")
    plot_mode.add_argument("--no-plots", dest="plot_mode", action="store_const",
                           const="none", help="Do not create or display any plots")
    ap.add_argument("--viewer-fps", type=float, default=60,
                    help="Upper bound on viewer refresh rate (default: 60; try 30 if viewer is slow)")
    ap.add_argument("--no-show-plots", dest="no_show_plots",
                    action="store_true", help="Save PNGs but do not open the plots after replay")
    ap.add_argument("--plots-only", action="store_true", help="Save graphs, skip MuJoCo viewer")
    ap.add_argument("--events-only", action="store_true", help="List events, no model needed")
    ap.add_argument("--show-raw-crossings", action="store_true", help="Include unselected GRF crossings as diagnostic candidates")
    args = ap.parse_args()
    if args.speed <= 0 or args.cycles < 0 or args.hold < 0 or args.flash <= 0 or args.viewer_fps <= 0:
        ap.error("Require --speed > 0, --cycles >= 0, --hold >= 0, --flash > 0, --viewer-fps > 0")
    if args.noise_deg < 0 or args.noise_cutoff_hz < 0:
        ap.error("Require --noise-deg >= 0 and --noise-cutoff-hz >= 0")
    if args.all_cycles and args.cycles == 0:
        ap.error("--all-cycles needs a finite --cycles count (--cycles 0 is continuous)")
    plot_cycles = args.cycles if args.all_cycles else 1

    ref = Reference(args.reference)
    events = get_events(ref, show_raw_crossings=args.show_raw_crossings)
    print(f"\nReference: {ref.path} | T={ref.T:.3f} s | forward displacement={ref.displacement[1]:.3f} m")
    for e in events:
        print(f"  phase={e.phase:6.3f} ({e.phase * ref.T:.3f} s) {e.label}: {e.detail}")
    print("NOTE: events are annotations, not simulated contacts; assumed TO stays assumed.")
    if args.events_only:
        return

    if args.model is None:
        ap.error("--model is required except for --events-only")
    import mujoco
    model = (mujoco.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower() == ".mjb"
             else mujoco.MjModel.from_xml_path(str(args.model)))
    if ref.qpos.shape[1] != model.nq or ref.qvel.shape[1] != model.nv:
        raise ValueError(f"Reference qpos/qvel {ref.qpos.shape}/{ref.qvel.shape} but model nq/nv={model.nq}/{model.nv}")
    joints = model_joint_indices(model, mujoco)
    noisy = args.noise_deg > 0
    if noisy:
        if 0 < args.noise_cutoff_hz < 1 / ref.T:
            ap.error(f"--noise-cutoff-hz must be >= 1/T = {1 / ref.T:.3f} Hz for this stride")
        # Applied once, so replay and graphs show the same noisy kinematics.
        ref.add_joint_noise([q for _, q, _ in joints], [v for _, _, v in joints],
                            math.radians(args.noise_deg), seed=args.noise_seed,
                            cutoff_hz=args.noise_cutoff_hz)
        kind = (f"low-pass (cutoff {args.noise_cutoff_hz:.2f} Hz)" if args.noise_cutoff_hz > 0
                else "white")
        print(f"Added {kind} noise to joint angles: std={args.noise_deg:.2f} deg "
              f"(seed {args.noise_seed})")
    stored_sites = load_stored_sites(ref)

    def finish_and_plot(show: bool):
        # Deliberately create figures AFTER playback so Matplotlib cannot slow MuJoCo.
        import matplotlib.pyplot as plt
        angle_fig = graph_window(
            ref, joints, events,
            args.output / f"{ref.path.stem}_angles_velocities.png",
            n_cycles=plot_cycles)
        if stored_sites is not None and not noisy:
            names, pos, quat, lin, ang = stored_sites
            sites = (names, repeat_cycles(pos, plot_cycles),
                     repeat_cycles(lin, plot_cycles),
                     repeat_cycles(quaternion_rpy_degrees(quat), plot_cycles),
                     repeat_cycles(ang, plot_cycles))
            print("Site plots use the stored NPZ kinematics (no FK recomputation).")
        else:
            sites = landmark_arrays(ref, model, mujoco, stored_sites, noisy,
                                   n_cycles=plot_cycles)
        site_fig = None
        orientation_fig = None
        if sites is not None:
            site_fig = save_site_graphs(
                ref, events, args.output / f"{ref.path.stem}_sites_positions_velocities.png",
                *sites[:3], n_cycles=plot_cycles)
            orientation_fig = save_site_orientation_graphs(
                ref, events, args.output / f"{ref.path.stem}_sites_orientations_angular_velocities.png",
                sites[0], sites[3], sites[4], n_cycles=plot_cycles)
        if show:
            print("Playback ended. Showing saved joint and landmark plots.")
            plt.show(block=True)
        plt.close(angle_fig)
        if site_fig is not None:
            plt.close(site_fig)
        if orientation_fig is not None:
            plt.close(orientation_fig)

    if args.plots_only:
        if args.plot_mode == "end":
            finish_and_plot(show=not args.no_show_plots)
        return

    import glfw
    import mujoco.viewer
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("pelvis body not found")
    sites = visual_marker_sites(model, mujoco)
    data = mujoco.MjData(model)
    state = {"t": 0.0, "paused": False, "hold_until": 0.0, "held": None, "quit": False}
    max_time = args.cycles * ref.T if args.cycles else math.inf

    def key_callback(key):
        if key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            state["hold_until"] = 0.0
        elif key == glfw.KEY_R:
            state.update(t=0.0, paused=False, held=None, hold_until=0.0)
        elif key == glfw.KEY_Q:
            state["quit"] = True
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            state["paused"] = True
            delta = ref.T / (len(ref.t) - 1)
            state["t"] = max(0.0, min(max_time, state["t"] + (-delta if key == glfw.KEY_LEFT else delta)))
            state["held"] = None
            state["hold_until"] = 0.0

    q, v, _ = ref.sample(0.0)
    data.qpos[:] = q
    data.qvel[:] = v
    mujoco.mj_forward(model, data)
    print("Viewer controls: SPACE pause | R restart | LEFT/RIGHT one sample | Q quit")
    if args.plot_mode == "end":
        print("Offline plots: figures will be saved and displayed ONLY after the viewer closes.")
    else:
        print("Plotting disabled: MuJoCo viewer only.")
    print("No physics stepping or muscle control.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.cam.distance = 2.7
            viewer.cam.elevation = -14
            viewer.cam.azimuth = 90
        last_wall = time.perf_counter()
        while viewer.is_running() and not state["quit"]:
            now = time.perf_counter()
            dt_wall = min(now - last_wall, 0.12)
            last_wall = now

            if not state["paused"]:
                if now < state["hold_until"]:
                    pass
                else:
                    state["held"] = None
                    proposed = min(max_time, state["t"] + dt_wall * args.speed)
                    upcoming = next_event_after(state["t"], ref.T, events)
                    if upcoming is not None and upcoming[0] <= proposed + 1e-9:
                        state["t"], e = upcoming
                        if e.provenance != "candidate":
                            state["held"] = e
                            state["hold_until"] = now + args.hold
                        print(f"Cycle {math.floor(state['t']/ref.T) + 1}: {e.label} at phase {e.phase:.3f}")
                    else:
                        state["t"] = proposed
                    if state["t"] >= max_time - 1e-9:
                        # The viewer closes automatically after the requested cycles;
                        # plots open immediately afterward.
                        print("Reached requested number of cycles.")

            q, v, phase = ref.sample(state["t"])
            with viewer.lock():
                data.qpos[:] = q
                data.qvel[:] = v
                data.time = state["t"]
                mujoco.mj_forward(model, data)
                if not args.no_follow:
                    viewer.cam.lookat[:] = data.xpos[pelvis_id]
                draw_event_spheres(viewer, model, data, mujoco, sites, events,
                                   phase, state["held"], args.flash, ref.T)
            viewer.sync()
            if args.cycles > 0 and state["t"] >= max_time - 1e-9:
                break
            time.sleep(1 / args.viewer_fps)

    # Also executes when the user quits with Q or closes the viewer.
    # Neither matplotlib nor graph generation runs inside the playback loop.
    if args.plot_mode == "end":
        finish_and_plot(show=not args.no_show_plots)


if __name__ == "__main__":
    main()