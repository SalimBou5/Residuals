#!/usr/bin/env python3
"""STEP 1 ONLY: turn measured right-HS -> right-HS MyoLeg strides into loopable references.

Dependencies: numpy scipy matplotlib (no MuJoCo XML required).
Inputs: standardized stepXX.npz with qpos, time_since_start_s, stride_duration_s.
Outputs: corrected qpos/qvel, exact stride displacement, comparison plot.
No site tracking, contacts, muscle activation, or model-based validation here.

Usage:
    python make_gait_loopable.py --input references_gait120_standardized_adjusted_height --output loopable_references
    python make_gait_loopable.py --input path/to/step01.npz --output loopable_demo
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import shutil

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wxyz_to_xyzw(q):
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def xyzw_to_wxyz(q):
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def build_loop(raw_q: np.ndarray, time: np.ndarray, seam: float):
    """Return sample(unwrapped_time) and the stride XY displacement.

    A corrected cyclic signal is made from root-height, root-orientation,
    scalar joints, and *root XY minus linear travel*. That linear travel is
    restored during sampling, including at t > stride_duration.

    Only the first and last `seam` fraction of measured time are changed.
    Inside the middle interval the original measured qpos samples are exact.
    """
    if not 0 < seam < 0.5:
        raise ValueError("--seam must be strictly between 0 and 0.5")
    if raw_q.ndim != 2 or raw_q.shape[1] < 8:
        raise ValueError("Expected [frames, nq] qpos with a free root in qpos[:7]")
    if len(time) != len(raw_q) or len(time) < 5:
        raise ValueError("qpos/time lengths do not match, or too few frames")
    if not np.all(np.isfinite(raw_q)) or not np.all(np.isfinite(time)):
        raise ValueError("Input contains NaN or inf")
    if not np.isclose(time[0], 0) or np.any(np.diff(time) <= 0):
        raise ValueError("time_since_start_s must start at zero and be strictly increasing")

    T = float(time[-1])
    width = seam * T
    if width < 2 * float(np.median(np.diff(time))):
        raise ValueError("Seam window is too short for this sampling rate")

    displacement = raw_q[-1, :2] - raw_q[0, :2]

    # Convert root quaternion to a locally smooth 3-vector for interpolation.
    root_rot = Rotation.from_quat(wxyz_to_xyzw(raw_q[:, 3:7]))
    root0 = root_rot[0]
    rel_rotvec = (root0.inv() * root_rot).as_rotvec()
    if np.max(np.linalg.norm(rel_rotvec, axis=1)) >= np.pi / 2:
        raise ValueError("Root orientation changes too much for this local rotvec representation")

    # [root xyz periodic residual, root rotation vector, scalar joint coordinates]
    x = np.column_stack([raw_q[:, :3], rel_rotvec, raw_q[:, 7:]])
    x[:, :2] -= (time / T)[:, None] * displacement
    measured = PchipInterpolator(time, x, axis=0)

    # Target endpoint position/slope: midpoint of the two measured endpoints.
    x0, xT = x[0], x[-1]
    v0, vT = measured.derivative()(np.array([0.0, T]))
    x_target = 0.5 * (x0 + xT)
    v_target = 0.5 * (v0 + vT)

    def periodic_residual(local_t):
        local_t = np.asarray(local_t, dtype=float)
        y = measured(local_t)
        s0 = np.clip(local_t / width, 0, 1)
        sT = np.clip((T - local_t) / width, 0, 1)

        # Quintic Hermite endpoint corrections; both correction and first two
        # derivatives vanish at the edges of the seam window.
        def value_basis(s):
            return 1 - 10*s**3 + 15*s**4 - 6*s**5

        def slope_basis(s):
            return s - 6*s**3 + 8*s**4 - 3*s**5

        correction0 = (value_basis(s0)[..., None] * (x_target - x0)
                       + width*slope_basis(s0)[..., None] * (v_target - v0))
        correctionT = (value_basis(sT)[..., None] * (x_target - xT)
                       - width*slope_basis(sT)[..., None] * (v_target - vT))
        return y + correction0 + correctionT

    def sample(unwrapped_t):
        """Return qpos at any time, even before 0 or after multiple cycles."""
        a = np.atleast_1d(np.asarray(unwrapped_t, dtype=float))
        local_t = np.mod(a, T)
        z = periodic_residual(local_t)
        q = np.empty((len(a), raw_q.shape[1]))
        q[:, :3] = z[:, :3]
        # CRITICAL: continuous drift, *not* only a jump once per cycle.
        q[:, :2] += (a / T)[:, None] * displacement
        q[:, 3:7] = xyzw_to_wxyz((root0 * Rotation.from_rotvec(z[:, 3:6])).as_quat())
        q[:, 7:] = z[:, 6:]
        return q[0] if np.ndim(unwrapped_t) == 0 else q

    return sample, displacement


def differentiate_positions(sample, times: np.ndarray, nq: int):
    """Numerical derivatives with quaternion-safe LOCAL root angular velocity.

    For MyoLeg's single free root + hinge joints, qvel has nq-1 entries.
    The model is not required for this purely kinematic preparation step.
    """
    dt = min(1e-4, float(np.min(np.diff(times))) / 20)
    q_prev = sample(times - dt)
    q_next = sample(times + dt)
    qvel = np.empty((len(times), nq - 1))
    qvel[:, :3] = (q_next[:, :3] - q_prev[:, :3]) / (2 * dt)
    Rprev = Rotation.from_quat(wxyz_to_xyzw(q_prev[:, 3:7]))
    Rnext = Rotation.from_quat(wxyz_to_xyzw(q_next[:, 3:7]))
    qvel[:, 3:6] = (Rprev.inv() * Rnext).as_rotvec() / (2 * dt)
    qvel[:, 6:] = (q_next[:, 7:] - q_prev[:, 7:]) / (2 * dt)
    return qvel


def comparison_plot(path: Path, t: np.ndarray, raw: np.ndarray, prepared: np.ndarray,
                    sampler, displacement: np.ndarray, width_frac: float):
    T = float(t[-1])
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    # MyoLeg qpos positions, as in standardized references.
    joints = [
        (7, "Right hip flexion"), (21, "Left hip flexion"),
        (12, "Right knee"), (26, "Left knee"),
        (15, "Right ankle"), (29, "Left ankle"),
    ]
    for ax, (col, name) in zip(axes.flat, joints):
        if col >= raw.shape[1]:
            ax.set_visible(False)
            continue
        ax.plot(t/T, np.rad2deg(raw[:, col]), color="0.5", lw=1.7, label="Measured")
        ax.plot(t/T, np.rad2deg(prepared[:, col]), color="tab:blue", lw=1.6, label="Loopable")
        ax.axvspan(0, width_frac, color="tab:orange", alpha=0.1)
        ax.axvspan(1-width_frac, 1, color="tab:orange", alpha=0.1)
        ax.set(title=name, xlabel="Stride phase (right HS = 0/1)", ylabel="Angle (deg)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")

    # Confirm pelvis keeps moving forward even after the first loop.
    ax = axes[3, 0]
    all_t = np.linspace(0, 3*T, 3*300+1)
    looped = sampler(all_t)
    orig_y = np.interp(np.mod(all_t, T), t, raw[:, 1]) + np.floor(all_t/T)*displacement[1]
    ax.plot(all_t/T, orig_y, color="0.5", label="Measured repeated", lw=1.6)
    ax.plot(all_t/T, looped[:, 1], color="tab:blue", label="Loopable repeated", lw=1.6)
    for k in (1, 2):
        ax.axvline(k, color="k", lw=0.7, ls="--")
    ax.set(title="Pelvis forward displacement over 3 strides", xlabel="Stride number", ylabel="World Y (m)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)

    ax = axes[3, 1]
    for col, name in [(12, "Right knee"), (26, "Left knee")]:
        if col < raw.shape[1]:
            ax.plot(t/T, np.rad2deg(prepared[:, col] - raw[:, col]), label=name, lw=1.4)
    ax.axvspan(0, width_frac, color="tab:orange", alpha=0.1)
    ax.axvspan(1-width_frac, 1, color="tab:orange", alpha=0.1)
    ax.set(title="Applied knee correction (should be zero in the middle)",
           xlabel="Stride phase", ylabel="Correction (deg)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.suptitle(path.stem + " — measured vs loopable (orange = correction windows)", fontsize=15)
    return fig


def process_file(npz_path: Path, outdir: Path, seam: float, sample_hz: float):
    with np.load(npz_path, allow_pickle=False) as d:
        raw_q = np.asarray(d["qpos"], dtype=float)
        raw_t = np.asarray(d["time_since_start_s"], dtype=float)
        T = float(d["stride_duration_s"])
        if not np.isclose(raw_t[-1], T, atol=1e-4):
            raise ValueError(f"Duration mismatch in {npz_path}")
        if raw_q.shape[1] != 35:
            raise ValueError("This first-step script expects MyoLeg nq=35")
        # Event timestamps are unchanged. Samplewise left-phase arrays from the
        # original 100-Hz grid are NOT copied onto the new 200-Hz grid.
        events = {k: np.asarray(d[k]).copy() for k in ("left_hs_s",) if k in d}

    sample, disp = build_loop(raw_q, raw_t, seam)
    n = max(2, int(np.ceil(T * sample_hz)))
    new_t = np.linspace(0, T, n+1)
    new_q = sample(new_t)
    new_v = differentiate_positions(sample, new_t, raw_q.shape[1])
    new_at_raw = sample(raw_t)
    # Invariant: samples outside seam windows remain EXACTLY measured.
    interior = (raw_t > seam*T + 1e-8) & (raw_t < (1-seam)*T - 1e-8)
    max_interior_error = float(np.max(np.abs(new_at_raw[interior, 7:] - raw_q[interior, 7:])))
    boundary_q = new_q[-1].copy()
    boundary_q[:2] -= disp
    max_joint_correction = float(np.max(np.abs(new_at_raw[:, 7:]-raw_q[:, 7:])))
    qvel_seam_error = float(np.max(np.abs(new_v[0]-new_v[-1])))
    if not np.allclose(boundary_q[:3], new_q[0, :3], atol=1e-8) or max_interior_error > 1e-8:
        raise RuntimeError("Loop/invariance check failed")

    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / f"{npz_path.stem}_loopable.npz"
    plot_path = outdir / f"{npz_path.stem}_comparison.png"
    np.savez_compressed(
        target,
        qpos=new_q, qvel=new_v,
        qpos_ref=new_q, qvel_ref=new_v,
        time_since_start_s=new_t, phase_right_01=new_t/T,
        stride_duration_s=np.array(T),
        root_displacement_xyz_m=np.array([disp[0], disp[1], 0.0]),
        forward_speed_m_s=np.array(disp[1]/T),
        seam_fraction=np.array(seam),
        **events,
    )
    fig = comparison_plot(npz_path, raw_t, raw_q, new_at_raw, sample, disp, seam)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    # Keep event provenance verbatim as companion files. No event is inferred.
    for suffix in ("_events.csv", "_manifest.json"):
        src = npz_path.with_name(npz_path.stem + suffix)
        if src.is_file():
            shutil.copyfile(src, outdir / (npz_path.stem + "_loopable" + suffix))
    details = {
        "source": str(npz_path), "output": str(target), "plot": str(plot_path),
        "stride_duration_s": T, "root_displacement_xy_m": disp.tolist(),
        "max_joint_correction_deg": float(np.rad2deg(max_joint_correction)),
        "max_interior_joint_error_rad": max_interior_error,
        "qvel_boundary_jump_max_per_s": qvel_seam_error,
        "note": "Kinematic preparation only. MuJoCo joint equality/contact validation is a later step.",
    }
    (outdir / f"{npz_path.stem}_diagnostics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="One standardized stepXX.npz OR their root folder")
    p.add_argument("--output", type=Path, default=Path("loopable_references"))
    p.add_argument("--seam", type=float, default=0.10, help="Fraction of stride modified at EACH end (default 0.10)")
    p.add_argument("--sample-hz", type=float, default=200.0)
    args = p.parse_args()
    if args.input.is_file():
        files = [args.input]
        base = args.input.parent
    else:
        base = args.input
        files = sorted(f for f in base.rglob("step*.npz") if f.stem[4:].isdigit())
    if not files:
        p.error("No stepXX.npz files found")
    summary = []
    for file in files:
        target_dir = args.output / file.parent.relative_to(base)
        try:
            detail = process_file(file, target_dir, args.seam, args.sample_hz)
            summary.append(detail)
            print(f"OK {file}: correction={detail['max_joint_correction_deg']:.2f} deg "
                  f"displacement={detail['root_displacement_xy_m'][1]:.3f} m")
        except Exception as exc:
            print(f"ERROR {file}: {exc}")
            summary.append({"source": str(file), "error": str(exc)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed {sum('error' not in x for x in summary)}/{len(files)} strides; output: {args.output}")


if __name__ == "__main__":
    main()
