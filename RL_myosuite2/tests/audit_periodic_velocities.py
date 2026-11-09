#!/usr/bin/env python3
"""Audit stored MyoLeg loopable qvel against a seam-aware qpos derivative.

Diagnostic ONLY: never edits the source NPZs or the in-memory RL references.

The original loopable references include BOTH t=0 and t=T. Their world-root
positions differ by one stride displacement; the last stored frame is NOT a
separate sample after t=0. For the derivative at the seam, use the preceding
frame shifted one cycle BACKWARD, and the first next frame shifted one cycle
FORWARD, with mj_differentiatePos to handle free-root quaternion rotations.

Usage:
  python audit_periodic_velocities.py --model myolegs.xml \
      --references gait_with_landmarks --output velocity_audit

For a quick audit add --max-files 15. For representative plots use --plots 8.
This is a velocity-CONSISTENCY audit, not an automatic biological filter.
"""
from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
import numpy as np


def periodic_neighbors(qpos: np.ndarray, times: np.ndarray, displacement: np.ndarray,
                       index: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return physical positions before/after sample i and their time separation.

    In particular, qpos[-1] == qpos[0] translated forward one stride, so
    applying a difference directly between those two endpoint rows is WRONG.
    """
    q = np.asarray(qpos, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    d = np.asarray(displacement, dtype=np.float64)
    n = len(t)
    if (q.ndim != 2 or n != len(q) or n < 4 or d.shape != (3,)
            or not 0 <= index < n or not np.all(np.diff(t) > 0)
            or not np.isclose(t[0], 0)):
        raise ValueError("Invalid reference sample/time/displacement")
    T = float(t[-1])
    if index == 0:
        before = q[-2].copy()
        before[:3] -= d
        after = q[1].copy()
        elapsed = (T - t[-2]) + t[1]
    elif index == n - 1:
        before = q[-2].copy()
        after = q[1].copy()
        after[:3] += d
        elapsed = (T - t[-2]) + t[1]
    else:
        before = q[index - 1].copy()
        after = q[index + 1].copy()
        elapsed = float(t[index + 1] - t[index - 1])
    if elapsed <= 0:
        raise ValueError("Nonpositive finite-difference time interval")
    return before, after, float(elapsed)


def estimate_velocities(mj, model, qpos, times, displacement):
    """Central/secant generalized velocities, respecting MuJoCo quaternion DoFs."""
    n = len(times)
    v = np.empty((n, model.nv), dtype=np.float64)
    for i in range(n):
        before, after, dt = periodic_neighbors(qpos, times, displacement, i)
        mj.mj_differentiatePos(model, v[i], dt, before, after)
    # The two stored endpoints represent the same periodic phase; rounding
    # differences in joint positions at T do not change the derived velocity.
    v[-1] = v[0]
    return v


def dof_names(mj, model):
    labels = [None] * model.nv
    for j in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        adr = int(model.jnt_dofadr[j])
        jtype = model.jnt_type[j]
        if jtype == mj.mjtJoint.mjJNT_FREE:
            suffixes = ("lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z")
        elif jtype == mj.mjtJoint.mjJNT_BALL:
            suffixes = ("ang_x", "ang_y", "ang_z")
        else:
            suffixes = ("velocity",)
        for off, suffix in enumerate(suffixes):
            labels[adr + off] = f"{name}/{suffix}"
    return [x or f"dof_{i}" for i, x in enumerate(labels)]


def save_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--references", required=True, type=Path)
    p.add_argument("--output", default=Path("velocity_audit"), type=Path)
    p.add_argument("--max-files", type=int, default=0, help="0=all references")
    p.add_argument("--plots", type=int, default=6, help="Number of plots for worst discrepancies; 0=none")
    p.add_argument("--joint-alert-rad-s", type=float, default=10.0)
    p.add_argument("--root-alert-m-s", type=float, default=3.0)
    args = p.parse_args()
    if args.max_files < 0 or args.plots < 0:
        p.error("--max-files and --plots must be nonnegative")

    import mujoco as mj
    model = (mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower() == ".mjb"
             else mj.MjModel.from_xml_path(str(args.model)))
    refs = ([args.references] if args.references.is_file()
            else sorted(args.references.rglob("*_loopable.npz")))
    if args.max_files:
        refs = refs[:args.max_files]
    if not refs:
        p.error("No references found")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = dof_names(mj, model)
    seam_rows, extreme_rows, summary = [], [], []
    plot_candidates = []
    for n_ref, path in enumerate(refs, 1):
        with np.load(path, allow_pickle=False) as d:
            q = np.asarray(d["qpos"], dtype=np.float64)
            v = np.asarray(d["qvel"], dtype=np.float64)
            t = np.asarray(d["time_since_start_s"], dtype=np.float64)
            disp = (np.asarray(d["root_displacement_xyz_m"], dtype=np.float64)
                    if "root_displacement_xyz_m" in d else q[-1, :3] - q[0, :3])
        if q.shape != (len(t), model.nq) or v.shape != (len(t), model.nv):
            raise ValueError(f"Incompatible model shape: {path}")
        recon = estimate_velocities(mj, model, q, t, disp)
        delta = np.abs(v - recon)
        seam_diff = np.maximum(delta[0], delta[-1])
        for j in range(model.nv):
            seam_rows.append(dict(reference=str(path), dof=j, name=labels[j],
                stored_at_start=float(v[0, j]), stored_at_end=float(v[-1, j]),
                reconstructed=float(recon[0, j]),
                absolute_difference=float(seam_diff[j])))
        # At each DOF, keep the most extreme stored velocity anywhere in the stride,
        # plus the estimated qpos-derived value at the exact same sample/time.
        for j in range(model.nv):
            i = int(np.argmax(np.abs(v[:, j])))
            threshold = args.root_alert_m_s if j < 3 else args.joint_alert_rad_s
            if abs(v[i, j]) >= threshold or abs(recon[i, j]) >= threshold:
                extreme_rows.append(dict(reference=str(path), dof=j, name=labels[j],
                    frame=i, phase=float(t[i] / t[-1]), stored=float(v[i, j]),
                    reconstructed=float(recon[i, j]),
                    absolute_difference=float(delta[i, j]),
                    estimate_also_large=bool(abs(recon[i, j]) >= threshold),
                    at_seam=bool(i in (0, len(t)-1))))
        worst = np.unravel_index(np.argmax(delta), delta.shape)
        summary.append(dict(reference=str(path), frames=len(t),
            duration_s=float(t[-1]), seam_max_abs_difference=float(np.max(seam_diff)),
            overall_max_abs_difference=float(delta[worst]),
            worst_dof=labels[worst[1]], worst_phase=float(t[worst[0]] / t[-1]),
            root_recorded_max_m_s=float(np.max(np.linalg.norm(v[:, :3], axis=1))),
            root_reconstructed_max_m_s=float(np.max(np.linalg.norm(recon[:, :3], axis=1)))))
        if args.plots:
            # Bounded top-N heap; do NOT retain the full 969-stride library
            # a second time just to make a few diagnostic figures.
            entry = (float(delta[worst]), n_ref, str(path), int(worst[1]),
                     t.copy(), v[:,worst[1]].copy(), recon[:,worst[1]].copy())
            if len(plot_candidates) < args.plots:
                heapq.heappush(plot_candidates, entry)
            elif entry[0] > plot_candidates[0][0]:
                heapq.heapreplace(plot_candidates, entry)
        if n_ref % 100 == 0:
            print(f"Audited {n_ref}/{len(refs)} references")

    save_csv(args.output / "seam_comparison.csv", seam_rows,
             ["reference", "dof", "name", "stored_at_start", "stored_at_end",
              "reconstructed", "absolute_difference"])
    save_csv(args.output / "velocity_extremes.csv", extreme_rows,
             ["reference", "dof", "name", "frame", "phase", "stored",
              "reconstructed", "absolute_difference", "estimate_also_large", "at_seam"])
    save_csv(args.output / "reference_summary.csv", summary,
             ["reference", "frames", "duration_s", "seam_max_abs_difference",
              "overall_max_abs_difference", "worst_dof", "worst_phase",
              "root_recorded_max_m_s", "root_reconstructed_max_m_s"])

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for rank, (_, _, path, dof, t, v, recon) in enumerate(
                sorted(plot_candidates, key=lambda item: item[0], reverse=True), 1):
            fig, ax = plt.subplots(figsize=(10, 3.8))
            ax.plot(t/t[-1], v, label="Stored qvel", lw=1.6)
            ax.plot(t/t[-1], recon, label="Seam-aware derivative from qpos", lw=1.3)
            ax.axvline(0, color="grey", linestyle=":")
            ax.axvline(1, color="grey", linestyle=":")
            ax.set(xlabel="Gait phase (RHS to RHS)", ylabel="Generalized velocity",
                   title=f"{Path(path).parent.parent.name}/{Path(path).parent.name}/{Path(path).stem} — {labels[dof]}")
            ax.legend()
            ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(args.output / f"velocity_comparison_{rank:02d}.png", dpi=145)
            plt.close(fig)
    seam_bad = sum(x["seam_max_abs_difference"] > 5 for x in summary)
    print(f"Audited {len(refs)} references. >5 SI-unit seam disagreements: {seam_bad}")
    print(f"Wrote {args.output}/seam_comparison.csv, velocity_extremes.csv, "
          f"reference_summary.csv and {min(args.plots, len(plot_candidates))} plots")
    print("No source/reference arrays were modified.")


if __name__ == "__main__":
    main()
