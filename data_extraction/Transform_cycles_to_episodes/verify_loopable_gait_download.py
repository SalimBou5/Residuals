#!/usr/bin/env python3
"""Verify STEP 1 output: three repeated strides + knee/ankle velocity at right HS.

Does NOT modify the references or reject strides. No MuJoCo required.

Examples:
  python verify_loopable_gait.py --input loopable_references \
      --source-root references_gait120_standardized_adjusted_height --output loop_verification --max-files 6
  python verify_loopable_gait.py --input loopable_references/S012/Trial01/step01_loopable.npz \
      --source-root references_gait120_standardized_adjusted_height --output loop_verification
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# Source MyoLeg scalar qpos index -> physical qvel index (free-root difference = 1).
JOINTS = [(12, "Right knee"), (26, "Left knee"),
          (15, "Right ankle"), (29, "Left ankle")]
DEG = 180.0 / np.pi


def three_cycles(t, x, T, displacement=None):
    """Repeat one inclusive-endpoint reference without duplicating junction samples.

    For root position arrays, provide one stride's displacement so that global
    translation continues on every repetition. Joint angles/velocities need none.
    """
    times = np.concatenate([t[:-1] + k*T for k in range(3)] + [t[-1:] + 2*T])
    repeat = []
    for k in range(3):
        block = x if k == 2 else x[:-1]
        block = np.asarray(block).copy()
        if displacement is not None:
            block += k * np.asarray(displacement)
        repeat.append(block)
    return times, np.concatenate(repeat, axis=0)


def source_file_for(processed: Path, source_root: Path | None) -> Path | None:
    if source_root is None:
        return None
    # Expected: loopable_references/Sxxx/Trialxx/stepyy_loopable.npz
    stem = processed.stem.removesuffix("_loopable")
    subj = next((part for part in processed.parts if part.startswith("S") and part[1:].isdigit()), None)
    trial = next((part for part in processed.parts if part.startswith("Trial") and part[5:].isdigit()), None)
    candidates = []
    if subj and trial:
        candidates.append(source_root / subj / trial / (stem + ".npz"))
    candidates.append(source_root / (stem + ".npz"))
    return next((p for p in candidates if p.is_file()), None)


def verify_one(processed: Path, output: Path, source: Path | None, zoom_fraction: float):
    with np.load(processed, allow_pickle=False) as d:
        q = np.array(d["qpos"], dtype=float)
        v = np.array(d["qvel"], dtype=float)
        t = np.array(d["time_since_start_s"], dtype=float)
        T = float(d["stride_duration_s"])
        disp = np.asarray(d["root_displacement_xyz_m"], dtype=float)
        seam = 0.025 #float(d["seam_fraction"]) if "seam_fraction" in d else 0.025
    if q.ndim != 2 or q.shape[1] != 35 or v.shape != (len(q), 34):
        raise ValueError("Expected MyoLeg qpos=(N,35), qvel=(N,34)")
    if len(t) != len(q) or len(t) < 6 or not np.all(np.diff(t) > 0):
        raise ValueError("Invalid sample time grid")
    if not np.isclose(t[-1], T):
        raise ValueError("Stride duration differs from time grid")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ValueError("Reference contains non-finite values")

    raw_q = raw_v = raw_t = None
    if source is not None:
        with np.load(source, allow_pickle=False) as d:
            raw_q = np.array(d["qpos"], dtype=float)
            raw_v = np.array(d["qvel"], dtype=float)
            raw_t = np.array(d["time_since_start_s"], dtype=float)
        if raw_q.shape[1] != 35 or raw_v.shape[1] != 34 or not np.isclose(raw_t[-1], T):
            raise ValueError("Original reference is incompatible with the processed reference")

    t3, q3 = three_cycles(t, q, T)
    _, v3 = three_cycles(t, v, T)
    if source is not None:
        raw_t3, raw_q3 = three_cycles(raw_t, raw_q, T)
        _, raw_v3 = three_cycles(raw_t, raw_v, T)

    fig, axs = plt.subplots(4, 2, figsize=(14, 12), constrained_layout=True)
    near = (t <= zoom_fraction*T) | (t >= (1-zoom_fraction)*T)
    corrected_region = (t <= seam*T + 1e-12) | (t >= (1-seam)*T - 1e-12)
    result = {
        "reference": str(processed), "original": str(source) if source else None,
        "stride_duration_s": T, "seam_fraction_each_end": seam,
        "sample_interval_s": float(np.median(np.diff(t))),
        "pelvis_forward_per_stride_m": float(disp[1]),
        "root_xy_loop_error_m": float(np.max(np.abs(q[-1, :2] - q[0, :2] - disp[:2]))),
        "joints": {},
    }

    for row, (j, name) in enumerate(JOINTS):
        ax, av = axs[row]
        # Three actual copies of the saved reference, including the HS junctions.
        if source is not None:
            ax.plot(raw_t3/T, raw_q3[:, j]*DEG, color="0.6", lw=1.2,
                    alpha=0.85, label="Measured repeated")
        ax.plot(t3/T, q3[:, j]*DEG, color="tab:blue", lw=1.4,
                label="Loopable repeated")
        for hs in (1, 2):
            ax.axvline(hs, color="0.2", ls="--", lw=0.8)
            ax.axvspan(hs-seam, hs+seam, color="tab:orange", alpha=0.12)
        ax.set(title=f"{name} — angle over 3 strides", xlabel="Stride number",
               ylabel="Angle (deg)", xlim=(0, 3))
        ax.grid(alpha=0.2)
        if row == 0:
            ax.legend(loc="best", fontsize=8)

        # Zoom on the FIRST repeating boundary (stride 1 -> stride 2).
        sl = (t3/T >= 1-zoom_fraction) & (t3/T <= 1+zoom_fraction)
        baseline_velocity = None
        if source is not None:
            # Baseline differentiated from the SAME PCHIP joint-position interpolant
            # used by make_gait_loopable.py. Unlike the supplied raw qvel, this
            # isolates velocity changes attributable to the seam correction itself.
            baseline_velocity = PchipInterpolator(raw_t, raw_q[:, j]).derivative()(t)
            _, baseline_v3 = three_cycles(t, baseline_velocity, T)
            av.plot((t3[sl]-T)*1000, baseline_v3[sl]*DEG,
                    color="0.35", lw=1.3, label="Original position interpolated")
            raw_sl = (raw_t3/T >= 1-zoom_fraction) & (raw_t3/T <= 1+zoom_fraction)
            av.plot((raw_t3[raw_sl]-T)*1000, raw_v3[raw_sl, j-1]*DEG,
                    color="0.7", ls=":", lw=1, alpha=0.8, label="Original supplied qvel")
        av.plot((t3[sl]-T)*1000, v3[sl, j-1]*DEG, color="tab:blue",
                lw=1.7, label="Loopable qvel")
        av.axvline(0, color="0.2", ls="--", lw=0.8)
        av.axvspan(-seam*T*1000, seam*T*1000, color="tab:orange", alpha=0.12)
        av.set(title=f"{name} — angular velocity around right HS",
               xlabel="Time relative to right HS (ms)", ylabel="Angular velocity (deg/s)",
               xlim=(-zoom_fraction*T*1000, zoom_fraction*T*1000))
        av.grid(alpha=0.2)
        if row == 0:
            av.legend(loc="best", fontsize=8)

        acc = np.gradient(v[:, j-1], t, edge_order=2) * DEG
        stats = {
            "angle_jump_at_hs_deg": float(abs(q[-1, j]-q[0, j])*DEG),
            "velocity_jump_at_hs_deg_s": float(abs(v[-1, j-1]-v[0, j-1])*DEG),
            "peak_corrected_velocity_near_hs_deg_s": float(np.max(abs(v[near, j-1]))*DEG),
            "peak_corrected_acceleration_near_hs_deg_s2": float(np.max(abs(acc[near]))),
        }
        if source is not None:
            raw_near = (raw_t <= zoom_fraction*T) | (raw_t >= (1-zoom_fraction)*T)
            measured_peak = float(np.max(abs(raw_v[raw_near, j-1]))*DEG)
            stats["peak_measured_velocity_near_hs_deg_s"] = measured_peak
            stats["peak_velocity_ratio_corrected_to_measured_near_hs"] = (
                stats["peak_corrected_velocity_near_hs_deg_s"] / max(measured_peak, 1e-12))
            raw_acc = np.gradient(raw_v[:, j-1], raw_t, edge_order=2) * DEG
            stats["peak_measured_acceleration_near_hs_deg_s2"] = float(np.max(abs(raw_acc[raw_near])))
            stats["peak_original_interpolated_velocity_near_hs_deg_s"] = float(
                np.max(abs(baseline_velocity[near]))*DEG)
            stats["max_velocity_change_due_to_seam_deg_s"] = float(
                np.max(abs(v[corrected_region, j-1] - baseline_velocity[corrected_region]))*DEG)
            stats["max_velocity_change_outside_seam_deg_s"] = float(
                np.max(abs(v[~corrected_region, j-1] - baseline_velocity[~corrected_region]))*DEG)
        result["joints"][name] = stats

    fig.suptitle(f"{processed.stem} — seam verification (orange: correction windows)", fontsize=14)
    output.mkdir(parents=True, exist_ok=True)
    figpath = output / f"{processed.stem}_verification.png"
    fig.savefig(figpath, dpi=150)
    plt.close(fig)
    result["plot"] = str(figpath)
    (output / f"{processed.stem}_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="One *_loopable.npz or the output folder")
    ap.add_argument("--source-root", type=Path, default=None,
                    help="Optional original references_gait120_standardized folder for comparisons")
    ap.add_argument("--output", type=Path, default=Path("loop_verification"))
    ap.add_argument("--max-files", type=int, default=0, help="Limit number of references (0=all)")
    ap.add_argument("--zoom", type=float, default=0.08,
                    help="Fraction of a stride shown on EACH side of heel strike (default 0.08)")
    args = ap.parse_args()
    if not 0 < args.zoom < 0.5:
        ap.error("--zoom must be between 0 and 0.5")
    paths = ([args.input] if args.input.is_file()
             else sorted(args.input.rglob("*_loopable.npz")))
    if args.max_files > 0:
        paths = paths[:args.max_files]
    if not paths:
        ap.error("No *_loopable.npz files found")
    args.output.mkdir(parents=True, exist_ok=True)
    all_results = []
    for p in paths:
        source = source_file_for(p, args.source_root)
        try:
            res = verify_one(p, args.output, source, args.zoom)
            all_results.append(res)
            gap = max(j["velocity_jump_at_hs_deg_s"] for j in res["joints"].values())
            print(f"OK {p.name} | max HS velocity jump = {gap:.5g} deg/s"
                  + (" | measured comparison available" if source is not None else " | no source comparison"))
        except Exception as exc:
            print(f"ERROR {p}: {exc}")
            all_results.append({"reference": str(p), "error": str(exc)})
    (args.output / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "joint", "angle_jump_deg", "velocity_jump_deg_s",
                         "peak_corrected_velocity_near_hs_deg_s", "peak_measured_velocity_near_hs_deg_s",
                         "peak_velocity_ratio", "peak_corrected_acceleration_near_hs_deg_s2",
                         "max_velocity_change_due_to_seam_deg_s", "max_velocity_change_outside_seam_deg_s"])
        for res in all_results:
            for name, stats in res.get("joints", {}).items():
                writer.writerow([res["reference"], name,
                                 stats["angle_jump_at_hs_deg"], stats["velocity_jump_at_hs_deg_s"],
                                 stats["peak_corrected_velocity_near_hs_deg_s"],
                                 stats.get("peak_measured_velocity_near_hs_deg_s", ""),
                                 stats.get("peak_velocity_ratio_corrected_to_measured_near_hs", ""),
                                 stats["peak_corrected_acceleration_near_hs_deg_s2"],
                                 stats.get("max_velocity_change_due_to_seam_deg_s", ""),
                                 stats.get("max_velocity_change_outside_seam_deg_s", "")])
    print(f"Plots + summary: {args.output} | successful: {sum('error' not in r for r in all_results)}/{len(all_results)}")


if __name__ == "__main__":
    main()
