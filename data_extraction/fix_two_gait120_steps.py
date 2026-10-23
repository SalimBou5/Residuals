"""Correct ONLY the two false left heel-strikes in the exported Gait120 references.

Affected: S029/Trial05/step01 and S115/Trial03/step02.
Use after batch extraction, in the same directory containing the exported files.

The false ~10-12% left-HS force crossing is removed from *_events.csv,
so even the original event replay no longer draws a false GREEN heel-strike.
The left HS annotation/phase is set to an explicitly ESTIMATED 50% stride
position; the original force data remain in the NPZ, and the original
CSV/NPZ/manifest are backed up before changes. No kinematics are modified.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil

import numpy as np

TARGETS = [
    ("S029", "Trial05", "step01", 0.10181995784713907),
    ("S115", "Trial03", "step02", 0.11788464413966933),
]


def correct(root: Path, subject: str, trial: str, step: str, bad_phase: float):
    base = root / subject / trial / step
    files = {
        "npz": base.with_suffix(".npz"),
        "manifest": base.with_name(step + "_manifest.json"),
        "events": base.with_name(step + "_events.csv"),
    }
    for kind, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{subject}/{trial}/{step}: missing {kind}: {path}")

    with np.load(files["npz"], allow_pickle=False) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    needed = {"time_s", "phase_right_01", "phase_left_01", "phase_left_rad", "left_hs_s"}
    if not needed.issubset(arrays):
        raise ValueError(f"{base}: missing NPZ fields: {sorted(needed - arrays.keys())}")
    time = np.asarray(arrays["time_s"], dtype=float)
    if len(time) < 2 or not np.all(np.diff(time) > 0):
        raise ValueError(f"{base}: invalid times")
    t0, t1 = float(time[0]), float(time[-1])
    duration = t1 - t0
    midpoint = t0 + 0.5 * duration

    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    with files["events"].open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {"time_s", "event", "foot_side"}.issubset(fields):
        raise ValueError(f"{files['events']}: unexpected CSV format")

    bad_rows = []
    for index, row in enumerate(rows):
        if (row["event"].strip().lower() in {"hs_candidate", "hs", "on"}
                and row["foot_side"].strip().lower() == "left"):
            phase = (float(row["time_s"]) - t0) / duration
            if abs(phase - bad_phase) < 0.025:
                bad_rows.append(index)
    if len(bad_rows) == 0:
        events = manifest.get("events", {})
        already = (str(events.get("left_hs_source", "")).upper().startswith("ASSUMED")
                   and abs(float(events.get("left_hs_phase_in_right_stride", -1)) - 0.5) < 1e-6)
        if already:
            print(f"ALREADY CORRECTED: {subject}/{trial}/{step}")
            return
        raise ValueError(f"{base}: expected false HS crossing at {bad_phase:.1%} is missing; refused to edit")
    if len(bad_rows) != 1:
        raise ValueError(f"{base}: expected one false HS crossing; found {len(bad_rows)}")

    rejected = rows[bad_rows[0]]
    kept = [row for i, row in enumerate(rows) if i not in bad_rows]
    # Fix the saved annotations, not the sampled kinematics or raw GRF signals.
    arrays["left_hs_s"] = np.array(midpoint, dtype=float)
    arrays["phase_left_01"] = np.mod(np.asarray(arrays["phase_right_01"], dtype=float) - 0.5, 1.0)
    arrays["phase_left_rad"] = 2.0 * np.pi * arrays["phase_left_01"]

    em = manifest.setdefault("events", {})
    old_selection = {
        "left_hs_time_s": em.get("left_hs_time_s"),
        "left_hs_source": em.get("left_hs_source"),
        "left_hs_phase_in_right_stride": em.get("left_hs_phase_in_right_stride"),
    }
    em["left_hs_time_s"] = midpoint
    em["left_hs_source"] = "ASSUMED_half_stride_shift_false_HS_QC"
    em["left_hs_phase_in_right_stride"] = 0.5
    manifest["two_step_event_fix"] = {
        "reason": "Early left HS force re-crossing inconsistent with observed toe-off and walking motion",
        "rejected_display_event": rejected,
        "previous_selected_annotation": old_selection,
        "replacement_left_HS": "Estimated at 50% of right-HS-to-right-HS stride; NOT measured",
        "raw_grf_still_in_npz": True,
        "original_events_csv_saved_as_backup": True,
    }

    # Preserve the originals for reproducibility. Refuse overwriting backups.
    for path in files.values():
        backup = path.with_name(path.name + ".before_two_step_fix.bak")
        if backup.exists():
            raise FileExistsError(f"Backup already exists, refusing to overwrite: {backup}")
    for path in files.values():
        shutil.copy2(path, path.with_name(path.name + ".before_two_step_fix.bak"))

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(kept)

    temp_npz = base.with_name(step + ".two_step_fix_tmp.npz")
    temp_manifest = base.with_name(step + "_manifest.json.two_step_fix_tmp")
    temp_csv = base.with_name(step + "_events.csv.two_step_fix_tmp")
    try:
        np.savez_compressed(temp_npz, **arrays)
        temp_manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp_csv.write_text(csv_buffer.getvalue(), encoding="utf-8")
        temp_npz.replace(files["npz"])
        temp_manifest.replace(files["manifest"])
        temp_csv.replace(files["events"])
    finally:
        for path in (temp_npz, temp_manifest, temp_csv):
            if path.exists():
                path.unlink()

    print(f"FIXED: {subject}/{trial}/{step}: deleted false green HS at {bad_phase:.1%}; "
          "estimated left HS at 50%; kept all toe-offs and original kinematics")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing S029 and S115 exports")
    args = parser.parse_args()
    for subject, trial, step, bad_phase in TARGETS:
        correct(args.root, subject, trial, step, bad_phase)


if __name__ == "__main__":
    main()
