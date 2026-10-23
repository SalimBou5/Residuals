"""Rotate Gait120 MyoLeg references to one forward world heading (+Y by default).

Each stride receives a single rigid yaw rotation based on its root's start-to-end
horizontal displacement. The root's horizontal starting position is then reset
 to (0, 0). No joints, gait phases, durations, or source files are changed.

Creates NEW files in --output-root. Recomputes qvel using the supplied MuJoCo
model, because free-root velocities must be consistent with the transformed qpos.
Original GRFs/markers are preserved in their original OpenSim laboratory frame;
this is documented in the output manifest. Their raw event CSV is also copied.

Recommended: run gait120_event_qc.py --apply on the source BEFORE this script.

Example (Windows PowerShell):
  python standardize_gait120_heading.py --model "$model" \
      --input-root gait120_references_full \
      --output-root gait120_references_forward

Test first: append --start-subject 60 --end-subject 60
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)


def apply_heading(qpos: np.ndarray, root_adr: int, target_axis: str = '+y'):
    """Return new qpos and transformation metadata, using one world-frame yaw."""
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 3 or q.shape[1] < root_adr + 7:
        raise ValueError('Unexpected qpos array dimensions')
    if not np.isfinite(q).all():
        raise ValueError('Nonfinite original qpos')
    start = q[0, root_adr:root_adr + 2].copy()
    displacement = q[-1, root_adr:root_adr + 2] - start
    norm = float(np.linalg.norm(displacement))
    if norm < 0.2:
        raise ValueError(f'Insufficient horizontal displacement to identify heading: {norm:.3f} m')

    angle_before = float(np.arctan2(displacement[1], displacement[0]))
    target_angle = np.pi / 2 if target_axis == '+y' else 0.0
    yaw = float(np.arctan2(np.sin(target_angle - angle_before),
                           np.cos(target_angle - angle_before)))
    co, si = np.cos(yaw), np.sin(yaw)
    Rxy = np.array([[co, -si], [si, co]])

    transformed = q.copy()
    transformed[:, root_adr:root_adr+2] = (q[:, root_adr:root_adr+2] - start) @ Rxy.T

    # Left multiplication: rotate the world-fixed root orientation by Rz(yaw).
    # This is *not* a local anatomical rotation and leaves all leg joints intact.
    quat_xyzw = np.roll(q[:, root_adr+3:root_adr+7], -1, axis=-1)
    original_rot = Rotation.from_quat(quat_xyzw)
    transformed_rot = Rotation.from_euler('z', yaw) * original_rot
    new_quat = as_quat_wxyz(transformed_rot)
    # Quaternions q and -q are physically equivalent. Choose continuous signs.
    for i in range(1, len(new_quat)):
        if np.dot(new_quat[i], new_quat[i-1]) < 0:
            new_quat[i] *= -1
    transformed[:, root_adr+3:root_adr+7] = new_quat

    new_displacement = transformed[-1, root_adr:root_adr+2] - transformed[0, root_adr:root_adr+2]
    expected = np.array([0.0, norm]) if target_axis == '+y' else np.array([norm, 0.0])
    if not np.allclose(new_displacement, expected, atol=1e-8):
        raise AssertionError(f'Incorrect heading alignment: {new_displacement} vs {expected}')
    if not np.array_equal(transformed[:, root_adr+7:], q[:, root_adr+7:]):
        raise AssertionError('Non-root coordinates changed unexpectedly')
    if np.max(np.abs(np.linalg.norm(new_quat, axis=1)-1)) > 1e-6:
        raise AssertionError('Non-unit quaternion after alignment')
    if np.any(np.sum(new_quat[:-1] * new_quat[1:], axis=1) < 0):
        raise AssertionError('Quaternion sign discontinuity after alignment')

    return transformed, {
        'original_heading_deg_from_plus_x': float(np.degrees(angle_before)),
        'applied_world_z_yaw_deg': float(np.degrees(yaw)),
        'initial_root_xy_m_in_original_mujoco_frame': start.tolist(),
        'horizontal_displacement_m': norm,
        'target_direction': target_axis,
        'operation': 'Single rigid world-frame yaw per stride; root XY reset to (0,0)',
        'joint_coordinates_changed': False,
        'original_grf_and_marker_frame': 'Original OpenSim laboratory coordinates (NOT rotated)',
        'original_events_and_phases': 'Unchanged',
        'qvel': 'Recomputed from aligned qpos using MuJoCo mj_differentiatePos',
    }


def differentiate_pos(model, t: np.ndarray, qpos: np.ndarray):
    import mujoco
    nv = model.nv
    v = np.zeros((len(t), nv), dtype=np.float64)
    for i in range(len(t)):
        a = max(0, i-1)
        b = min(len(t)-1, i+1)
        if a != b:
            mujoco.mj_differentiatePos(model, v[i], float(t[b]-t[a]), qpos[a], qpos[b])
    return v


def process_one(model, src: Path, in_root: Path, out_root: Path, overwrite: bool):
    relative = src.relative_to(in_root)
    dest = out_root / relative
    if dest.exists() and not overwrite:
        return {'reference': str(relative), 'status': 'SKIPPED_EXISTS'}

    manifest_path = src.with_name(src.stem + '_manifest.json')
    event_csv_path = src.with_name(src.stem + '_events.csv')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest not found for {src}')
    original_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if 'heading_alignment' in original_manifest:
        raise ValueError('Input was already heading-aligned: refusing a second transformation')
    with np.load(src, allow_pickle=False) as data:
        entries = {key: data[key].copy() for key in data.files}
    if 'qpos' not in entries or 'qvel' not in entries or 'time_s' not in entries:
        raise ValueError('Missing qpos/qvel/time_s')
    qpos_original = entries['qpos']
    t = np.asarray(entries['time_s'], dtype=np.float64)
    if len(t) != len(qpos_original) or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid reference timestamps')
    if qpos_original.shape[1] != model.nq or entries['qvel'].shape != (len(t), model.nv):
        raise ValueError('Reference is incompatible with supplied MuJoCo model')
    root_adr = int(model.joint('root').qposadr[0])
    qpos_new, transform = apply_heading(qpos_original, root_adr, '+y')
    qvel_new = differentiate_pos(model, t, qpos_new)
    if not np.isfinite(qvel_new).all():
        raise ValueError('Nonfinite new qvel')
    entries['qpos'] = qpos_new
    entries['qvel'] = qvel_new
    # Preserve measured GRF and raw TRC marker coordinates as stored in source.
    original_manifest['heading_alignment'] = transform
    original_manifest['heading_alignment']['source_reference_sha256'] = hashlib.sha256(src.read_bytes()).hexdigest()
    original_manifest['heading_alignment']['source_reference_path'] = str(src)
    original_manifest['heading_alignment']['note'] = (
        'For comparisons with retained OpenSim-lab marker/GRF coordinates, '
        'apply the documented OpenSim-to-MuJoCo basis change and this yaw/origin transform.'
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_npz = dest.with_name(dest.stem + '.heading_tmp.npz')
    temp_manifest = dest.with_name(dest.stem + '_manifest.json.heading_tmp')
    try:
        np.savez_compressed(temp_npz, **entries)
        temp_manifest.write_text(json.dumps(original_manifest, indent=2) + '\n', encoding='utf-8')
        temp_npz.replace(dest)
        temp_manifest.replace(dest.with_name(dest.stem + '_manifest.json'))
        if event_csv_path.exists():
            shutil.copy2(event_csv_path, dest.with_name(dest.stem + '_events.csv'))
    finally:
        for p in (temp_npz, temp_manifest):
            if p.exists():
                p.unlink()
    return {'reference': str(relative), 'status': 'ALIGNED',
            'old_heading_deg': round(transform['original_heading_deg_from_plus_x'], 3),
            'yaw_applied_deg': round(transform['applied_world_z_yaw_deg'], 3),
            'displacement_m': round(transform['horizontal_displacement_m'], 5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--model', type=Path, required=True, help='Path to complete myolegs.xml')
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--start-subject', type=int, default=11)
    ap.add_argument('--end-subject', type=int, default=120)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    if args.input_root.resolve() == args.output_root.resolve():
        ap.error('--input-root and --output-root must be different (source is never overwritten)')
    if args.input_root.resolve() in args.output_root.resolve().parents:
        ap.error('--output-root must not be inside --input-root')
    if not args.input_root.is_dir():
        ap.error(f'Input directory missing: {args.input_root}')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.model))
    sources = sorted(args.input_root.glob('S[0-9][0-9][0-9]/Trial*/step*.npz'))
    sources = [p for p in sources if args.start_subject <= int(p.parent.parent.name[1:]) <= args.end_subject]
    if args.limit:
        sources = sources[:args.limit]
    print(f'Discovered {len(sources)} stride references')
    rows = []
    for i, src in enumerate(sources, 1):
        try:
            result = process_one(model, src, args.input_root, args.output_root, args.overwrite)
            print(f'[{i}/{len(sources)}] {result["reference"]}: {result["status"]}' +
                  (f', yaw={result["yaw_applied_deg"]:.2f} deg' if result['status']=='ALIGNED' else ''))
        except Exception as e:
            result = {'reference': str(src.relative_to(args.input_root)), 'status': 'FAILED',
                      'error': f'{type(e).__name__}: {e}'}
            print(f'[{i}/{len(sources)}] {result["reference"]}: {result["error"]}')
        rows.append(result)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / 'heading_alignment_summary.csv').open('w', newline='', encoding='utf-8') as f:
        fieldnames = list(dict.fromkeys(key for r in rows for key in r))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    print(f'Output: {args.output_root} | aligned {sum(r["status"]=="ALIGNED" for r in rows)}, '
          f'failed {sum(r["status"]=="FAILED" for r in rows)}')


if __name__ == '__main__':
    main()
