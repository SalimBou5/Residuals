"""Rotate Gait120 MyoLeg references to one forward world heading (+Y by default).

Each stride receives a single rigid yaw rotation based on its root's start-to-end
horizontal displacement. The root's horizontal starting position is reset to (0, 0).
Additionally, a single constant root-height correction is estimated from foot-to-floor
collision-geometry distances measured at mid-stance, using the saved HS/TO events.
This floor alignment can be disabled with --no-floor-align. No joints, gait phases,
durations, or source files are changed.

Creates NEW files in --output-root. Recomputes qvel using the supplied MuJoCo
model, because free-root velocities must be consistent with the transformed qpos.
Original GRFs/markers are preserved in their original OpenSim laboratory frame;
this is documented in the output manifest. Their raw event CSV is also copied.

Recommended: run gait120_event_qc.py --apply on the source BEFORE this script.

Example (Windows PowerShell):
  python standardize_gait120_heading.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --input-root ../references_gait120 --output-root ../references_gait120_standardized

Test first: append --start-subject 60 --end-subject 60 --limit 2
Review printed collision geom names; specify --floor-geom and --right-foot-geoms/
--left-foot-geoms if auto-detection does not match your model.
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



def geom_label(model, geom_id: int) -> str:
    import mujoco
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name or f'<unnamed geom {geom_id}>'


def body_label(model, body_id: int) -> str:
    import mujoco
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f'<body {body_id}>'


def detect_contact_geoms(model, floor_name=None, right_names=None, left_names=None):
    """Find horizontal floor and foot collision geoms; allow explicit CLI overrides.

    Automatic foot detection uses names of collision geoms and their parent bodies.
    Review the printed selection for your specific MyoLeg XML.
    """
    import mujoco
    import re

    def ids_for_names(names):
        ids = []
        for name in names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise ValueError(f'Unknown geom {name!r}; use --list-contact-geoms')
            ids.append(int(gid))
        return ids

    if floor_name:
        floors = ids_for_names([floor_name])
    else:
        planes = [i for i in range(model.ngeom)
                  if model.geom_bodyid[i] == 0
                  and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE]
        named_planes = [i for i in planes if any(
            w in geom_label(model, i).lower() for w in ('floor', 'ground'))]
        floors = named_planes if named_planes else planes
        if len(floors) != 1:
            raise ValueError(f'Expected one ground plane; found {len(floors)}. '
                             'Specify --floor-geom NAME (use --list-contact-geoms).')

    # A constant vertical root translation is valid only for horizontal ground.
    # World geom +Z is the plane normal, and a box top has the same requirement.
    for gid in floors:
        if model.geom_bodyid[gid] != 0:
            raise ValueError('Ground geom must be world-fixed')
        if model.geom_type[gid] not in (mujoco.mjtGeom.mjGEOM_PLANE,
                                        mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError('Ground must be a horizontal plane or box')
        # Check the static model-space orientation (geom quaternion wxyz).
        from scipy.spatial.transform import Rotation
        q = model.geom_quat[gid]
        normal = Rotation.from_quat(np.roll(q, -1)).apply([0, 0, 1])
        if np.dot(normal, [0, 0, 1]) < 0.99:
            raise ValueError(f'Ground geom {geom_label(model, gid)} is not horizontal')

    def relevant_names(geom_id):
        # Geom itself and its ancestor bodies (some foot geoms have generic names).
        names = [geom_label(model, geom_id)]
        b = int(model.geom_bodyid[geom_id])
        while b > 0:
            names.append(body_label(model, b))
            b = int(model.body_parentid[b])
        return names

    def is_foot(names, side):
        foot_re = re.compile(r'foot|heel|calcn|calcane|talus|toes?|metatars|\bmtp\b|tarsal|hallux', re.I)
        for name in names:
            lo = name.lower()
            has_side = bool(re.search(r'right' if side == 'r' else r'left', lo))
            has_side |= bool(re.search(rf'(^|[_\-.]){side}($|[_\-.])', lo))
            has_side |= bool(re.search(rf'^{side}(?:foot|heel|cal|toe|mtp|talus)', lo))
            if has_side and foot_re.search(lo):
                return True
        return False

    def select_foot(side, override):
        if override is not None:
            return ids_for_names(override)
        result = []
        for i in range(model.ngeom):
            # We want physical contact geometries, not purely visual meshes.
            if model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
                continue
            if is_foot(relevant_names(i), side):
                result.append(i)
        if not result:
            raise ValueError(f'Could not detect {"right" if side == "r" else "left"} '
                             'foot collision geoms; specify --right-foot-geoms or '
                             '--left-foot-geoms using --list-contact-geoms')
        return result

    return floors, select_foot('r', right_names), select_foot('l', left_names)


def sample_stance_phases(manifest: dict, points_per_foot: int = 7):
    """Mid-stance samples based on existing HS/TO annotations, including estimated TO."""
    ev = manifest.get('events', {})
    required = ('right_toe_off_phase', 'left_hs_phase_in_right_stride', 'left_toe_off_phase')
    missing = [k for k in required if ev.get(k) is None]
    if missing:
        raise ValueError(f'Missing event phases in manifest: {missing}')
    rto, lhs, lto = (float(ev[k]) for k in required)
    if not (0 < rto < 1 and 0 < lhs < 1 and 0 <= lto < 1):
        raise ValueError(f'Invalid HS/TO phases: RTO={rto}, LHS={lhs}, LTO={lto}')
    lto_next = lto + (1.0 if lto <= lhs else 0.0)
    if not 0.2 < lto_next - lhs < 0.9:
        raise ValueError('Left stance duration implausible; check event phases')
    f = np.linspace(0.25, 0.75, points_per_foot)
    return {'right': rto * f,
            'left': np.mod(lhs + (lto_next - lhs) * f, 1.0)}


def interpolate_pose(t, qpos, phase, root_adr):
    """Interpolate the measured samples; normalize the free-root quaternion."""
    ts = float(t[0] + float(phase) * (t[-1] - t[0]))
    hi = int(np.clip(np.searchsorted(t, ts, side='right'), 1, len(t)-1))
    lo = hi-1
    alpha = (ts - t[lo]) / (t[hi] - t[lo])
    q0, q1 = qpos[lo], qpos[hi]
    q = (1-alpha)*q0 + alpha*q1
    # Keep quaternion interpolation on the shortest arc (SLERP is unnecessary
    # for the small frame-to-frame rotations used to estimate floor height).
    qa = q0[root_adr+3:root_adr+7]
    qb = q1[root_adr+3:root_adr+7]
    if np.dot(qa, qb) < 0:
        qb = -qb
    quat = (1-alpha)*qa + alpha*qb
    q[root_adr+3:root_adr+7] = quat / np.linalg.norm(quat)
    return q


def estimate_floor_alignment(model, qpos, t, root_adr, manifest,
                             floor_ids, right_ids, left_ids,
                             max_dist=0.30, warning_m=0.025):
    """Estimate median stance clearance and validate a *constant* root-z shift.

    mj_geomDistance reports the signed surface separation.  This is queried
    explicitly because normal data.contact only includes near-contact pairs.
    """
    import mujoco
    data = mujoco.MjData(model)
    phases = sample_stance_phases(manifest)
    fromto = np.zeros(6)

    def measure(q, foot_ids):
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        d = min(float(mujoco.mj_geomDistance(model, data, gnd, fg, max_dist, fromto))
                for gnd in floor_ids for fg in foot_ids)
        if not np.isfinite(d) or d >= max_dist - 1e-8:
            raise ValueError('No reliable foot-ground signed distance within '
                             f'{max_dist:.2f} m. Check selected collision geoms.')
        return d

    before = {}
    for side, foot_ids in (('right', right_ids), ('left', left_ids)):
        before[side] = np.asarray([
            measure(interpolate_pose(t, qpos, p, root_adr), foot_ids)
            for p in phases[side]
        ], dtype=float)
    all_before = np.concatenate([before['right'], before['left']])
    dz = -float(np.median(all_before))

    # Verify with the actual collision-distance routine after shifting.
    after = {}
    for side, foot_ids in (('right', right_ids), ('left', left_ids)):
        vals = []
        for p in phases[side]:
            q = interpolate_pose(t, qpos, p, root_adr)
            q[root_adr+2] += dz
            vals.append(measure(q, foot_ids))
        after[side] = np.asarray(vals)
    all_after = np.concatenate([after['right'], after['left']])
    max_error = float(np.max(np.abs(all_after)))
    warnings = []
    if max_error > warning_m:
        warnings.append(f'Stance geometry still off by up to {max_error*1000:.1f} mm '
                        'after one root-height shift; inspect this stride')
    if abs(dz) > 0.08:
        warnings.append(f'Large height correction: {dz*1000:+.1f} mm; check model and reference')
    return dz, {
        'method': 'median signed ground-to-stance-foot collision-geom distance',
        'delta_z_m': dz,
        'sample_count_per_foot': len(phases['right']),
        'floor_geoms': [geom_label(model, g) for g in floor_ids],
        'right_foot_geoms': [geom_label(model, g) for g in right_ids],
        'left_foot_geoms': [geom_label(model, g) for g in left_ids],
        'median_distance_before_m': float(np.median(all_before)),
        'right_median_before_m': float(np.median(before['right'])),
        'left_median_before_m': float(np.median(before['left'])),
        'right_median_after_m': float(np.median(after['right'])),
        'left_median_after_m': float(np.median(after['left'])),
        'max_abs_distance_after_m': max_error,
        'warnings': warnings,
        'foot_clearance_sign': 'negative=penetration; positive=separation',
        'operation': 'One constant world-z translation for entire stride; no joint or velocity change',
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


def process_one(model, src: Path, in_root: Path, out_root: Path, overwrite: bool,
                align_floor: bool, floor_ids, right_ids, left_ids):
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
    floor_info = None
    if align_floor:
        dz, floor_info = estimate_floor_alignment(
            model, qpos_new, t, root_adr, original_manifest,
            floor_ids, right_ids, left_ids)
        qpos_new[:, root_adr+2] += dz  # One constant vertical offset, all frames.
    qvel_new = differentiate_pos(model, t, qpos_new)
    if not np.isfinite(qvel_new).all():
        raise ValueError('Nonfinite new qvel')
    entries['qpos'] = qpos_new
    entries['qvel'] = qvel_new
    # Preserve measured GRF and raw TRC marker coordinates as stored in source.
    original_manifest['heading_alignment'] = transform
    if floor_info is not None:
        original_manifest['floor_alignment'] = floor_info
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
    result = {'reference': str(relative), 'status': 'ALIGNED',
              'old_heading_deg': round(transform['original_heading_deg_from_plus_x'], 3),
              'yaw_applied_deg': round(transform['applied_world_z_yaw_deg'], 3),
              'displacement_m': round(transform['horizontal_displacement_m'], 5)}
    if floor_info is not None:
        result['delta_z_m'] = round(floor_info['delta_z_m'], 6)
        result['max_stance_clearance_error_m'] = round(floor_info['max_abs_distance_after_m'], 6)
        result['floor_warning'] = ' | '.join(floor_info['warnings'])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--model', type=Path, required=True, help='Path to complete myolegs.xml')
    ap.add_argument('--input-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--start-subject', type=int, default=11)
    ap.add_argument('--end-subject', type=int, default=120)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--no-floor-align', action='store_true',
                    help='Disable automatic stance-foot root-height correction')
    ap.add_argument('--floor-geom', type=str, default=None,
                    help='Name of the fixed horizontal floor collision geom (auto-detects single ground plane)')
    ap.add_argument('--right-foot-geoms', nargs='+', default=None, metavar='NAME',
                    help='Exact names of right foot collision geoms (default: auto-detect)')
    ap.add_argument('--left-foot-geoms', nargs='+', default=None, metavar='NAME',
                    help='Exact names of left foot collision geoms (default: auto-detect)')
    ap.add_argument('--list-contact-geoms', action='store_true',
                    help='Print geom IDs/names/bodies/contype for configuring floor alignment')
    args = ap.parse_args()
    if args.input_root.resolve() == args.output_root.resolve():
        ap.error('--input-root and --output-root must be different (source is never overwritten)')
    if args.input_root.resolve() in args.output_root.resolve().parents:
        ap.error('--output-root must not be inside --input-root')
    if not args.input_root.is_dir():
        ap.error(f'Input directory missing: {args.input_root}')
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if args.list_contact_geoms:
        for gid in range(model.ngeom):
            print(f'{gid:4d} {geom_label(model, gid):35s} '
                  f'body={body_label(model, int(model.geom_bodyid[gid])):30s} '
                  f'contype={model.geom_contype[gid]} '
                  f'conaffinity={model.geom_conaffinity[gid]}')
        return
    if args.no_floor_align:
        floor_ids = right_ids = left_ids = None
    else:
        floor_ids, right_ids, left_ids = detect_contact_geoms(
            model, args.floor_geom, args.right_foot_geoms, args.left_foot_geoms)
        print('Floor geoms:', [geom_label(model, i) for i in floor_ids])
        print('Right contact geoms:', [geom_label(model, i) for i in right_ids])
        print('Left contact geoms:', [geom_label(model, i) for i in left_ids])
    sources = sorted(args.input_root.glob('S[0-9][0-9][0-9]/Trial*/step*.npz'))
    sources = [p for p in sources if args.start_subject <= int(p.parent.parent.name[1:]) <= args.end_subject]
    if args.limit:
        sources = sources[:args.limit]
    print(f'Discovered {len(sources)} stride references')
    rows = []
    for i, src in enumerate(sources, 1):
        try:
            result = process_one(model, src, args.input_root, args.output_root,
                                 args.overwrite, not args.no_floor_align,
                                 floor_ids, right_ids, left_ids)
            print(f'[{i}/{len(sources)}] {result["reference"]}: {result["status"]}' +
                  (f', yaw={result["yaw_applied_deg"]:.2f} deg'
                   + (f', dz={result["delta_z_m"]*1000:+.1f} mm' if 'delta_z_m' in result else '')
                   + (' [CHECK FLOOR]' if result.get('floor_warning') else '')
                   if result['status']=='ALIGNED' else ''))
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
