"""Batch export Gait120 LevelWalking steps to MyoLeg reference arrays.

Uses the EXISTING, tested mapping functions from:
  gait120_kinematic_compatibility.py
  check_gait120_reference_feasibility.py

No original data are altered. This is a candidate reference exporter, NOT a
validation of dynamics or the source IK. All strides stay at their original
sampling rate and original duration. A gait-phase coordinate is added.

Examples:
  python batch_extract_gait120.py --model path/to/myolegs.xml \
    --data-root path/to/Gait120_011_to_020 --output-root references --limit 10

  python batch_extract_gait120.py --model path/to/myolegs.xml \
    --data-root path/to/Downloads --output-root references

Requires numpy, pandas, scipy and mujoco, plus the two helper scripts above.
This version fixes quaternion SIGN discontinuities before velocity computation.
Use --summary-name batch_summary_S060_retry.csv to preserve the previous report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gait120_kinematic_compatibility import LEG_JOINTS, read_mot, read_trc

THRESHOLD = 20.0  # N, Gait120 paper


def unwrap_root_quaternion_signs(model, qpos):
    """Choose a continuous quaternion representation without altering any pose.

    q and -q describe exactly the same rotation. Reject actual large rotation
    steps instead of masking them as a sign-convention issue. Returns a COPY.
    """
    out = np.asarray(qpos, dtype=float).copy()
    adr = int(model.joint('root').qposadr[0]) + 3
    quats = out[:, adr:adr + 4]
    if not np.isfinite(quats).all():
        raise ValueError('Nonfinite root quaternion')
    err = np.max(np.abs(np.linalg.norm(quats, axis=1) - 1.0))
    if err > 1e-6:
        raise ValueError(f'Root quaternions are not unit length (max error={err:.3g})')

    flips = 0
    for i in range(1, len(quats)):
        if np.dot(quats[i-1], quats[i]) < 0:
            quats[i] *= -1
            flips += 1

    dots = np.sum(quats[:-1] * quats[1:], axis=1)
    if np.any(dots < 0):
        raise ValueError('Quaternion sign unwrapping failed')
    # A sign flip is harmless, but a large physical rotation jump is not.
    steps_deg = np.rad2deg(2 * np.arccos(np.clip(dots, 0.0, 1.0)))
    max_step = float(steps_deg.max()) if len(steps_deg) else 0.0
    if max_step > 15.0:
        raise ValueError(f'Large physical root orientation jump: {max_step:.3f} deg/frame')
    return out, flips, max_step


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def on_off_crossings(time, force, threshold=THRESHOLD, refractory_s=0.020):
    """20 N crossings, ignoring any event within 20 ms of an accepted event.

    Called independently for each force plate. No fabricated end events.
    """
    active = np.asarray(force) >= threshold
    crossings = []
    last_accepted = float('-inf')
    for idx in np.flatnonzero(active[1:] != active[:-1]) + 1:
        y0, y1 = float(force[idx-1]), float(force[idx])
        frac = (threshold-y0)/(y1-y0) if y1 != y0 else 1.0
        tc = float(time[idx-1] + np.clip(frac, 0, 1)*(time[idx]-time[idx-1]))
        if tc - last_accepted < refractory_s:
            continue
        crossings.append((tc, 'HS_candidate' if active[idx] else 'TO_candidate'))
        last_accepted = tc
    return crossings, bool(active[0]), bool(active[-1])


def assign_foot_by_cop(event_t, plate, grf_time, grf_columns, grf_values,
                       marker_time, marker_names, marker_xyz, event_kind='on'):
    """Assign force event to foot by CoP proximity to heel/foot, in OpenSim X-Z.

    Labels are conservative; the plate number is never interpreted as a side.
    Heel-to-CoP distance is used at onset, with a foot-segment consistency check.
    All distances are in meters. Uncertain geometry returns 'unassigned'.
    """
    px, pz = (f'ground_force{plate}_p{axis}' for axis in ('x','z'))
    if any(x not in grf_columns for x in (px, pz)):
        return 'unassigned', 'missing_cop', np.nan, np.nan
    if not all(x in marker_names for x in ('LHEE','RHEE','LTOE','RTOE')):
        return 'unassigned', 'missing_foot_markers', np.nan, np.nan
    gi = int(np.argmin(abs(grf_time-event_t)))
    mi = int(np.argmin(abs(marker_time-event_t)))
    cop = grf_values[gi, [grf_columns.index(px), grf_columns.index(pz)]]
    scores = {}
    for side in ('left', 'right'):
        prefix = 'L' if side == 'left' else 'R'
        heel = marker_xyz[mi, marker_names.index(f'{prefix}HEE'), :]
        toe = marker_xyz[mi, marker_names.index(f'{prefix}TOE'), :]
        a, b = heel[[0,2]], toe[[0,2]]
        ab = b-a
        f = np.clip(np.dot(cop-a, ab)/max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        d_segment = float(np.linalg.norm(cop-(a+f*ab)))
        d_heel = float(np.linalg.norm(cop-a))
        scores[side] = (d_heel, d_segment, float(heel[1]))
    # CoP can migrate towards toes during stance. For HS prioritize heel; for
    # TO / truncated contact prioritize distance to whole foot segment.
    key = 0 if event_kind == 'on' else 1
    order = sorted(scores, key=lambda s: scores[s][key])
    a, b = order
    nearest, other = scores[a][key], scores[b][key]
    margin = other-nearest
    # Deliberately conservative; make uncertain events explicit.
    close = nearest <= (0.18 if event_kind == 'on' else 0.14)
    distinct = margin >= 0.07
    if close and distinct:
        return a, 'cop_marker_geometry', nearest, margin
    return 'unassigned', 'ambiguous_cop_marker_geometry', nearest, margin


def detect_force_events(grf_time, grf_cols, grf_values, marker_time,
                        marker_names, marker_xyz, t0, t1):
    """Provide all observed threshold crossings and conservative foot labels."""
    events, contacts = [], {}
    for plate in (1,2):
        col = f'ground_force{plate}_vy'
        if col not in grf_cols:
            raise ValueError(f'Missing {col}')
        f = grf_values[:, grf_cols.index(col)]
        crossings, first_active, last_active = on_off_crossings(grf_time, f)
        contacts[str(plate)] = {
            'starts_loaded': first_active, 'ends_loaded': last_active,
            'number_threshold_crossings': len(crossings),
        }
        current_side = None
        if first_active:
            # First sample is inside an already-started contact; no HS event.
            current_side, _, _, _ = assign_foot_by_cop(
                grf_time[0], plate, grf_time, grf_cols, grf_values,
                marker_time, marker_names, marker_xyz, event_kind='off')
        for when, kind in crossings:
            if not (t0-0.002 <= when <= t1+0.002):
                continue
            if kind == 'HS_candidate':
                side, method, d, margin = assign_foot_by_cop(
                    when, plate, grf_time, grf_cols, grf_values,
                    marker_time, marker_names, marker_xyz, event_kind='on')
                current_side = side
            else:
                side = current_side or 'unassigned'
                if side == 'unassigned':
                    side, method, d, margin = assign_foot_by_cop(
                        when, plate, grf_time, grf_cols, grf_values,
                        marker_time, marker_names, marker_xyz, event_kind='off')
                else:
                    method, d, margin = 'same_plate_contact_bout', np.nan, np.nan
                current_side = None
            events.append({
                'time_s': when, 'time_since_start_s': when-t0,
                'normalized_phase': (when-t0)/(t1-t0),
                'plate': plate, 'force_channel': col, 'threshold_N': THRESHOLD,
                'event': kind, 'foot_side': side, 'side_method': method,
                'nearest_distance_m': float(d), 'side_margin_m': float(margin),
                'event_source': 'observed_20N_crossing',
            })
    return sorted(events, key=lambda x: x['time_s']), contacts


def build_phase_and_events(t, force_events):
    """Right phase based on paper's right-HS-to-right-HS segmented boundaries."""
    t0, t1 = float(t[0]), float(t[-1])
    T = t1-t0
    phase_right = (t-t0)/T
    observed_left_hs = [e for e in force_events
                        if e['event'] == 'HS_candidate' and e['foot_side'] == 'left'
                        and 0.10 < e['normalized_phase'] < 0.90]
    if observed_left_hs:
        # If multiple, select the plausible opposite-leg event near midstride;
        # retain EVERY observation in the event CSV regardless.
        lhs = min(observed_left_hs, key=lambda e: abs(e['normalized_phase']-0.5))
        left_hs_time = lhs['time_s']
        left_method = 'observed_20N_crossing_cop_labeled'
    else:
        left_hs_time = t0 + T/2
        left_method = 'ASSUMED_half_stride_shift_not_measured'
    left_offset = (left_hs_time-t0)/T
    phase_left = np.mod(phase_right-left_offset, 1.0)
    # Dataset boundaries: paper documents right HS -> right HS; the actual
    # crossing may lie just outside the truncated MOT. Mark accordingly.
    observed_rhs = [e['time_s'] for e in force_events
                    if e['event'] == 'HS_candidate' and e['foot_side'] == 'right']
    boundary_sources = [
        'force_corrob_within_30ms' if any(abs(x-tb)<=0.03 for x in observed_rhs)
        else 'dataset_segmentation_right_HS_inferred'
        for tb in (t0, t1)]
    # NB: Toe-off is ONLY a directly observed downward force crossing; if
    # absent, it remains unknown. Do not guess a stance ratio.
    return phase_right, phase_left, {
        'stride_start_right_hs_s': t0, 'stride_end_right_hs_s': t1,
        'right_hs_boundary_sources': boundary_sources,
        'left_hs_time_s': float(left_hs_time), 'left_hs_source': left_method,
        'left_hs_phase_in_right_stride': float(left_offset),
        'observed_right_toe_off_s': [e['time_s'] for e in force_events if e['event']=='TO_candidate' and e['foot_side']=='right'],
        'observed_left_toe_off_s': [e['time_s'] for e in force_events if e['event']=='TO_candidate' and e['foot_side']=='left'],
    }


def discover_steps(root: Path, start_subject: int, end_subject: int):
    """Support a Gait120 ZIP folder, one subject folder, or parent of ZIP folders."""
    candidates = []
    if root.name.upper().startswith('S') and root.name[1:].isdigit():
        candidates = [root]
    else:
        candidates.extend(root.glob('S[0-9][0-9][0-9]'))
        candidates.extend(root.glob('Gait120_*/S[0-9][0-9][0-9]'))
    seen = set()
    for subj in sorted(candidates):
        if not subj.is_dir():
            continue
        try:
            sid = int(subj.name[1:])
        except ValueError:
            continue
        if not (start_subject <= sid <= end_subject):
            continue
        joint_base = subj/'JointAngle'/'LevelWalking'
        for jd in sorted(joint_base.glob('Trial*')):
            for mot in sorted(jd.iterdir() if jd.is_dir() else []):
                if mot.suffix.lower() != '.mot' or not mot.stem.lower().startswith('step'):
                    continue
                key = (sid, jd.name.lower(), mot.stem.lower())
                if key in seen:
                    continue
                seen.add(key)
                yield subj, jd.name, mot


def find_partner(subject: Path, segment: str, trial: str, stem: str, suffix: str):
    directory = subject/'MotionCapture'/'LevelWalking'/segment/trial
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    matches = [p for p in directory.iterdir()
               if p.stem.lower() == stem.lower() and p.suffix.lower() == suffix.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {stem}{suffix} under {directory}, got {len(matches)}')
    return matches[0]


def export_step(model, subject, trial, joint_path, dest, overwrite):
    import mujoco
    from check_gait120_reference_feasibility import candidate_qpos, differentiate_qpos
    out = dest/subject.name/trial/joint_path.stem.lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.npz').exists() and not overwrite:
        return {'subject':subject.name, 'trial':trial, 'step':joint_path.stem,
                'status':'SKIPPED_EXISTS'}
    trc_path = find_partner(subject, 'TRC', trial, joint_path.stem, '.trc')
    grf_path = find_partner(subject, 'MOT', trial, joint_path.stem, '.mot')
    joints, grf = read_mot(joint_path), read_mot(grf_path)
    mt, mn, mx = read_trc(trc_path)
    t = joints['time'].to_numpy(float)
    tg = grf['time'].to_numpy(float)
    if len(t) < 3 or len(tg) < 2 or np.any(np.diff(t) <= 0) or np.any(np.diff(tg) <= 0):
        raise ValueError('Invalid timestamps')
    if len(t) != len(mt) or not np.allclose(t, mt, atol=1e-5, rtol=0):
        raise ValueError('TRC/joint time mismatch')
    if t[0] < tg[0]-0.001 or t[-1] > tg[-1]+0.001:
        raise ValueError('GRF interval does not cover kinematic interval')
    if not set(LEG_JOINTS).issubset(joints.columns):
        raise ValueError('Missing leg coordinate(s)')
    for label, array in [('joint MOT', joints.select_dtypes(include='number').to_numpy()),
                         ('force MOT', grf.select_dtypes(include='number').to_numpy()),
                         ('marker TRC', mx)]:
        if not np.isfinite(array).all():
            raise ValueError(f'Nonfinite values: {label}')

    qpos = candidate_qpos(model, joints)
    # IMPORTANT: unwrap BEFORE computing qvel, then validate the saved states.
    qpos, quat_flips, max_root_step_deg = unwrap_root_quaternion_signs(model, qpos)
    qvel = differentiate_qpos(model, t, qpos)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Nonfinite mapped states')

    # Strict knee equalities and independent joint-limit QC.
    max_eq = 0.0
    for ei in range(model.neq):
        if model.eq_type[ei] != mujoco.mjtEq.mjEQ_JOINT: continue
        jid1, jid2 = int(model.eq_obj1id[ei]), int(model.eq_obj2id[ei])
        if jid2 < 0: continue
        dep, src = int(model.jnt_qposadr[jid1]), int(model.jnt_qposadr[jid2])
        x = qpos[:, src]-model.qpos0[src]
        pred = model.qpos0[dep]+np.polynomial.polynomial.polyval(x, model.eq_data[ei, :5])
        max_eq = max(max_eq, float(np.max(abs(qpos[:, dep]-pred))))
    if max_eq > 1e-10:
        raise ValueError(f'Knee equality residual: {max_eq}')
    rootadr = int(model.joint('root').qposadr[0])
    quat = qpos[:, rootadr+3:rootadr+7]
    if np.max(abs(np.linalg.norm(quat, axis=1)-1)) > 1e-6 or np.any(np.sum(quat[:-1]*quat[1:], axis=1) < 0):
        raise ValueError('Root quaternion invalid/discontinuous')
    joint_warnings = []
    for name in LEG_JOINTS:
        j = model.joint(name)
        lo, hi = model.jnt_range[j.id]
        q = qpos[:, j.qposadr[0]]
        count = int(np.count_nonzero((q < lo-1e-8) | (q > hi+1e-8)))
        if count:
            joint_warnings.append({'joint':name,'frames_outside':count})

    columns = [c for c in grf.columns if c != 'time']
    gv = grf[columns].to_numpy(float)
    force_events, contacts = detect_force_events(tg, columns, gv, mt, list(mn), mx, float(t[0]), float(t[-1]))
    phi_r, phi_l, event_meta = build_phase_and_events(t, force_events)
    T = float(t[-1]-t[0])
    # Spatial progression in the OpenSim laboratory horizontal plane.
    pelvis_xz = joints[['pelvis_tx','pelvis_tz']].to_numpy(float)
    disp_xz = pelvis_xz[-1]-pelvis_xz[0]
    displacement = float(np.linalg.norm(disp_xz))
    mean_speed = displacement/T

    np.savez_compressed(str(out)+'.npz',
        time_s=t, time_since_start_s=t-t[0], stride_duration_s=np.float64(T),
        phase_right_01=phi_r, phase_right_rad=2*np.pi*phi_r,
        phase_left_01=phi_l, phase_left_rad=2*np.pi*phi_l,
        left_hs_s=np.float64(event_meta['left_hs_time_s']),
        walking_speed_est_m_s=np.float64(mean_speed),
        pelvis_horizontal_displacement_m=np.float64(displacement),
        qpos=qpos, qvel=qvel,
        original_joint_columns=np.asarray(list(joints.columns),dtype='U'),
        original_joint_values=joints.to_numpy(float),
        grf_time_s=tg, grf_channel_names=np.asarray(columns,dtype='U'),
        grf_values=gv, marker_time_s=mt,
        marker_names=np.asarray(mn,dtype='U'), marker_xyz_os_m=mx)

    fields = ['time_s','time_since_start_s','normalized_phase','plate','force_channel',
              'threshold_N','event','foot_side','side_method','nearest_distance_m',
              'side_margin_m','event_source']
    with Path(str(out)+'_events.csv').open('w',newline='',encoding='utf-8') as fp:
        w = csv.DictWriter(fp,fieldnames=fields)
        w.writeheader(); w.writerows(force_events)
    manifest = {
        'status':'CANDIDATE_ONLY_NOT_DYNAMICS_VALIDATED',
        'subject':subject.name, 'trial':trial, 'step':joint_path.stem,
        't0_s':float(t[0]), 't1_s':float(t[-1]), 'frames':int(len(t)),
        'stride_duration_s':T,
        'mean_speed_est_m_s':mean_speed,
        'speed_method':'magnitude horizontal OpenSim pelvis displacement / stride duration; not directly measured walking speed',
        'timing':'original samples and durations preserved; phases added without temporal resampling',
        'phase_boundary':'dataset right-HS-to-right-HS segmentation, see Boo et al 2025',
        'events':event_meta,
        'contacts_by_plate':contacts,
        'events_warning':'20N crossing only if observable; force plate zero is not foot airborne; foot side from CoP relative to heel/toe markers',
        'mapping':{'pelvis':'OpenSim intrinsic ZXY; T transform to MuJoCo',
                   'sign_reversed':['knee_angle_r','knee_angle_l','hip_adduction_l','hip_rotation_l'],
                   'other_leg_signs':'unchanged', 'dependent_knee':'MuJoCo equality polynomials'},
        's011_subtalar_flag':subject.name.upper()=='S011',
        'max_equality_residual':max_eq,
        'root_quaternion_sign_unwrap': {'frames_flipped':quat_flips,
                                        'max_physical_step_deg':max_root_step_deg,
                                        'method':'flip q[i] to maximize dot with q[i-1] before differentiation'},
        'independent_joint_limit_warnings':joint_warnings,
        'input_sha256':{
            'joint_mot':sha256(joint_path), 'grf_mot':sha256(grf_path),
            'trc':sha256(trc_path)},
    }
    Path(str(out)+'_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return {
        'subject':subject.name,'trial':trial,'step':joint_path.stem,
        'status':'EXPORTED' if not joint_warnings else 'EXPORTED_WITH_WARNINGS',
        'frames':len(t),'duration_s':T,'speed_est_m_s':mean_speed,
        'left_hs_source':event_meta['left_hs_source'],
        'left_hs_phase':event_meta['left_hs_phase_in_right_stride'],
        'right_TO_observed_count':len(event_meta['observed_right_toe_off_s']),
        'left_TO_observed_count':len(event_meta['observed_left_toe_off_s']),
        'force_events_count':len(force_events),
        'joint_warnings_count':len(joint_warnings),
        'flag_s011_subtalar':int(subject.name.upper()=='S011'),
        'root_quaternion_frames_flipped':quat_flips,
        'root_max_physical_step_deg':max_root_step_deg,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model',required=True,type=Path)
    ap.add_argument('--data-root',required=True,type=Path)
    ap.add_argument('--output-root',default=Path('gait120_myoleg_references'),type=Path)
    ap.add_argument('--start-subject',type=int,default=11)
    ap.add_argument('--end-subject',type=int,default=120)
    ap.add_argument('--limit',type=int,default=0,help='First N steps only (0=all)')
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--summary-name',default='batch_summary.csv',
                    help='CSV filename within output-root (use to preserve previous summaries)')
    ap.add_argument('--fail-fast',action='store_true')
    args=ap.parse_args()
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(args.model))
    steps=list(discover_steps(args.data_root,args.start_subject,args.end_subject))
    if args.limit: steps=steps[:args.limit]
    print(f'Found {len(steps)} LevelWalking steps; MyoLeg nq={model.nq}, nv={model.nv}')
    if not steps:
        print('No steps discovered. Check --data-root and unzip folder structure.')
        return
    rows=[]
    for i,(subject,trial,joint) in enumerate(steps,1):
        label=f'{subject.name}/{trial}/{joint.name}'
        try:
            row=export_step(model,subject,trial,joint,args.output_root,args.overwrite)
            print(f'[{i}/{len(steps)}] {label}: {row["status"]}')
        except Exception as e:
            row={'subject':subject.name,'trial':trial,'step':joint.stem,
                 'status':'FAILED','error':f'{type(e).__name__}: {e}'}
            print(f'[{i}/{len(steps)}] {label}: FAILED ({row["error"]})')
            if args.fail_fast:
                rows.append(row)
                break
        rows.append(row)
    args.output_root.mkdir(parents=True,exist_ok=True)
    all_fields=list(dict.fromkeys(k for row in rows for k in row))
    if Path(args.summary_name).name != args.summary_name:
        ap.error('--summary-name must be a filename, not a path')
    summary=args.output_root/args.summary_name
    with summary.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=all_fields)
        w.writeheader();w.writerows(rows)
    print(f'Summary: {summary}; {sum(r["status"].startswith("EXPORTED") for r in rows)} exported, '
          f'{sum(r["status"] == "FAILED" for r in rows)} failed.')

if __name__=='__main__':
    main()
