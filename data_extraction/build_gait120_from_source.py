"""Rebuild Gait120 LevelWalking -> MyoLeg kinematic and *measured* bilateral GRF events.

SOURCE: Sxxx/{JointAngle,MotionCapture}/LevelWalking/Trialyy/stepzz.{mot,trc}

Usage (PowerShell):
  python build_gait120_from_source.py --model 'C:\\path\\myolegs.xml' \
    --data-root 'C:\\path\\Gait120_001_to_120' --output-root gait120_clean

Scientific distinctions:
  - The Gait120 paper defines 20 N crossings for RIGHT-foot HS/TO. Here the
    identical physical threshold is applied to both force plates, and each
    contact bout is assigned L/R from experimental foot markers and CoP.
  - Data QC (20 ms internal chatter rejection, geometric foot assignment) is
    our documented extension, NOT a claim about the paper's original code.
  - Events absent from the recorded plate forces remain MISSING, never estimated.
  - Adjacent steps in a trial are combined BEFORE GRF event detection; the
    extracted step's own time window is used only to assign observed events.
  - No changes are made to the source MOT/TRC files.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

THRESHOLD_N = 20.0
T = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
LEG_JOINTS = (
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
)
REVERSE_SIGNS = frozenset({'knee_angle_r', 'knee_angle_l',
                           'hip_adduction_l', 'hip_rotation_l'})
EVENT_FIELDS = ['time_s', 'time_since_start_s', 'normalized_phase', 'plate',
                'force_channel', 'threshold_N', 'event', 'foot_side',
                'side_method', 'nearest_distance_m', 'side_margin_m',
                'event_source', 'event_status']
RAW_FIELDS = EVENT_FIELDS + ['rejected_reason', 'bout_peak_N']
# Two visually verified false left-HS assignments from the original audit.
# This is project QC, NOT a rule from the Gait120 paper. Raw GRFs remain untouched.
KNOWN_FALSE_LEFT_HS = {
    ('S029', 'Trial05', 'step01'): 0.10181995784713907,
    ('S115', 'Trial03', 'step02'): 0.11788464413966933,
}


def read_mot(path: Path, *, angles: bool = False) -> pd.DataFrame:
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    try:
        header_end = next(i for i, line in enumerate(lines)
                          if line.strip().lower() == 'endheader')
    except StopIteration:
        raise ValueError(f'No endheader in {path}')
    if angles and not any(line.strip().lower() == 'indegrees=yes'
                          for line in lines[:header_end]):
        raise ValueError(f'Expected inDegrees=yes: {path}')
    df = pd.read_csv(path, sep=r'\s+', skiprows=header_end + 1)
    if 'time' not in df or len(df) < 2:
        raise ValueError(f'Missing time / too few data rows: {path}')
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite values: {path}')
    if not np.all(np.diff(df.time.to_numpy(dtype=float)) > 0):
        raise ValueError(f'Non-monotonic time: {path}')
    return df


def read_trc(path: Path):
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    keys, values = lines[1].split('\t'), lines[2].split('\t')
    units = dict(zip(keys, values)).get('Units', '').strip()
    if units not in ('mm', 'm'):
        raise ValueError(f'Unknown TRC units {units}: {path}')
    names = [x.strip() for x in lines[3].split('\t')[2:] if x.strip()]
    rows = []
    for line in lines[5:]:
        if not line.strip():
            continue
        fields = line.split('\t')
        rows.append([float(x) if x.strip() else np.nan
                     for x in fields[1:2 + 3 * len(names)]])
    arr = np.asarray(rows, dtype=float)
    tt = arr[:, 0]
    xyz = arr[:, 1:].reshape(len(arr), len(names), 3)
    if not np.isfinite(arr).all() or np.any(np.diff(tt) <= 0):
        raise ValueError(f'Invalid TRC: {path}')
    return tt, names, xyz / (1000.0 if units == 'mm' else 1.0)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve_partner(subject: Path, component: str, trial: str,
                    step: str, suffix: str) -> Path:
    base = subject / 'MotionCapture' / 'LevelWalking' / component / trial
    if not base.is_dir():
        raise FileNotFoundError(base)
    matches = [p for p in base.iterdir()
               if p.stem.casefold() == step.casefold() and p.suffix.lower() == suffix]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one {step}{suffix} in {base}; found {len(matches)}')
    return matches[0]


def discover(root: Path, first: int, last: int):
    if root.name.lower().startswith('s') and root.name[1:].isdigit():
        subjects = [root]
    else:
        subjects = list(root.glob('S[0-9][0-9][0-9]')) + list(root.glob('Gait120_*/S[0-9][0-9][0-9]'))
    seen = {}
    for subject in sorted(subjects):
        if not subject.is_dir():
            continue
        sid = int(subject.name[1:])
        if not first <= sid <= last:
            continue
        for tr in sorted((subject / 'JointAngle' / 'LevelWalking').glob('Trial*')):
            for jp in sorted(tr.glob('*.mot')):
                if not jp.stem.lower().startswith('step'):
                    continue
                key = (sid, tr.name.casefold(), jp.stem.casefold())
                if key in seen:
                    raise ValueError(f'Duplicate source {key}: {jp} vs {seen[key]}')
                seen[key] = jp
                yield subject, tr.name, jp


@dataclass
class Step:
    subject: Path
    trial: str
    joint_path: Path
    grf_path: Path
    marker_path: Path
    joints: pd.DataFrame
    grf: pd.DataFrame
    marker_t: np.ndarray
    marker_names: list[str]
    marker_xyz: np.ndarray

    @property
    def id(self):
        return f'{self.subject.name}/{self.trial}/{self.joint_path.stem.lower()}'

    @property
    def times(self):
        return self.joints.time.to_numpy(dtype=float)


def load_step(subject: Path, trial: str, jp: Path) -> Step:
    gp = resolve_partner(subject, 'MOT', trial, jp.stem, '.mot')
    mp = resolve_partner(subject, 'TRC', trial, jp.stem, '.trc')
    joints, grf = read_mot(jp, angles=True), read_mot(gp)
    mt, names, xyz = read_trc(mp)
    t, tg = joints.time.to_numpy(float), grf.time.to_numpy(float)
    if len(t) < 3 or len(mt) != len(t) or not np.allclose(t, mt, atol=1e-5, rtol=0):
        raise ValueError(f'TRC/JointAngle time mismatch: {jp}')
    if t[0] < tg[0] - 0.001 or t[-1] > tg[-1] + 0.001:
        raise ValueError(f'Force MOT does not cover joint trajectory: {gp}')
    needed = set(LEG_JOINTS) | {'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                                 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation'}
    if not needed.issubset(joints.columns):
        raise ValueError(f'Missing joint angles: {sorted(needed-set(joints.columns))}')
    for k in (1, 2):
        if f'ground_force{k}_vy' not in grf.columns:
            raise ValueError(f'Missing force plate {k} vertical component: {gp}')
    return Step(subject, trial, jp, gp, mp, joints, grf, mt, names, xyz)


def combine_tables(steps: list[Step], *, grf: bool):
    """Union step-level samples within a trial, without inventing samples in gaps."""
    times, vals, columns = [], [], None
    for step in steps:
        if grf:
            frame = step.grf
            cols = [x for x in frame.columns if x != 'time']
            t = frame.time.to_numpy(float)
            v = frame[cols].to_numpy(float)
        else:
            cols = step.marker_names
            t = step.marker_t
            v = step.marker_xyz
        if columns is None:
            columns = list(cols)
        if columns != list(cols):
            raise ValueError(f'Incompatible {"GRF" if grf else "TRC"} columns across trial')
        times.append(t)
        vals.append(v)
    t = np.concatenate(times)
    v = np.concatenate(vals)
    sort_idx = np.argsort(t, kind='stable')
    t, v = t[sort_idx], v[sort_idx]
    keep = np.concatenate([[True], np.diff(t) > 1e-7])
    # Small overlaps of the same force data are expected; check the signal agrees.
    duplicates = np.flatnonzero(~keep)
    conflicts = 0
    for ix in duplicates:
        if not np.allclose(v[ix], v[ix-1], atol=(0.1 if grf else 1e-4), rtol=0):
            conflicts += 1
    if conflicts:
        raise ValueError(f'{conflicts} inconsistent duplicate samples across steps')
    return t[keep], columns, v[keep]


def gap_blocks(t: np.ndarray, max_gap_s: float):
    borders = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    starts = np.concatenate([[0], borders]); ends = np.concatenate([borders, [len(t)]])
    return [(int(a), int(b)) for a,b in zip(starts, ends) if b > a]


def segments(x: np.ndarray):
    changes = np.flatnonzero(x[1:] != x[:-1]) + 1
    boundaries = np.concatenate([[0], changes, [len(x)]])
    return [(int(a), int(b), bool(x[a])) for a, b in zip(boundaries[:-1], boundaries[1:])]


def suppress_chatter(t: np.ndarray, force: np.ndarray, min_dwell_s: float):
    """Remove *interior* runs shorter than dwell if flanked by same state.

    This does not shift event times artificially: accepted crossings are later
    interpolated on the original unsmoothed force samples.
    """
    raw_state = force >= THRESHOLD_N
    state = raw_state.copy()
    if min_dwell_s <= 0:
        return state, raw_state
    for _ in range(100):
        runs = segments(state)
        changed = False
        for j in range(1, len(runs) - 1):
            a, b, value = runs[j]
            prev = runs[j-1][2]; nxt = runs[j+1][2]
            duration = float(t[b]-t[a]) if b < len(t) else float(t[b-1]-t[a] + np.median(np.diff(t)))
            if duration < min_dwell_s and prev == nxt and prev != value:
                state[a:b] = prev
                changed = True
                break
        if not changed:
            break
    return state, raw_state


def threshold_intersection(t: np.ndarray, f: np.ndarray, left: int, right: int) -> float:
    y0, y1 = float(f[left]), float(f[right])
    frac = (THRESHOLD_N-y0)/(y1-y0) if y0 != y1 else 0.5
    return float(t[left] + np.clip(frac, 0.0, 1.0)*(t[right]-t[left]))


def marker_at(time: float, marker_times: np.ndarray, marker_xyz: np.ndarray,
              max_dist_s: float = 0.035):
    ix = int(np.searchsorted(marker_times, time))
    i0 = min(max(ix-1, 0), len(marker_times)-1)
    i1 = min(max(ix, 0), len(marker_times)-1)
    if abs(marker_times[i0]-time) > max_dist_s and abs(marker_times[i1]-time) > max_dist_s:
        return None
    if i0 == i1:
        return marker_xyz[i0]
    span = float(marker_times[i1] - marker_times[i0])
    if span > 0.04:  # do not interpolate across a missing marker-recording interval
        return marker_xyz[i0] if abs(marker_times[i0]-time) < 0.01 else None
    frac = float(np.clip((time-marker_times[i0])/span, 0.0, 1.0))
    return (1-frac)*marker_xyz[i0] + frac*marker_xyz[i1]


def assign_bout_side(t, values, cols, plate, start, end, mt, marker_names, mx):
    """The *same side* belongs to the HS and TO of one loaded contact bout.

    Use CoP while plate actually carries load, not the CoP at near-zero TO.
    The geometry is only a labeling heuristic, not part of the paper's 20 N rule.
    """
    needed = ('LHEE','LTOE','RHEE','RTOE')
    if not all(k in marker_names for k in needed):
        return 'unassigned', 'missing_foot_markers', np.nan, np.nan
    force_idx = cols.index(f'ground_force{plate}_vy')
    cpnames = (f'ground_force{plate}_px', f'ground_force{plate}_pz')
    if not set(cpnames).issubset(cols):
        return 'unassigned', 'missing_cop', np.nan, np.nan
    cpidx = [cols.index(c) for c in cpnames]
    fi = values[start:end, force_idx]
    if len(fi) == 0:
        return 'unassigned', 'empty_bout', np.nan, np.nan
    peak = float(np.max(fi))
    indices = np.arange(start, end)
    candidates = indices[fi >= max(THRESHOLD_N, min(80.0, 0.25*peak))]
    if len(candidates) == 0:
        candidates = indices
    pick = candidates[np.unique(np.linspace(0, len(candidates)-1, min(9,len(candidates))).astype(int))]
    foot_dist = {'left': [], 'right': []}
    for idx in pick:
        xyz = marker_at(float(t[idx]), mt, mx)
        if xyz is None:
            continue
        cop = values[idx, cpidx]
        if not np.isfinite(cop).all():
            continue
        for side, prefix in (('left', 'L'), ('right','R')):
            a = xyz[marker_names.index(prefix+'HEE'), [0,2]]
            b = xyz[marker_names.index(prefix+'TOE'), [0,2]]
            ab = b-a
            frac = np.clip(np.dot(cop-a,ab)/max(float(np.dot(ab,ab)),1e-12), 0.0, 1.0)
            foot_dist[side].append(float(np.linalg.norm(cop-(a+frac*ab))))
    if min(len(v) for v in foot_dist.values()) == 0:
        return 'unassigned', 'no_matching_marker_samples', np.nan, np.nan
    dist = {side: float(np.median(ds)) for side, ds in foot_dist.items()}
    side = min(dist, key=dist.get)
    other = 'right' if side == 'left' else 'left'
    nearest = dist[side]
    margin = dist[other] - nearest
    if nearest <= 0.18 and margin >= 0.06:
        return side, 'loaded_bout_CoP_to_foot_markers', nearest, margin
    return 'unassigned', 'ambiguous_loaded_bout_geometry', nearest, margin


def raw_crossings(t, f, plate):
    raw_active = f >= THRESHOLD_N
    crossings = []
    for k in np.flatnonzero(raw_active[1:] != raw_active[:-1]) + 1:
        crossings.append({'time_s': threshold_intersection(t, f, k-1, k),
                          'plate': plate,
                          'event': 'HS_candidate' if raw_active[k] else 'TO_candidate'})
    return crossings


def detect_trial_events(steps, min_dwell_s: float):
    ft, cols, fv = combine_tables(steps, grf=True)
    mt, mn, mx = combine_tables(steps, grf=False)
    intervals = gap_blocks(ft, max_gap_s=max(0.01, 5.0*float(np.median(np.diff(ft)))))
    all_events, all_raw = [], []
    for plate in (1, 2):
        fi = cols.index(f'ground_force{plate}_vy')
        for a, b in intervals:
            if b-a < 3:
                continue
            tt, ff = ft[a:b], fv[a:b, fi]
            clean_state, raw_state = suppress_chatter(tt, ff, min_dwell_s)
            kept = []
            for start, end, loaded in segments(clean_state):
                if not loaded:
                    continue
                s, e = start+a, end+a
                side, method, dist, margin = assign_bout_side(
                    ft, fv, cols, plate, s, e, mt, mn, mx)
                peak = float(np.max(fv[s:e, fi]))
                # Crossings must have *both* sides recorded within this time block.
                for kind, idx_l, idx_r in (
                    ('HS_candidate', start-1, start),
                    ('TO_candidate', end-1, end),
                ):
                    if idx_l < 0 or idx_r >= len(tt):
                        continue
                    moment = threshold_intersection(tt, ff, idx_l, idx_r)
                    kept.append({
                        'time_s': moment, 'plate': plate,
                        'force_channel': f'ground_force{plate}_vy',
                        'threshold_N': THRESHOLD_N, 'event': kind,
                        'foot_side': side, 'side_method': method,
                        'nearest_distance_m': float(dist), 'side_margin_m': float(margin),
                        'event_source': 'observed_20N_crossing',
                        'event_status': 'MEASURED_CANDIDATE',
                        'rejected_reason': '', 'bout_peak_N': peak,
                    })
            # For auditing, keep every initial unfiltered threshold crossing.
            for raw in raw_crossings(tt, ff, plate):
                same = next((x for x in kept if x['event']==raw['event']
                             and abs(x['time_s']-raw['time_s']) < 1e-4), None)
                if same is not None:
                    all_raw.append(same.copy())
                else:
                    all_raw.append({
                        'time_s': raw['time_s'], 'plate': plate,
                        'force_channel': f'ground_force{plate}_vy',
                        'threshold_N': THRESHOLD_N, 'event': raw['event'],
                        'foot_side': 'unassigned', 'side_method': 'not_used',
                        'nearest_distance_m': np.nan, 'side_margin_m': np.nan,
                        'event_source': 'observed_raw_20N_crossing',
                        'event_status': 'REJECTED',
                        'rejected_reason': 'short_threshold_chatter', 'bout_peak_N': np.nan,
                    })
            all_events.extend(kept)
    all_events.sort(key=lambda e:(e['time_s'], e['plate'],e['event']))
    all_raw.sort(key=lambda e:(e['time_s'], e['plate'],e['event']))
    return all_events, all_raw, (ft, cols, fv), (mt, mn, mx)


def reject_two_verified_false_hs(events, raw, steps):
    """Apply exactly the two previously reviewed false-HS exclusions.

    This is applied at TRIAL level so trial_measured_events.csv and step-level
    event CSVs agree. The observed force crossing remains in trial_raw_crossings.
    """
    rejected=[]
    for step in steps:
        key=(step.subject.name.upper(),step.trial,step.joint_path.stem.lower())
        bad=KNOWN_FALSE_LEFT_HS.get(key)
        if bad is None:
            continue
        t0,t1=float(step.times[0]),float(step.times[-1]);duration=t1-t0
        for e in events:
            if (e['event']=='HS_candidate' and e['foot_side']=='left'
                and abs((e['time_s']-t0)/duration-bad)<0.025):
                rejected.append(e)
    reject_ids={id(e) for e in rejected}
    for x in raw:
        for e in rejected:
            if (x['plate']==e['plate'] and x['event']==e['event']
                and abs(x['time_s']-e['time_s']) < 1e-4):
                x['event_status']='REJECTED'
                x['rejected_reason']='known_false_left_HS_visual_QC'
    return [e for e in events if id(e) not in reject_ids], len(rejected)

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)

def build_qpos(model, joints: pd.DataFrame):
    import mujoco
    n = len(joints)
    q = np.repeat(model.qpos0[None,:], n, axis=0)
    root = int(model.joint('root').qposadr[0])
    position = joints[['pelvis_tx','pelvis_ty','pelvis_tz']].to_numpy(float)
    rotation_os = Rotation.from_euler('ZXY',
        joints[['pelvis_tilt','pelvis_list','pelvis_rotation']].to_numpy(float),
        degrees=True).as_matrix()
    q[:,root:root+3] = (T @ position.T).T
    q[:,root+3:root+7] = as_quat_wxyz(Rotation.from_matrix(T @ rotation_os @ T.T))
    for name in LEG_JOINTS:
        sign = -1.0 if name in REVERSE_SIGNS else 1.0
        q[:, int(model.joint(name).qposadr[0])] = sign*np.deg2rad(joints[name].to_numpy(float))
    # Equality constraints: dependent knee/patellar coordinates are not filled by mj_forward.
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        dep_j, src_j = int(model.eq_obj1id[i]), int(model.eq_obj2id[i])
        if src_j < 0:
            continue
        dep, src = int(model.jnt_qposadr[dep_j]), int(model.jnt_qposadr[src_j])
        x = q[:,src]-model.qpos0[src]
        q[:,dep] = model.qpos0[dep]+np.polynomial.polynomial.polyval(x,model.eq_data[i,:5])
    return q


def continuous_quats(model, qpos):
    q = qpos.copy()
    root = int(model.joint('root').qposadr[0])
    qs = q[:,root+3:root+7]
    if np.max(abs(np.linalg.norm(qs, axis=1)-1.0)) > 1e-6:
        raise ValueError('Non-unit root quaternions')
    flips = 0
    for i in range(1,len(qs)):
        if np.dot(qs[i-1],qs[i])<0:
            qs[i] *= -1
            flips += 1
    dots = np.clip(np.sum(qs[:-1]*qs[1:],axis=1), 0, 1)
    step_deg = np.rad2deg(2*np.arccos(dots))
    max_step = float(step_deg.max()) if len(step_deg) else 0.0
    if max_step > 15.0:
        raise ValueError(f'Physical root rotation jump of {max_step:.3f} degrees/frame')
    return q, {'flipped_quaternions': flips, 'max_physical_rotation_deg_per_frame': max_step}


def differentiate(model, t, q):
    import mujoco
    v = np.zeros((len(t), model.nv),dtype=float)
    for i in range(len(t)):
        a,b = max(0,i-1),min(len(t)-1,i+1)
        if a<b:
            mujoco.mj_differentiatePos(model, v[i], float(t[b]-t[a]), q[a],q[b])
    return v


def heading_align(model, q):
    root = int(model.joint('root').qposadr[0])
    xy_start = q[0,root:root+2].copy()
    delta = q[-1,root:root+2]-xy_start
    disp = float(np.linalg.norm(delta))
    if disp < 0.2:
        raise ValueError(f'Insufficient root displacement to align heading: {disp:.4f} m')
    yaw = float(np.arctan2(np.sin(np.pi/2-np.arctan2(delta[1],delta[0])),
                           np.cos(np.pi/2-np.arctan2(delta[1],delta[0]))))
    rot = Rotation.from_euler('z',yaw)
    r = q.copy()
    c,s=np.cos(yaw),np.sin(yaw)
    Rxy=np.array([[c,-s],[s,c]])
    r[:,root:root+2]=(q[:,root:root+2]-xy_start) @ Rxy.T
    q_xyz_w = np.roll(q[:,root+3:root+7],-1,axis=1)
    r[:,root+3:root+7] = np.roll((rot*Rotation.from_quat(q_xyz_w)).as_quat(),1,axis=1)
    r, stats = continuous_quats(model,r)
    if not np.array_equal(r[:,root+7:], q[:,root+7:]):
        raise AssertionError('Leg joints modified by heading normalization')
    if not np.allclose(r[-1,root:root+2]-r[0,root:root+2], [0, disp], atol=1e-8):
        raise AssertionError('Heading transform failed')
    return r, {'applied_world_z_yaw_deg': float(np.degrees(yaw)),
               'root_initial_original_xy_m': xy_start.tolist(),
               'target_progression': '+Y', 'horizontal_displacement_m': disp,
               'quaternion_continuity_after_rotation': stats,
               'raw_marker_and_grf_frame': 'original_OpenSim_lab_unrotated'}


def check_q(model, q):
    import mujoco
    max_eq=0.0
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT: continue
        dep_j, src_j = int(model.eq_obj1id[i]),int(model.eq_obj2id[i])
        if src_j < 0: continue
        dep,src=int(model.jnt_qposadr[dep_j]),int(model.jnt_qposadr[src_j])
        pred=model.qpos0[dep]+np.polynomial.polynomial.polyval(q[:,src]-model.qpos0[src],model.eq_data[i,:5])
        max_eq=max(max_eq,float(np.max(np.abs(q[:,dep]-pred))))
    if max_eq>1e-10: raise ValueError(f'Knee/patella constraint residual {max_eq}')
    out=[]
    for name in LEG_JOINTS:
        joint=model.joint(name)
        a=int(joint.qposadr[0]);lo,hi=model.jnt_range[joint.id]
        n=int(np.count_nonzero((q[:,a] < lo-1e-8)|(q[:,a]>hi+1e-8)))
        if n:out.append({'joint':name,'frames_outside':n})
    return {'max_equality_residual':max_eq,'independent_joint_warnings':out}


def step_events(events, t0, t1):
    # Neighboring step ranges are kept independent. An event on the final edge
    # can also appear as a boundary in the next step; shared raw event ID/time
    # remains visible and is not double-counted in the per-trial event report.
    return [e for e in events if t0-1e-7 <= e['time_s'] <= t1+1e-7]


def write_csv(path, rows, headers):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def add_relative_columns(items, t0, t1):
    duration=t1-t0
    return [{**e,'time_since_start_s':e['time_s']-t0,
             'normalized_phase':(e['time_s']-t0)/duration} for e in items]


def export_step(model, step: Step, events, raw_events, output, overwrite, align):
    stem=step.joint_path.stem.lower()
    dest=output/step.subject.name/step.trial/stem
    paths={'npz':dest.with_suffix('.npz'),
           'events':dest.with_name(stem+'_events.csv'),
           'raw':dest.with_name(stem+'_raw_crossings.csv'),
           'manifest':dest.with_name(stem+'_manifest.json')}
    if any(p.exists() for p in paths.values()) and not overwrite:
        raise FileExistsError(f'Output exists; use a new folder or --overwrite: {dest}')
    times=step.times
    t0,t1=float(times[0]),float(times[-1]);duration=t1-t0
    q=build_qpos(model,step.joints)
    q,quat_stats=continuous_quats(model,q)
    qc=check_q(model,q)
    heading={'target_progression':'not_normalized'}
    if align:
        q,heading=heading_align(model,q)
    v=differentiate(model,times,q)
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError('Invalid converted MyoLeg states')
    se=add_relative_columns(step_events(events,t0,t1),t0,t1)
    sr=add_relative_columns(step_events(raw_events,t0,t1),t0,t1)
    # Reject ONLY the two previously visually verified false HS annotations,
    # unless the 20 ms chatter filter already removed them. Retain the crossing
    # and its rejection reason in raw_crossings.csv.
    known_key=(step.subject.name.upper(), step.trial, stem)
    known_bad_phase=KNOWN_FALSE_LEFT_HS.get(known_key)
    manual_rejected=0
    if known_bad_phase is not None:
        rejected = [e for e in se if e['event']=='HS_candidate' and e['foot_side']=='left'
                    and abs(e['normalized_phase']-known_bad_phase)<0.025]
        rejected_times={e['time_s'] for e in rejected}
        se=[e for e in se if e['time_s'] not in rejected_times]
        manual_rejected=len(rejected)
        for e in sr:
            if e['event']=='HS_candidate' and any(abs(e['time_s']-tt)<1e-4 for tt in rejected_times):
                e['event_status']='REJECTED'
                e['rejected_reason']='known_false_left_HS_visual_QC'
    # No assumption of four observed events; we distinguish observed force
    # crossings, paper's nominal right-HS stride boundaries, and unknowns.
    labels={side:{kind:[float(e['time_s']) for e in se if e['foot_side']==side
                       and e['event']==f'{kind}_candidate'] for kind in ('HS','TO')}
            for side in ('left','right')}
    phase_right=(times-t0)/duration
    left_hs = labels['left']['HS']
    # A measured left phase offset exists ONLY when exactly one left HS is
    # observed in the midstride interval. Multiple/missing contacts are UNKNOWN.
    midhs=[x for x in left_hs if 0.25 <= (x-t0)/duration <= 0.75]
    lhs=float(midhs[0]) if len(midhs)==1 else np.nan
    phase_left=np.mod(phase_right-(lhs-t0)/duration,1.0) if np.isfinite(lhs) else np.full_like(phase_right,np.nan)
    pelvis=step.joints[['pelvis_tx','pelvis_tz']].to_numpy(float)
    disp=float(np.linalg.norm(pelvis[-1]-pelvis[0]))
    # Store ORIGINAL per-step force and marker samples, not rotated.
    grfcols=[c for c in step.grf.columns if c!='time']
    npz=dict(
        time_s=times, time_since_start_s=times-t0, stride_duration_s=np.float64(duration),
        phase_right_01=phase_right, phase_right_rad=2*np.pi*phase_right,
        phase_left_01=phase_left, phase_left_rad=2*np.pi*phase_left,
        left_hs_s=np.float64(lhs),
        walking_speed_est_m_s=np.float64(disp/duration),
        pelvis_horizontal_displacement_m=np.float64(disp),
        qpos=q,qvel=v,
        original_joint_columns=np.asarray(list(step.joints.columns),dtype='U'),
        original_joint_values=step.joints.to_numpy(float),
        grf_time_s=step.grf.time.to_numpy(float),
        grf_channel_names=np.asarray(grfcols,dtype='U'),
        grf_values=step.grf[grfcols].to_numpy(float),
        marker_time_s=step.marker_t,
        marker_names=np.asarray(step.marker_names,dtype='U'),marker_xyz_os_m=step.marker_xyz)
    manifest={
        'source': step.id, 'status':'KINEMATIC_REFERENCE_NOT_DYNAMICS_VALIDATED',
        't0_s':t0,'t1_s':t1,'stride_duration_s':duration,'frames':len(times),
        'speed_est_m_s':disp/duration,
        'mapping': {'open_sim_to_mujoco_world_matrix_T':T.tolist(),
                    'pelvis_euler_order':'intrinsic ZXY (tilt,list,rotation)',
                    'joint_sign_reversals':sorted(REVERSE_SIGNS),
                    'dependent_knee_coordinates':'MuJoCo compiled equality polynomials',
                    'subject_specific_rescaling': False},
        'quaternion_fix':quat_stats,'kinematic_qc':qc,
        'heading_alignment':heading if align else None,
        's011_subtalar_flag':step.subject.name.upper()=='S011',
        'known_two_step_false_HS_case': known_key in KNOWN_FALSE_LEFT_HS,
        'paper_method':'For level walking, paper uses 20N right HS upward and right TO downward crossings.',
        'bilateral_extension':'The same force criterion was applied to both plates, with foot assignment using loaded-bout CoP and markers.',
        'event_filter':'20 ms (configurable) interior threshold-chatter removal; project adjustment not stated in paper',
        'events':{
            'observed_by_foot':labels,
            'left_hs_time_s':float(lhs) if np.isfinite(lhs) else None,
            'left_hs_source':'observed_unique_midstride_GRF' if np.isfinite(lhs) else 'NOT_OBSERVED_OR_AMBIGUOUS',
            'right_hs_boundaries_s':[t0,t1],
            'right_hs_boundary_source':'source_step_segmentation_not_always_force_verified',
            'missing_event_policy':'Never replace unavailable force crossings with typical phase estimates',
            'unassigned_observed_count':sum(e['foot_side']=='unassigned' for e in se),
            'rejected_short_crossings_count':sum(e['rejected_reason']=='short_threshold_chatter' for e in sr),
            'manually_rejected_known_false_left_HS':manual_rejected,
        },
        'raw_markers_and_grf_still_in_OpenSim_lab_frame':True,
        'provenance_sha256':{'joint_mot':sha256(step.joint_path),
                             'force_mot':sha256(step.grf_path),'marker_trc':sha256(step.marker_path)},
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(paths['npz'],**npz)
    write_csv(paths['events'],se,EVENT_FIELDS)
    write_csv(paths['raw'],sr,RAW_FIELDS)
    paths['manifest'].write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {'subject':step.subject.name,'trial':step.trial,'step':stem,
            'status':'EXPORTED' if not qc['independent_joint_warnings'] else 'EXPORTED_WITH_JOINT_WARNINGS',
            'frames':len(times),'stride_duration_s':duration,'speed_est_m_s':disp/duration,
            'left_HS_measured':len(labels['left']['HS']),'left_TO_measured':len(labels['left']['TO']),
            'right_HS_measured':len(labels['right']['HS']),'right_TO_measured':len(labels['right']['TO']),
            'left_HS_selected_for_phase':int(np.isfinite(lhs)),
            'unassigned_crossings':sum(e['foot_side']=='unassigned' for e in se),
            'rejected_short_crossings':sum(e['rejected_reason']=='short_threshold_chatter' for e in sr),
            'manually_rejected_known_false_left_HS':manual_rejected,
            'joint_limit_warnings':len(qc['independent_joint_warnings']),
            'heading_yaw_deg':heading.get('applied_world_z_yaw_deg','')}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--start-subject',type=int,default=11)
    parser.add_argument('--end-subject',type=int,default=120)
    parser.add_argument('--limit',type=int,default=0,help='For trial-aware detection, limits COMPLETE trials, not individual steps')
    parser.add_argument('--min-dwell-ms',type=float,default=20.0,
                        help='Remove internal force-state excursions shorter than this duration; 0 disables')
    parser.add_argument('--no-heading',action='store_true',help='Leave walking direction in original converted world frame')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.min_dwell_ms < 0:
        parser.error('--min-dwell-ms must be >= 0')
    if args.output_root.resolve() == args.data_root.resolve() or args.data_root.resolve() in args.output_root.resolve().parents:
        parser.error('Output must not be the input directory or inside it')
    import mujoco
    model=mujoco.MjModel.from_xml_path(str(args.model))
    selected=list(discover(args.data_root,args.start_subject,args.end_subject))
    groups={}
    for subject,trial,jp in selected:
        groups.setdefault((subject.name,trial),[]).append((subject,trial,jp))
    trial_groups=list(groups.items())
    if args.limit:
        trial_groups=trial_groups[:args.limit]
    print(f'Found {len(trial_groups)} complete trial group(s), total {sum(len(v) for _,v in trial_groups)} steps')
    summary=[]
    for i,((subject_name, trial),refs) in enumerate(trial_groups,1):
        print(f'[{i}/{len(trial_groups)}] {subject_name}/{trial}')
        try:
            steps=[load_step(*entry) for entry in refs]
            events,raw,_,_=detect_trial_events(steps,args.min_dwell_ms/1000.)
            events, rejected_known = reject_two_verified_false_hs(events,raw,steps)
            if rejected_known:
                print(f'   Removed {rejected_known} previously verified false left-HS annotation(s); raw crossing retained')
            trial_dir=args.output_root/subject_name/trial
            trial_dir.mkdir(parents=True,exist_ok=True)
            write_csv(trial_dir/'trial_measured_events.csv',events,RAW_FIELDS)
            write_csv(trial_dir/'trial_raw_crossings.csv',raw,RAW_FIELDS)
            for step in steps:
                try:
                    row=export_step(model,step,events,raw,args.output_root,args.overwrite,not args.no_heading)
                except Exception as exc:
                    row={'subject':subject_name,'trial':trial,'step':step.joint_path.stem.lower(),
                         'status':'FAILED','error':f'{type(exc).__name__}: {exc}'}
                summary.append(row)
                print(f'   {row["step"]}: {row["status"]} '
                      f'LHS={row.get("left_HS_measured","?")} LTO={row.get("left_TO_measured","?")} '
                      f'RHS={row.get("right_HS_measured","?")} RTO={row.get("right_TO_measured","?")}')
        except Exception as exc:
            for subj,tr,jp in refs:
                summary.append({'subject':subj.name,'trial':tr,'step':jp.stem.lower(),
                                'status':'FAILED','error':f'{type(exc).__name__}: {exc}'})
            print(f'   FAILED: {type(exc).__name__}: {exc}')
    args.output_root.mkdir(parents=True,exist_ok=True)
    headers=list(dict.fromkeys(k for r in summary for k in r))
    write_csv(args.output_root/'batch_summary.csv',summary,headers)
    print('Summary:',args.output_root/'batch_summary.csv')
    print('Exported:',sum(r['status'].startswith('EXPORTED') for r in summary),
          '| Failed:',sum(r['status']=='FAILED' for r in summary))

if __name__=='__main__':
    main()
