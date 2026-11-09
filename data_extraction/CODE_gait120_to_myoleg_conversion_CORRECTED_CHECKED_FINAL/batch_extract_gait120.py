"""Batch export Gait120 LevelWalking steps to MyoLeg reference arrays.

Uses the EXISTING, tested mapping functions from:
  gait120_kinematic_compatibility.py
  check_gait120_reference_feasibility.py

No original data are altered. This is a candidate reference exporter, NOT a
validation of dynamics or the source IK. All strides stay at their original
sampling rate and original duration. A gait-phase coordinate is added.

Examples:
  python batch_extract_gait120.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --data-root "..\\Gait120_011_to_120" --output-root references_gait120

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
import pandas as pd

import mujoco
from scipy.spatial.transform import Rotation

"""
# T is the OpenSim-to-MyoLeg rotation matrix for pelvis translation and rotation.
    # Necessary because OpenSim uses Y as its vertical axis, whereas MuJoCo uses Z.
# The OpenSim pelvis rotation is applied in ZXY order, then rotated into MyoLeg's frame.
# The OpenSim pelvis translation is rotated into MyoLeg's frame, then added to the root position.
"""
T = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])

LEG_JOINTS = (
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
)

'''
    The OpenSim joint angles describe the reconstructed motion; 
    the markers provide independent experimental information; 
    the forces provide measured contact-related signals. 
    Keep all three for subsequent validation. 
    TRC positions are converted from mm to m when needed.
'''
def read_mot(filename):
    lines = Path(filename).read_text(encoding='utf-8-sig').splitlines()
    end = next(i for i, line in enumerate(lines) if line.strip().lower() == 'endheader')
    assert any('indegrees=yes' == line.strip().lower() for line in lines[:end]), 'Expected angles in degrees'
    return pd.read_csv(filename, sep=r'\s+', skiprows=end + 1)


def read_trc(filename):
    lines = Path(filename).read_text(encoding='utf-8-sig').splitlines()
    names = [x.strip() for x in lines[3].split('\t')[2:] if x.strip()]
    keys, values = lines[1].split('\t'), lines[2].split('\t')
    units = dict(zip(keys, values))['Units']
    out = []
    for line in lines[5:]:
        if line.strip():
            fields = line.split('\t')
            out.append([float(x) if x.strip() else np.nan for x in fields[1:2 + 3*len(names)]])
    arr = np.asarray(out)
    t, xyz = arr[:, 0], arr[:, 1:].reshape(len(arr), len(names), 3)
    assert units in ('mm', 'm'), f'Unexpected TRC units: {units}'
    return t, names, xyz / (1000. if units == 'mm' else 1.)

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

    # Verify that the motion is physically continuous, not just a sign flip. Reject large physical jumps.
    dots = np.sum(quats[:-1] * quats[1:], axis=1)
    if np.any(dots < 0):
        raise ValueError('Quaternion sign unwrapping failed')
    # A sign flip is harmless, but a large physical rotation jump is not.
    steps_deg = np.rad2deg(2 * np.arccos(np.clip(dots, 0.0, 1.0)))  # Here we compute the physical rotation between 2 consecutive frames
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


def on_off_crossings(time, force, threshold=THRESHOLD):
    """Return observed force threshold crossings; no fabricated end events.
    This is based on the Gait120 paper's 20N threshold"""
    active = np.asarray(force) >= threshold
    crossings = []
    for idx in np.flatnonzero(active[1:] != active[:-1]) + 1:
        y0, y1 = float(force[idx-1]), float(force[idx])
        frac = (threshold-y0)/(y1-y0) if y1 != y0 else 1.0
        tc = float(time[idx-1] + np.clip(frac, 0, 1)*(time[idx]-time[idx-1]))
        crossings.append((tc, 'HS_candidate' if active[idx] else 'TO_candidate'))
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
            # Skip events outside the dataset boundaries, with a small tolerance for numerical noise.
            if not (t0-0.002 <= when <= t1+0.002):
                continue

            # If the event is a heel strike candidate, assign the foot side based on CoP proximity.
            # If the HS candidate was assigned to a foot, then the next TO candidate is assigned to the same foot 
            # regardless of CoP proximity, to avoid mislabeling a TO event that occurs after a foot has left the plate.
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
                        and 0.3 < e['normalized_phase'] < 0.7]
    if observed_left_hs:
        # If multiple, select the one closest to 50% of the right stride;
        # retain EVERY observation in the event CSV regardless.
        lhs = min(observed_left_hs, key=lambda e: abs(e['normalized_phase']-0.5))
        left_hs_time = lhs['time_s']
        left_method = 'observed_20N_crossing_cop_labeled'
    else:
        # If no left HS is observed, assume a half-stride shift from the right HS.
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

    # Right toe-off: use observed event, or assume 60% of the stride.
    observed_right_to = [e for e in force_events
                         if e['event'] == 'TO_candidate' and e['foot_side'] == 'right'
                         and 0 <= e['normalized_phase'] <= 1]

    if observed_right_to:
        rto = min(observed_right_to, key=lambda e: abs(e['normalized_phase'] - 0.6))
        right_to_phase = float(rto['normalized_phase'])
        right_to_source = 'observed_20N_crossing'
    else:
        right_to_phase = 0.6
        right_to_source = 'ASSUMED_0.6_not_measured'

    # Left toe-off: use observed event, or assume 10% of the right stride.
    observed_left_to = [e for e in force_events
                        if e['event'] == 'TO_candidate' and e['foot_side'] == 'left'
                        and 0 <= e['normalized_phase'] <= 1]

    if observed_left_to:
        lto = min(observed_left_to, key=lambda e: abs(e['normalized_phase'] - 0.1))
        left_to_phase = float(lto['normalized_phase'])
        left_to_source = 'observed_20N_crossing'
    else:
        left_to_phase = 0.1
        left_to_source = 'ASSUMED_0.1_not_measured'

    return phase_right, phase_left, {
        'stride_start_right_hs_s': t0, 'stride_end_right_hs_s': t1,
        'right_hs_boundary_sources': boundary_sources,
        'left_hs_time_s': float(left_hs_time), 'left_hs_source': left_method,
        'left_hs_phase_in_right_stride': float(left_offset),
        'left_toe_off_phase': left_to_phase,
        'left_toe_off_source': left_to_source,
        'right_toe_off_phase': right_to_phase,
        'right_toe_off_source': right_to_source,
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


def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)

def os_pelvis_rotation(joints, convention='ZXY'):
    """
    Reconstruct the OpenSim pelvis rotation matrix from the OpenSim joint angles.
    * OpenSim provide three pelvis angles: tilt, list, and rotation, in degrees.
    * We reconstruct the rotation matrix using the intrinsic OpenSim ZXY convention:
        $$R_{OS}=R_z(\text{tilt})R_x(\text{list})R_y(\text{rotation})$$
    """
    return Rotation.from_euler(convention, joints[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].to_numpy(), degrees=True)

def mapped_angle_deg(joints, name):
    values = joints[name].to_numpy()
    # Knee flexion sign is flipped between openSim and MyoLeg. OpenSim flexion is positive, MyoLeg flexion is negative.
    # ALSO : we need to apply a sign change for the LEFT hip adduction and LEFT hip rotation angles to match the MyoLeg coordinate system.
    #        Indeed Myoleg left hip adduction and rotation are defined in -X and -Y respectively, while OpenSim left hip adduction and rotation are defined in +X and +Y respectively.
    #        We should not do this for the right side because the right hip adduction and rotation are defined in +X and +Y respectively in both MyoLeg and OpenSim.
    return -values if name in ('knee_angle_l', 'knee_angle_r', 'hip_adduction_l', 'hip_rotation_l') else values

def update_knee_dependencies(model, data, mj):
    """Fill both knees' 14 dependent qpos values from the compiled equality constraints."""
    for i in range(model.neq): # We iterate over all the equality constraints in the model and we select those coupling two joint coordinates
        if model.eq_type[i] != mj.mjtEq.mjEQ_JOINT:
            continue
        dependent = int(model.eq_obj1id[i]) # joint whose coordinate must be computed
        source = int(model.eq_obj2id[i]) # joint driving that coordinate (typically knee flexion)
        if source < 0:
            continue
        dep_index = int(model.jnt_qposadr[dependent])
        src_index = int(model.jnt_qposadr[source])
        """
        The dependent joint coordinate is computed from the source joint coordinate using a polynomial relationship defined in the model's equality constraints. The polynomial coefficients are stored in model.eq_data[i, :5].
        Mathematically, the dependent joint coordinate is computed as: 

        $$
        data.qpos[dep_index] =  model.qpos0[dep_index] + c_0 + c_1 delta +c_2 delta^2 + c_3 delta^3 + c_4 delta^4
        $$ 

        The polynomial is evaluated using numpy's polyval function, which takes the coefficients and the value of the source joint coordinate (delta) as input and returns the value of the dependent joint coordinate.
        The delta is computed as the difference between the current value of the source joint coordinate (data.qpos[src_index]) and the default value of the source joint coordinate (model.qpos0[src_index]). 
        The dependent joint coordinate is then updated in the data.qpos array by adding the computed value to the default value of the dependent joint coordinate (model.qpos0[dep_index]).
        """
        delta = data.qpos[src_index] - model.qpos0[src_index]
        data.qpos[dep_index] = model.qpos0[dep_index] + np.polynomial.polynomial.polyval(delta, model.eq_data[i, :5])

def candidate_qpos(model, joints):
    # Takes care of the OpenSim-to-Myoleg conversion, including the pelvis rotation and knee dependencies.

    """
        * n: Number of recorded kinematic frames.
        * model.qpos0: MyoLeg's default joint configuration.
        * q: Creates an (n, 35) matrix initialized with that default configuration.
        * root: Finds where the floating-base coordinates begin in qpos.
        * p: Extracts the experimental pelvis translation from the OpenSim joint angles.
        * R_os: Computes the OpenSim pelvis rotation matrix from the OpenSim joint angles.
    """
    n = len(joints)
    q = np.repeat(model.qpos0[None, :], n, axis=0)
    root = int(model.joint('root').qposadr[0])
    p = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    R_os = os_pelvis_rotation(joints).as_matrix()

    # Apply the transformation to the pelvis translation and rotation.
    q[:, root:root + 3] = (T @ p.T).T # T is the transformation matrix that converts OpenSim coordinates to MyoLeg coordinates.
    q[:, root + 3:root + 7] = as_quat_wxyz(Rotation.from_matrix(T @ R_os @ T.T)) # Convert the OpenSim pelvis rotation to a quaternion in MyoLeg's coordinate system.
    """
    Note : Multiply by T on both sides because we're expressing the same physical rotation in a different coordinate basis.
    - T.T: converts coordinates from the MuJoCo basis into the OpenSim basis.
    - R_os: applies the original rotation.
    - T: converts the result back into the MuJoCo basis.
    """

    # Convert all angles from degrees to radians and apply a sign change appropriate for each joint.
    for name in LEG_JOINTS:
        value = np.deg2rad(mapped_angle_deg(joints, name))
        q[:, int(model.joint(name).qposadr[0])] = value

    """
    Note : At this point, we have assigned:
    * 7 independent floating-root coordinates (3 translations + 4 quaternion components)
    * 14 independent leg-joint coordinates (7 per leg)
    The remaining 14 dependent knee/patellar coordinates are derived from the MuJoCo equality constraints.
    We need to reconstruct these coordinates because in Gait120, the modified Rajagopal model has one rotation degree of freedm per knee.
    However, the MyoLeg model has 8 dependent coordinates (tibial translations and patellar movements) per knee, which are derived from the 6 independent coordinates per knee.
    The dependent coordinates are derived from the independent coordinates through polynomial relationships defined in MyoLeg, using the MuJoCo equality constraints.
    The update_knee_dependencies function takes care of this derivation.
    """
    data = mujoco.MjData(model)
    for k in range(n):
        data.qpos[:] = q[k] 
        # Reconstructing the dependent knee and patellar coordinates
        update_knee_dependencies(model, data, mujoco)
        q[k] = data.qpos

    # We successfully constructed the complete 35-dimensional MyoLeg configuration for every frame.
    return q

def differentiate_qpos(model, t, q):
    """Quaternion-aware finite differences, rather than qpos subtraction."""
    v = np.zeros((len(q), model.nv))
    for k in range(len(q)):
        a = max(0, k - 1)
        b = min(len(q) - 1, k + 1)
        if a == b:
            continue
        mujoco.mj_differentiatePos(model, v[k], float(t[b] - t[a]), q[a], q[b])
    """
    For every frame, we obtain:
    - Root linear velocity (m/s).
    - Root angular velocity (rad/s).
    - Independent leg-joint angular velocities (rad/s).
    - Dependent knee/patellar coordinate velocities (rad/s or m/s, depending on the joint).
    """
    return v

def export_step(model, subject, trial, joint_path, dest, overwrite):
    """For each step, load the joint angles, GRF and TRC data, and export the reference arrays"""
    
    out = dest/subject.name/trial/joint_path.stem.lower()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.npz').exists() and not overwrite:
        return {'subject':subject.name, 'trial':trial, 'step':joint_path.stem,
                'status':'SKIPPED_EXISTS'}

    # Locate the TRC and GRF MOT files that correspond to the joint MOT file. 
    # The TRC and GRF files are normally in the same trial folder as the joint MOT file, with the same stem name but different suffixes.
    trc_path = find_partner(subject, 'TRC', trial, joint_path.stem, '.trc')
    grf_path = find_partner(subject, 'MOT', trial, joint_path.stem, '.mot')

    """Read the joint angles, GRF and TRC data into pandas DataFrames and numpy arrays.
    * Joints : pandas Dataframe containing OpenSim joint angles (generalized coordinates)in degrees, with time in seconds as the first column.
    * GRF : pandas Dataframe containing ground reaction forces in Newtons, with time in seconds as the first column.
    * mt : NumPy array containing the time stamps of the TRC data in seconds.
    * mn : list of marker names corresponding to the TRC data (LHEE, RHEE, LTOE, RTOE, etc.).
    * mx : NumPy array containing the marker positions in meters, with shape (num_frames, num_markers, 3).
    """
    joints, grf = read_mot(joint_path), read_mot(grf_path) # reads the tabular values and requires an inDegrees=yes declaration.
    mt, mn, mx = read_trc(trc_path) # checks whether marker coordinates are expressed in millimeters or meters and converts them to meters.
    t = joints['time'].to_numpy(float)
    tg = grf['time'].to_numpy(float)

    # Safety checks: ensure that the time stamps are valid and consistent across the joint, GRF, and TRC data.
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

    # STEP 4: Map the OpenSim joint angles to MyoLeg generalized coordinates (qpos) and compute the corresponding velocities (qvel).
    # STEP 4.1 : Construct the complete 35-dimensional MyoLeg configuration for every frame.
    qpos = candidate_qpos(model, joints)

    # STEP 4.2 : Compute the corresponding velocities (qvel) using quaternion-aware finite differences, rather than simple qpos subtraction.
    # IMPORTANT: unwrap BEFORE computing qvel, then validate the saved states. 
    #            Indeed a quat q and -q describe the same rotation, but the finite-difference velocity will be different. 
    #            Therefore, we make the quanternion continuous by unwrapping the signs (using dot products) before computing the velocities.
    #            This was a problem for subject060
    qpos, quat_flips, max_root_step_deg = unwrap_root_quaternion_signs(model, qpos)
    qvel = differentiate_qpos(model, t, qpos)
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError('Nonfinite mapped states')

    # Safety check : Verify the knee equalities equality constraints and independent joint-limit QC
    # Because we calculated the dependentknee coordinates satisfying the cnstraints sequentially.
    # Here we check if now after calculation all the constraints are satisfied simultaneously
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

    # STEP 5: Detect force events and assign foot sides based on CoP proximity to heel/toe markers.
    columns = [c for c in grf.columns if c != 'time']
    gv = grf[columns].to_numpy(float)
    force_events, contacts = detect_force_events(tg, columns, gv, mt, list(mn), mx, float(t[0]), float(t[-1]))

    # STEP 6: Build the normalized gait phase for each leg based on the right-HS-to-right-HS segmentation documented in the Gait120 paper.
    phi_r, phi_l, event_meta = build_phase_and_events(t, force_events)

    T = float(t[-1]-t[0])
    # Spatial progression in the OpenSim laboratory horizontal plane (X-Z in OpenSim).
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


    """
    Unlike the CSV, the manifest describes the entire stride. It records:
        - Recording information: Subject, trial, stride duration, number of frames, walking-speed estimate.
        - Conversion information: Pelvis transformation, reversed joint signs, dependent knee calculation.
        - Quality control: Joint-limit warnings, knee-constraint residual, quaternion corrections.
        - Gait events: Selected left HS, right HS boundary reliability, observed toe-off candidates.
        - Reproducibility: SHA-256 hashes of the original MOT, GRF and TRC files.
         (The SHA-256 hashes are particularly useful: they allow us to verify later that the input recordings haven't changed.)
    """
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

    # -------- Initialization ------------

    # STEP 1: Loading myoleg --> We need the model's joint definitions, coordinate ordering
    # and constraints to convert the experimental kinematics correctly
    model=mujoco.MjModel.from_xml_path(str(args.model))

    # STEP 2: Discover all the steps to be processed, based on the data root and subject range
    # Until now, we have not loaded any data, we just identified the recordings to be processed
    steps=list(discover_steps(args.data_root,args.start_subject,args.end_subject))

    # If the user specified a limit, we will only process the first N steps
    if args.limit: steps=steps[:args.limit]
    
    # Safety check: if no steps were found, we will not proceed 
    print(f'Found {len(steps)} LevelWalking steps; MyoLeg nq={model.nq}, nv={model.nv}')
    if not steps:
        print('No steps discovered. Check --data-root and unzip folder structure.')
        return
    
    rows=[]
    for i,(subject,trial,joint) in enumerate(steps,1):
        label=f'{subject.name}/{trial}/{joint.name}'
        try:
            # STEP 3 : Process one Gait120 recording at a time: 
            # load the joint angles, GRF and TRC data, and export the reference arrays
            """ 
            The files are exported to the output root in a structured folder hierarchy:
                step01.npz : Actual numerical reference data
                step01_events.csv : Force-event observations
                step01_manifest.json : Metadata, assumptions and validation results
            """
            row = export_step(model,subject,trial,joint,args.output_root,args.overwrite)
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
