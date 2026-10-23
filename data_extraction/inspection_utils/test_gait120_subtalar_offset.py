"""Diagnostic only: does a constant per-subject subtalar offset explain the sign conflict?

Requires gait120_kinematic_compatibility.py in the same folder.
Keeps left hip-adduction/rotation sign corrections, no left MTP reversal.
Optimizes the LTOE rigid-marker variation in the toes_l body frame across
all selected trials of each subject. Evaluates LHEE separately as a check.
Does NOT modify MOT, TRC or model files.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from inspection.inspection_utils.gait120_kinematic_compatibility import (
    LEG_JOINTS, T, mapped_angle_deg, os_pelvis_rotation,
    read_mot, read_trc, update_knee_dependencies,
)

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)


def step_file(folder, extension):
    candidates = [p for p in folder.iterdir()
                  if p.is_file() and p.stem.lower() == 'step01'
                  and p.suffix.lower() == extension]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one Step01{extension} in {folder}; got {candidates}')
    return candidates[0]


def load_trial(model, root, subject, trial):
    base = root / subject
    mot = step_file(base / 'JointAngle' / 'LevelWalking' / trial, '.mot')
    trc = step_file(base / 'MotionCapture' / 'LevelWalking' / 'TRC' / trial, '.trc')
    joints = read_mot(mot)
    times, markers, xyz_os = read_trc(trc)
    if not np.allclose(joints.time.to_numpy(), times, atol=1e-8):
        raise ValueError(f'Timestamps mismatch: {mot} {trc}')
    if not np.isfinite(xyz_os).all() or not np.isfinite(joints.select_dtypes('number').to_numpy()).all():
        raise ValueError(f'Nonfinite data: {mot} {trc}')

    pos_os = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    R_os = os_pelvis_rotation(joints).as_matrix()
    root_quats = as_quat_wxyz(Rotation.from_matrix(T @ R_os @ T.T))
    jadr = {name: model.joint(name).qposadr[0] for name in LEG_JOINTS}
    rootadr = model.joint('root').qposadr[0]
    q0 = np.tile(model.qpos0, (len(joints), 1))
    q0[:, rootadr:rootadr + 3] = (T @ pos_os.T).T
    q0[:, rootadr + 3:rootadr + 7] = root_quats
    for name in LEG_JOINTS:
        angles_rad = np.deg2rad(mapped_angle_deg(joints, name))
        sign = -1 if name in ('hip_adduction_l', 'hip_rotation_l') else 1
        q0[:, jadr[name]] = sign * angles_rad

    expected = {}
    for marker_name in ('LTOE', 'LHEE'):
        if marker_name not in markers:
            raise ValueError(f'Missing marker {marker_name} in {trc}')
        expected[marker_name] = (T @ xyz_os[:, markers.index(marker_name), :].T).T
    return dict(qbase=q0, expected=expected, name=f'{subject}:{trial}',
                source_degrees=joints['subtalar_angle_l'].to_numpy())


def score_trial(model, trial_data, sign, offset_deg):
    """Marker-frame 3D variation RMS in mm, independently centered by marker."""
    data = mujoco.MjData(model)
    subtalar_adr = model.joint('subtalar_angle_l').qposadr[0]
    toe_local, heel_local = [], []
    offset = np.deg2rad(offset_deg)
    for qframe, a in zip(trial_data['qbase'], trial_data['source_degrees']):
        data.qpos[:] = qframe
        data.qpos[subtalar_adr] = sign * np.deg2rad(a) + offset
        update_knee_dependencies(model, data, mujoco)
        mujoco.mj_forward(model, data)
        for marker, bodyname, dest in (
            ('LTOE', 'toes_l', toe_local),
            ('LHEE', 'calcn_l', heel_local),
        ):
            b = data.body(bodyname)
            dest.append(b.xmat.reshape(3, 3).T @ (trial_data['expected'][marker][len(dest)] - b.xpos))

    def mse_mm2(points):
        ar = np.asarray(points)
        return 1e6 * np.mean(np.sum((ar - ar.mean(axis=0))**2, axis=1))

    return mse_mm2(toe_local), mse_mm2(heel_local)


def combined(model, trials, sign, offset):
    values = [score_trial(model, t, sign, offset) for t in trials]
    weights = np.array([len(t['qbase']) for t in trials], dtype=float)
    return np.average(np.asarray(values), axis=0, weights=weights)


def fit_subject(model, trials, max_offset):
    q_source = np.concatenate([t['source_degrees'] for t in trials])
    low, high = np.rad2deg(model.joint('subtalar_angle_l').range)
    print(f'  Recorded subtalar range: [{q_source.min():+.2f}, {q_source.max():+.2f}] deg')
    print(f'  MyoLeg joint limits: [{low:+.2f}, {high:+.2f}] deg')

    for sign in (+1, -1):
        angles = sign * q_source
        lb = max(-max_offset, low - angles.min())
        ub = min(max_offset, high - angles.max())
        if lb > ub:
            print(f'  sign {sign:+d}: no offset satisfies all joint limits')
            continue
        # 0 deg is the reference where possible; if outside joint limits report that.
        if lb <= 0 <= ub:
            base_toe, base_heel = np.sqrt(combined(model, trials, sign, 0.0))
            baseline = f'toe={base_toe:.2f}, heel={base_heel:.2f}'
        else:
            baseline = 'not within limits'

        if ub - lb < 1e-7:
            best_offset = lb
        else:
            sol = minimize_scalar(lambda b: combined(model, trials, sign, b)[0],
                                  bounds=(lb, ub), method='bounded',
                                  options={'xatol': 0.1})
            best_offset = float(sol.x)
        opt_toe, opt_heel = np.sqrt(combined(model, trials, sign, best_offset))
        print(f'  sign {sign:+d} | offset 0: {baseline} mm'
              f' | best b={best_offset:+.2f} deg'
              f' | toe={opt_toe:.2f}, heel={opt_heel:.2f} mm')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--trials', nargs='+', required=True,
                   help='E.g. S011:Trial01 S011:Trial02 S013:Trial02')
    p.add_argument('--max-offset', type=float, default=20.,
                   help='Maximum absolute subtalar offset to test, in degrees')
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))
    grouped = defaultdict(list)
    for identifier in args.trials:
        try:
            subject, trial_name = identifier.split(':', 1)
        except ValueError as exc:
            raise ValueError(f'Expected S011:Trial01, got {identifier}') from exc
        grouped[subject].append(load_trial(model, args.data_root, subject, trial_name))

    for subject, trials in grouped.items():
        print(f'\n=== {subject}: {", ".join(t["name"] for t in trials)} ===')
        fit_subject(model, trials, args.max_offset)
    print('\nDiagnostic only: an offset that minimizes marker variation is not proof of anatomical calibration.')


if __name__ == '__main__':
    main()
