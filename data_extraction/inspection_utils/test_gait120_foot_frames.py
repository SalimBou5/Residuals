"""Diagnose Gait120 -> MyoLeg foot mapping without editing source data.

Uses your existing `gait120_kinematic_compatibility.py` beside this file.

Example (Windows, one line):
python test_gait120_foot_frames.py --model "C:\\path\\to\\myolegs.xml" --data-root "C:\\path\\to\\Gait120_011_to_020" --trials S011:Trial01 S011:Trial02 S013:Trial02 S019:Trial05

Question: Is the left subtalar sign truly reversed, or did comparing a toe
marker within the calcaneus frame confound the earlier experiment?

Experimental mappings: left hip adduction and rotation signs reversed in all
conditions; independent subtalar and MTP sign reversals are varied factorially.
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from inspection.inspection_utils.gait120_kinematic_compatibility import (
    LEG_JOINTS, T, mapped_angle_deg, os_pelvis_rotation, read_mot, read_trc,
    update_knee_dependencies,
)

# Foot landmarks considered in both plausible rigid-body frames.
# LTOE/RTOE may be on the forefoot rather than on a rigid calcaneus segment.
TESTS = (
    ('left heel | calcaneus', 'LHEE', 'calcn_l'),
    ('left toe  | calcaneus', 'LTOE', 'calcn_l'),
    ('left toe  | toes', 'LTOE', 'toes_l'),
    ('right heel| calcaneus', 'RHEE', 'calcn_r'),
    ('right toe | calcaneus', 'RTOE', 'calcn_r'),
    ('right toe | toes', 'RTOE', 'toes_r'),
)


def find_step(folder, extension):
    matches = [p for p in folder.iterdir()
               if p.is_file() and p.stem.lower() == 'step01'
               and p.suffix.lower() == extension.lower()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected exactly one step01{extension} in {folder}, found {matches}')
    return matches[0]

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)

def run_trial(model, joints, times, marker_names, trc):
    if not np.allclose(joints.time.to_numpy(), times, atol=1e-8):
        raise ValueError('Joint-angle MOT and TRC timestamps do not match.')
    if not np.isfinite(trc).all():
        raise ValueError('Missing / non-finite TRC marker values')

    observed = {m: (T @ trc[:, marker_names.index(m), :].T).T
                for _, m, _ in TESTS}

    qangles = {name: np.deg2rad(mapped_angle_deg(joints, name))
               for name in LEG_JOINTS}
    pelvis_R = os_pelvis_rotation(joints, 'ZXY').as_matrix()
    pelvis_p = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    rootadr = int(model.joint('root').qposadr[0])
    jointadr = {name: int(model.joint(name).qposadr[0]) for name in LEG_JOINTS}
    quats = as_quat_wxyz(Rotation.from_matrix(T @ pelvis_R @ T.T))

    # Always apply the two repeatedly supported hip sign corrections.
    fixed_flips = {'hip_adduction_l', 'hip_rotation_l'}
    outcomes = {}

    for flip_subtalar in (False, True):
        for flip_mtp in (False, True):
            flips = fixed_flips.copy()
            if flip_subtalar:
                flips.add('subtalar_angle_l')
            if flip_mtp:
                flips.add('mtp_angle_l')
            transformed = {k: [] for k, _, _ in TESTS}
            data = mujoco.MjData(model)

            for frame in range(len(joints)):
                data.qpos[:] = model.qpos0
                data.qpos[rootadr:rootadr + 3] = T @ pelvis_p[frame]
                data.qpos[rootadr + 3:rootadr + 7] = quats[frame]
                for name in LEG_JOINTS:
                    data.qpos[jointadr[name]] = (-1 if name in flips else 1) * qangles[name][frame]
                update_knee_dependencies(model, data, mujoco)
                mujoco.mj_forward(model, data)

                for label, marker, body_name in TESTS:
                    body = data.body(body_name)
                    xyz_local = body.xmat.reshape(3, 3).T @ (observed[marker][frame] - body.xpos)
                    transformed[label].append(xyz_local)

            rms = {}
            for label in transformed:
                xyz = np.asarray(transformed[label])
                xyz -= xyz.mean(axis=0, keepdims=True)
                rms[label] = 1000 * np.sqrt(np.mean(np.sum(xyz**2, axis=1)))
            outcomes[(flip_subtalar, flip_mtp)] = rms

    return outcomes, qangles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--trials', nargs='+', required=True,
                        help='E.g. S011:Trial01 S011:Trial02 S013:Trial02')
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model))

    for trial in args.trials:
        try:
            subject, trial_name = trial.split(':', 1)
        except ValueError as e:
            raise ValueError(f'Invalid trial spec {trial!r}; expected S011:Trial01') from e
        base = args.data_root / subject
        mot = find_step(base / 'JointAngle' / 'LevelWalking' / trial_name, '.mot')
        trc_path = find_step(base / 'MotionCapture' / 'LevelWalking' / 'TRC' / trial_name, '.trc')
        joints = read_mot(mot)
        times, names, trc = read_trc(trc_path)
        if not all(marker in names for _, marker, _ in TESTS):
            raise ValueError(f'Missing required foot markers in {trc_path}')
        outcome, qangles = run_trial(model, joints, times, names, trc)

        print(f'\n=== {trial}: {len(joints)} frames ===')
        for angle in ('subtalar_angle_l', 'mtp_angle_l'):
            ar = np.rad2deg(qangles[angle])
            print(f'{angle}: [{ar.min():.2f}, {ar.max():.2f}] degrees')

        colwidth = 20
        print(f'{"Marker / body frame":<26}' + ''.join(
            f'{(("S-" if s else "S+") + "/" + ("M-" if m else "M+")):>{colwidth}}'
            for s, m in outcome))
        for label, _, _ in TESTS:
            print(f'{label:<26}' + ''.join(
                f'{values[label]:{colwidth}.2f}' for values in outcome.values()))
        print('Units: RMS marker variation within the indicated moving frame (mm).')
        print('S- = flipped left subtalar sign, M- = flipped left MTP sign.')


if __name__ == '__main__':
    main()
