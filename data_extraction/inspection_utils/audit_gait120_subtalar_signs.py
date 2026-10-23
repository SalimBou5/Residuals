"""Audit whether a single left-subtalar sign convention fits multiple Gait120 subjects.

Uses the unmodified original joint-angle MOT and marker TRC files. No optimization,
no subject-specific sign corrections, no changes to the original dataset.

Requires: numpy, pandas, scipy, mujoco and
          gait120_kinematic_compatibility.py in the same folder.

Example (Windows; run as one line):
python audit_gait120_subtalar_signs.py --model "C:\\path\\to\\myolegs.xml" --data-root "C:\\path\\to\\Gait120_011_to_020"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
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

def marker_local_rms(local):
    a = np.asarray(local)
    return 1000.0 * np.sqrt(np.mean(np.sum((a - a.mean(axis=0)) ** 2, axis=1)))


def matching_file(folder: Path, filename: str):
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing TRC folder: {folder}")
    candidates = [f for f in folder.iterdir()
                  if f.is_file() and f.name.lower() == filename.lower().replace('.mot', '.trc')]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one TRC matching {filename} in {folder}; got {candidates}")
    return candidates[0]


def inspect_trial(model, mot_path, trc_path):
    j = read_mot(mot_path)
    times, names, xyz_os = read_trc(trc_path)
    if not np.allclose(j['time'].to_numpy(), times, rtol=0, atol=1e-6):
        raise ValueError(f"Times differ: {mot_path} vs {trc_path}")
    if not np.isfinite(j.select_dtypes('number').to_numpy()).all() or not np.isfinite(xyz_os).all():
        raise ValueError(f"Nonfinite data: {mot_path}, {trc_path}")
    for required in ('LTOE', 'LHEE'):
        if required not in names:
            raise ValueError(f"Missing {required} in {trc_path}")

    # Measured 3D markers in MuJoCo world axes (metres).
    measured = {name: (T @ xyz_os[:, names.index(name)].T).T
                for name in ('LTOE', 'LHEE')}
    sep_mm = 1000.0 * np.linalg.norm(measured['LTOE'] - measured['LHEE'], axis=1)

    # Prepare root and independent joints. The two validated left-hip sign
    # corrections are applied identically under both subtalar hypotheses.
    q = np.repeat(model.qpos0[None, :], len(j), axis=0)
    root = int(model.joint('root').qposadr[0])
    pos_os = j[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    R_os = os_pelvis_rotation(j).as_matrix()
    q[:, root:root + 3] = (T @ pos_os.T).T
    q[:, root + 3:root + 7] = as_quat_wxyz(Rotation.from_matrix(T @ R_os @ T.T))
    for name in LEG_JOINTS:
        value = np.deg2rad(mapped_angle_deg(j, name))
        if name in ('hip_adduction_l', 'hip_rotation_l'):
            value = -value
        q[:, int(model.joint(name).qposadr[0])] = value

    sub_adr = int(model.joint('subtalar_angle_l').qposadr[0])
    sub_source_rad = np.deg2rad(j['subtalar_angle_l'].to_numpy())
    data = mujoco.MjData(model)
    results = {}

    for sign in (1, -1):
        local_toe_calc, local_toe_toes, local_heel_calc = [], [], []
        for k in range(len(j)):
            data.qpos[:] = q[k]
            data.qpos[sub_adr] = sign * sub_source_rad[k]
            update_knee_dependencies(model, data, mujoco)
            mujoco.mj_forward(model, data)

            def in_body(marker, body):
                b = data.body(body)
                return b.xmat.reshape(3, 3).T @ (measured[marker][k] - b.xpos)

            local_toe_calc.append(in_body('LTOE', 'calcn_l'))
            local_toe_toes.append(in_body('LTOE', 'toes_l'))
            local_heel_calc.append(in_body('LHEE', 'calcn_l'))

        results[sign] = (
            marker_local_rms(local_toe_calc),
            marker_local_rms(local_toe_toes),
            marker_local_rms(local_heel_calc),
        )

    return {
        'frames': len(j),
        'subtalar_min_deg': float(np.rad2deg(sub_source_rad.min())),
        'subtalar_max_deg': float(np.rad2deg(sub_source_rad.max())),
        'toe_heel_mean_mm': float(sep_mm.mean()),
        'toe_heel_sd_mm': float(sep_mm.std()),
        'toe_calc_plus_mm': results[1][0],
        'toe_calc_minus_mm': results[-1][0],
        'toe_toes_plus_mm': results[1][1],
        'toe_toes_minus_mm': results[-1][1],
        'heel_calc_plus_mm': results[1][2],
        'heel_calc_minus_mm': results[-1][2],
        'toe_sign_advantage_mm': results[1][1] - results[-1][1],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--subjects', nargs='*', help='Optional subjects, e.g. S011 S013 S019')
    p.add_argument('--max-steps-per-trial', type=int, default=1,
                   help='Default 1 (first step per trial); 0 includes every step')
    p.add_argument('--output', type=Path, default=Path('gait120_subtalar_sign_audit.csv'))
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    subjects = sorted(x for x in args.data_root.iterdir() if x.is_dir()
                      and x.name.startswith('S')
                      and (not args.subjects or x.name in args.subjects))
    if not subjects:
        raise ValueError('No subject directories found. Check --data-root.')

    rows = []
    for subject_dir in subjects:
        motion = subject_dir / 'JointAngle' / 'LevelWalking'
        if not motion.is_dir():
            print(f'Skipping {subject_dir.name}: no LevelWalking joint angles')
            continue
        for trial_dir in sorted(motion.glob('Trial*')):
            mot_files = sorted(f for f in trial_dir.iterdir()
                               if f.is_file() and f.suffix.lower() == '.mot'
                               and f.stem.lower().startswith('step'))
            if args.max_steps_per_trial > 0:
                mot_files = mot_files[:args.max_steps_per_trial]
            for mot in mot_files:
                trc_dir = subject_dir / 'MotionCapture' / 'LevelWalking' / 'TRC' / trial_dir.name
                trc = matching_file(trc_dir, mot.name)
                try:
                    result = inspect_trial(model, mot, trc)
                except (ValueError, FileNotFoundError) as err:
                    print(f'FAILED {subject_dir.name}/{trial_dir.name}/{mot.stem}: {err}')
                    continue
                rows.append(dict(subject=subject_dir.name, trial=trial_dir.name,
                                 step=mot.stem, **result))
                print(f'{subject_dir.name}/{trial_dir.name}/{mot.stem} | '
                      f'toe S+ {result["toe_toes_plus_mm"]:5.1f} mm '
                      f'S- {result["toe_toes_minus_mm"]:5.1f} mm')

    if not rows:
        raise RuntimeError('No trials were successfully processed')
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print('\n=== Median toe-marker variation by subject, in toes_l frame ===')
    med = df.groupby('subject')[['toe_toes_plus_mm', 'toe_toes_minus_mm',
                                 'toe_sign_advantage_mm', 'toe_heel_sd_mm']].median()
    print(med.round(2).to_string())
    print('\nPositive sign advantage = reversal improved the marker-rigidity score.')
    print('Note: This does not identify a constant subtalar offset or certify IK accuracy.')
    print(f'Saved full per-trial results to {args.output.resolve()}')


if __name__ == '__main__':
    main()
