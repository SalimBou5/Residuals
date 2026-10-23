"""Investigate Gait120 steps rejected by the batch exporter's root-quaternion check.

No MuJoCo required. No files are changed.

Use either:
    python diagnose_gait120_root_quaternions.py --joint-mot path/to/step01.mot
or:
    python diagnose_gait120_root_quaternions.py --subject-root path/to/S060

Keep alongside gait120_kinematic_compatibility.py.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from inspection.inspection_utils.gait120_kinematic_compatibility import T, os_pelvis_rotation, read_mot

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)

def inspect(mot: Path):
    df = read_mot(mot)
    time = df['time'].to_numpy(dtype=float)
    original = os_pelvis_rotation(df).as_matrix()
    world = T @ original @ T.T
    q = as_quat_wxyz(Rotation.from_matrix(world))

    raw_dot = np.sum(q[1:] * q[:-1], axis=1)
    # q and -q encode the same physical rotation. This reports a sign-invariant
    # angular step, distinct from the raw sign of the quaternion dot product.
    rotation_per_step_deg = np.rad2deg(2 * np.arccos(np.clip(np.abs(raw_dot), 0, 1)))
    raw_sign_transitions = np.flatnonzero(raw_dot < 0)

    continuous_q = q.copy()
    for i in range(1, len(continuous_q)):
        if np.dot(continuous_q[i - 1], continuous_q[i]) < 0:
            continuous_q[i] *= -1
    new_dot = np.sum(continuous_q[1:] * continuous_q[:-1], axis=1)

    print(f'\n{mot}  |  frames={len(df)}')
    print(f'   Raw negative neighbor dots: {len(raw_sign_transitions)}')
    print(f'   Minimum raw dot: {raw_dot.min():.8f}; minimum |dot|: {np.abs(raw_dot).min():.8f}')
    print(f'   Largest physical root orientation step: {rotation_per_step_deg.max():.3f} deg/frame')
    print(f'   Steps exceeding 15 deg/frame: {(rotation_per_step_deg > 15).sum()}')
    print(f'   After sign-unwrapping: min neighbor dot={new_dot.min():.8f}')
    for k in raw_sign_transitions[:5]:
        print(f'   Sign change at {time[k]:.4f} -> {time[k+1]:.4f} s; '
              f'raw dot={raw_dot[k]:.6f}; physical change={rotation_per_step_deg[k]:.3f} deg')
    if rotation_per_step_deg.max() > 15:
        print('   WARNING: At least one large PHYSICAL orientation jump; do not auto-accept this stride.')
    elif raw_sign_transitions.size:
        print('   Likely only quaternion representation discontinuity; sign-unwrapping resolves it.')
    else:
        print('   No quaternion representation discontinuity found.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--joint-mot', type=Path)
    group.add_argument('--subject-root', type=Path,
                       help='Subject directory containing JointAngle/LevelWalking/Trial*/step*.mot')
    args = ap.parse_args()
    if args.joint_mot:
        files = [args.joint_mot]
    else:
        base = args.subject_root / 'JointAngle' / 'LevelWalking'
        files = sorted(p for p in base.glob('Trial*/*') if p.suffix.lower() == '.mot' and p.stem.lower().startswith('step'))
    if not files:
        ap.error('No step MOT files found')
    for f in files:
        inspect(f)

if __name__ == '__main__':
    main()
