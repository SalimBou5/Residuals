"""Diagnose velocity peaks in a Gait120 -> MyoLeg candidate trajectory.

Run in the environment where the real MyoLeg model loads:
python inspect_gait120_reference_velocities.py --model "C:\\path\\to\\myolegs.xml" --candidate "S012_T01_check_candidate.npz"

Reads only; never edits the NPZ, original MOT/TRC files, or model.
The 7-sample Savitzky-Golay derivative is a DIAGNOSTIC comparison,
not a replacement reference or a validated filtering method.

python inspect_gait120_reference_velocities.py --model "C:\Users\BoussofaraS\OneDrive - University of Twente\Documents\myosuite\myosuite\simhive\myo_sim\leg\myolegs.xml" --candidate "S012_T01_check_candidate.npz"
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from inspection.inspection_utils.gait120_kinematic_compatibility import LEG_JOINTS

WATCH = ('ankle_angle_l', 'subtalar_angle_r', 'ankle_angle_r')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    with np.load(args.candidate) as npz:
        t = np.asarray(npz['time'], dtype=float)
        qpos = np.asarray(npz['qpos'], dtype=float)
        qvel = np.asarray(npz['qvel'], dtype=float)

    assert qpos.shape == (len(t), model.nq), (qpos.shape, model.nq)
    assert qvel.shape == (len(t), model.nv), (qvel.shape, model.nv)
    assert len(t) >= 7 and np.isfinite(qpos).all() and np.isfinite(qvel).all()
    assert np.all(np.diff(t) > 0)
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, atol=1e-5, rtol=0.01):
        raise ValueError('Nonuniform timestamps; this diagnostic expects constant frame rate.')

    root_idx = int(model.joint('root').qposadr[0])
    quats = qpos[:, root_idx+3:root_idx+7]
    consecutive_dots = np.sum(quats[:-1] * quats[1:], axis=1)
    print(f'Frames: {len(t)} | duration: {t[-1]-t[0]:.3f} s | sampling: {1/dt:.1f} Hz')
    print(f'Root quaternion: min consecutive dot = {consecutive_dots.min():.6f}, '
          f'sign discontinuities = {(consecutive_dots < 0).sum()}')
    print('Peak speeds (deg/s), with 7-frame Savitzky-Golay derivative for comparison:')
    print(f'{"Joint":20s} {"max raw":>9s} {"95% raw":>9s} {"max SG7":>9s} {"peak t":>8s} {"near edge":>10s}')

    for name in LEG_JOINTS:
        joint = model.joint(name)
        pos_i = int(joint.qposadr[0])
        vel_i = int(joint.dofadr[0])
        angle = np.rad2deg(qpos[:, pos_i])
        raw = np.rad2deg(qvel[:, vel_i])
        sg7 = savgol_filter(angle, window_length=7, polyorder=3,
                            deriv=1, delta=dt, mode='interp')
        i = int(np.argmax(np.abs(raw)))
        p95 = np.percentile(np.abs(raw), 95)
        edge = i < 3 or i >= len(t)-3
        print(f'{name:20s} {np.max(abs(raw)):9.1f} {p95:9.1f} '
              f'{np.max(abs(sg7)):9.1f} {t[i]:8.3f} {str(edge):>10s}')

        # Check that saved MuJoCo velocity matches numerical angle differentiation.
        gradient = np.gradient(angle, t)
        max_difference = np.max(np.abs(gradient - raw))
        if max_difference > 2:
            print(f'  CHECK: qvel differs from np.gradient by {max_difference:.2f} deg/s')

        if name in WATCH:
            print(f'  {name}: five frames around largest raw speed')
            print('  time [s]    angle [deg]  raw [deg/s]  SG7 [deg/s]')
            for k in range(max(0, i-2), min(len(t), i+3)):
                print(f'  {t[k]:8.3f} {angle[k]:13.3f} {raw[k]:12.1f} {sg7[k]:12.1f}')

            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            elapsed = t - t[0]
            axes[0].plot(elapsed, angle, label='Recorded mapped angle')
            axes[0].set_ylabel('Angle (deg)')
            axes[0].grid(alpha=0.25)
            axes[1].plot(elapsed, raw, label='Original finite-difference velocity')
            axes[1].plot(elapsed, sg7, '--', label='SG7 diagnostic derivative')
            axes[1].axvline(t[i]-t[0], linestyle=':', color='gray')
            axes[1].set(xlabel='Time since stride start (s)', ylabel='Angular velocity (deg/s)')
            axes[1].grid(alpha=0.25)
            axes[1].legend(fontsize=8)
            fig.suptitle(name)
            fig.tight_layout()
            out = args.output_dir / f'{args.candidate.stem}_{name}_velocity.png'
            fig.savefig(out, dpi=160)
            plt.close(fig)
            print('  Plot:', out)

    print('\nNo trajectory values were changed. Inspect plots before deciding whether filtering is necessary.')


if __name__ == '__main__':
    main()
