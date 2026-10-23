"""Compare transferred MyoLeg pelvis tilt with Gait120's measured upper-back motion.

A *diagnostic*, not a torso retargeter: C7-T10 is a marker-based proxy for
upper-trunk direction, not an anatomical thorax coordinate frame. This script
only reads the extracted NPZ; it never modifies references or the model.

Usage:
    python diagnose_gait120_trunk_sway.py S012_T01_reference.npz
Dependencies: numpy, scipy, matplotlib.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

T = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference', type=Path, help='NPZ produced by extract_gait120_myoleg_reference.py')
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    with np.load(args.reference, allow_pickle=False) as d:
        t = d['time_s'].copy()
        q = d['qpos'].copy()
        names = list(d['marker_names'])
        markers_os = d['marker_xyz_os_m'].copy()

    assert q.shape[0] == t.size == markers_os.shape[0]
    assert q.shape[1] == 35, 'This diagnostic assumes the original 35-qpos MyoLeg.'
    if not all(n in names for n in ('C7', 'T10')):
        raise ValueError('TRC is missing the C7 or T10 marker')

    # In the original MyoLeg, the pelvis is fixed to root by a +90deg X
    # quaternion. Therefore its anatomical "up" maps to the ROOT Z axis.
    root_R = Rotation.from_quat(q[:, 3:7], scalar_first=True).as_matrix()
    pelvis_up = root_R[:, :, 2]
    pelvis_forward = root_R[:, :, 0]
    forward = np.mean(pelvis_forward, axis=0)
    forward[2] = 0.
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0], 0.])

    # Gait120 TRC is in OpenSim coordinates. Change world basis using T.
    xyz = np.einsum('ij,tkj->tki', T, markers_os)
    upper_back_up = xyz[:, names.index('C7')] - xyz[:, names.index('T10')]
    upper_back_up /= np.linalg.norm(upper_back_up, axis=1, keepdims=True)

    def lateral_tilt(unit_direction):
        return np.rad2deg(np.arcsin(np.clip(unit_direction @ left, -1., 1.)))

    pelvis_lateral = lateral_tilt(pelvis_up)
    measured_back_lateral = lateral_tilt(upper_back_up)
    pelvis_yaw = np.rad2deg(np.unwrap(np.arctan2(
        pelvis_forward[:, 1], pelvis_forward[:, 0])))
    pelvis_yaw -= pelvis_yaw[0]

    # According to the uploaded myolegs_chain.xml: root free joint occupies
    # qpos[0:7]; independent hip_rotation_r and hip_rotation_l are 9 and 23.
    hip_rotation_r = np.rad2deg(q[:, 9])
    hip_rotation_l = np.rad2deg(q[:, 23])
    ts = t - t[0]
    for seconds in (0.06, 0.60):
        i = int(np.argmin(abs(ts - seconds)))
        print(f't={ts[i]:.2f}s: pelvis lateral tilt={pelvis_lateral[i]:+.2f} deg; '
              f'upper-back (C7-T10) proxy={measured_back_lateral[i]:+.2f} deg')
    i1 = int(np.argmin(abs(ts - 0.06)))
    i2 = int(np.argmin(abs(ts - 0.60)))
    print(f'Between {ts[i1]:.2f}s and {ts[i2]:.2f}s: '
          f'pelvis tilt change={pelvis_lateral[i2]-pelvis_lateral[i1]:+.2f} deg, '
          f'upper-back proxy change={measured_back_lateral[i2]-measured_back_lateral[i1]:+.2f} deg')
    print(f'Pelvis yaw excursion: {np.ptp(pelvis_yaw):.2f} deg')
    print(f'Left hip rotation: {hip_rotation_l.min():+.2f} to {hip_rotation_l.max():+.2f} deg')
    print(f'Right hip rotation: {hip_rotation_r.min():+.2f} to {hip_rotation_r.max():+.2f} deg')

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    axes[0].plot(ts, pelvis_lateral, label='Transferred pelvis superior axis')
    axes[0].plot(ts, measured_back_lateral, label='Recorded C7–T10 upper-back proxy')
    axes[0].axhline(0, linewidth=.8, color='gray')
    axes[0].set_ylabel('Lateral lean (deg)\npositive = left')
    axes[0].set_title('Gait120 S012: why the rigid MyoLeg trunk can look tilted')
    axes[0].legend()
    axes[1].plot(ts, pelvis_yaw, label='Pelvis heading change')
    axes[1].set_ylabel('Change in yaw (deg)')
    axes[1].legend()
    axes[2].plot(ts, hip_rotation_l, label='Left hip axial rotation')
    axes[2].plot(ts, hip_rotation_r, label='Right hip axial rotation')
    axes[2].set_ylabel('Hip rotation (deg)')
    axes[2].set_xlabel('Time since step start (s)')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    out = args.output if args.output else args.reference.with_name(args.reference.stem + '_trunk_sway.png')
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'Saved plot: {out}')
    print('C7–T10 is only a segment-direction proxy; compare changes, not absolute trunk angles.')


if __name__ == '__main__':
    main()
