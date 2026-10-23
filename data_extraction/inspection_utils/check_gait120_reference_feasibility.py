"""Inspect a Gait120 step replayed as a *candidate* MyoLeg reference.

No source files or MuJoCo model files are modified.

Usage (Windows, one line):
python check_gait120_reference_feasibility.py --model "C:\\path\\to\\myolegs.xml" --joint-mot "C:\\path\\to\\step01.mot" --trc "C:\\path\\to\\step01.trc" --output-prefix reference_check

Needs: numpy, pandas, scipy, mujoco, matplotlib and
       gait120_kinematic_compatibility.py in this directory.

This deliberately uses ONE GLOBAL mapping (no subject-specific subtalar flip):
  - reverse Gait120 knee angles (flexion is negative in this MOT)
  - reverse LEFT hip adduction and LEFT hip rotation
  - keep subtalar and MTP signs unmodified
  - derive 14 dependent knee coordinates from MuJoCo equalities
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from inspection.inspection_utils.gait120_kinematic_compatibility import (
    LEG_JOINTS, T, mapped_angle_deg, os_pelvis_rotation,
    read_mot, read_trc, update_knee_dependencies,
)

# Association of experimental markers with *rigid bodies*, not necessarily
# with MyoLeg's nominal marker sites. Marker locations are fitted in each
# body frame to allow different subject anatomy / marker placement.
MARKERS = {
    'LASI': 'pelvis', 'RASI': 'pelvis',
    'LPSI': 'pelvis', 'RPSI': 'pelvis',
    'LTHI': 'femur_l', 'RTHI': 'femur_r',
    'LTIB': 'tibia_l', 'RTIB': 'tibia_r',
    'LHEE': 'calcn_l', 'RHEE': 'calcn_r',
    'LTOE': 'toes_l', 'RTOE': 'toes_r',
}

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)

def candidate_qpos(model, joints):
    n = len(joints)
    q = np.repeat(model.qpos0[None, :], n, axis=0)
    root = int(model.joint('root').qposadr[0])
    p = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    R_os = os_pelvis_rotation(joints).as_matrix()
    q[:, root:root + 3] = (T @ p.T).T
    q[:, root + 3:root + 7] = as_quat_wxyz(Rotation.from_matrix(T @ R_os @ T.T))

    for name in LEG_JOINTS:
        value = np.deg2rad(mapped_angle_deg(joints, name))
        if name in ('hip_adduction_l', 'hip_rotation_l'):
            value = -value
        q[:, int(model.joint(name).qposadr[0])] = value

    data = mujoco.MjData(model)
    for k in range(n):
        data.qpos[:] = q[k]
        update_knee_dependencies(model, data, mujoco)
        q[k] = data.qpos
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
    return v


def evaluate(model, q, obs, marker_names):
    data = mujoco.MjData(model)
    names = [m for m in MARKERS if m in marker_names]
    if not names:
        raise ValueError('No matching anatomical markers in TRC.')
    body_p = {m: [] for m in names}
    body_R = {m: [] for m in names}
    ground_penetration = []
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')

    for k in range(len(q)):
        data.qpos[:] = q[k]
        mujoco.mj_forward(model, data)
        for m in names:
            body = data.body(MARKERS[m])
            body_p[m].append(body.xpos.copy())
            body_R[m].append(body.xmat.reshape(3, 3).copy())
        distances = [c.dist for c in data.contact[:data.ncon]
                     if ground_id >= 0 and (c.geom1 == ground_id or c.geom2 == ground_id)]
        ground_penetration.append(max(0.0, -min(distances)) if distances else 0.0)

    residual_mm = {}
    rms_mm = {}
    for m in names:
        measured_mj = (T @ obs[:, marker_names.index(m), :].T).T
        p = np.asarray(body_p[m])
        R = np.asarray(body_R[m])
        # Fit ONE constant marker location in the moving body frame.
        local = np.einsum('tji,tj->ti', R, measured_mj - p)
        local_mean = np.mean(local, axis=0)
        reconstructed = p + np.einsum('tij,j->ti', R, local_mean)
        per_frame = 1000 * np.linalg.norm(reconstructed - measured_mj, axis=1)
        residual_mm[m] = per_frame
        rms_mm[m] = float(np.sqrt(np.mean(per_frame ** 2)))
    return residual_mm, rms_mm, np.asarray(ground_penetration)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--joint-mot', type=Path, required=True)
    ap.add_argument('--trc', type=Path, required=True)
    ap.add_argument('--output-prefix', default='reference_check')
    args = ap.parse_args()

    joints = read_mot(args.joint_mot)
    t, names, trc = read_trc(args.trc)
    if not np.allclose(joints.time.to_numpy(), t, atol=1e-6, rtol=0):
        raise ValueError('TRC / joint MOT timestamps are not synchronized.')
    if not np.isfinite(trc).all() or not np.isfinite(joints.select_dtypes('number').to_numpy()).all():
        raise ValueError('Missing / non-finite input values.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('Time must be strictly increasing.')
    if 'S011' in str(args.joint_mot).upper():
        print('WARNING: S011 has a known subtalar-sign inconsistency. Diagnostic only!')

    model = mujoco.MjModel.from_xml_path(str(args.model))
    q = candidate_qpos(model, joints)
    v = differentiate_qpos(model, t, q)
    errors, rms, ground_depth = evaluate(model, q, trc, names)

    print(f'Frames: {len(q)}, time: {t[0]:.3f} .. {t[-1]:.3f} s')
    print('Mapping: reverse both knee angles, left hip adduction and rotation; subtalar unchanged.')
    print('RMS marker error after ONE constant per-marker body offset (mm):')
    for marker, value in rms.items():
        print(f'  {marker:4} [{MARKERS[marker]:7}] {value:7.2f}')
    print('These errors assess body-relative motion; they do NOT establish subject-scaled IK accuracy.')
    print(f'Maximum simulated terrain penetration: {ground_depth.max()*1000:.1f} mm')
    print(f'Frames with terrain penetration >5 mm: {np.count_nonzero(ground_depth > 0.005)}/{len(q)}')
    print('Penetration is a diagnostic of raw transfer / subject scaling, not proof of a sign error.')

    print('Peak absolute speeds (deg/s) for independent joint coordinates:')
    for name in LEG_JOINTS:
        dofadr = int(model.joint(name).dofadr[0])
        print(f'  {name:19s} {np.rad2deg(np.abs(v[:, dofadr]).max()):7.1f}')

    out = Path(args.output_prefix)
    np.savez(str(out) + '_candidate.npz', time=t, qpos=q, qvel=v)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in ('LTHI', 'LTIB', 'LHEE', 'LTOE', 'RTHI', 'RTIB', 'RHEE', 'RTOE'):
        if name in errors:
            ax.plot(t - t[0], errors[name], label=name, linewidth=1.4)
    ax.set(xlabel='Time from step start (s)', ylabel='Marker motion error (mm)',
           title='Experimental markers vs MyoLeg FK (subject-specific marker offset fitted)')
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out) + '_marker_errors.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t - t[0], 1000 * ground_depth, label='Max terrain penetration')
    ax.axhline(5, ls='--', color='gray', label='5 mm diagnostic threshold')
    ax.set(xlabel='Time from step start (s)', ylabel='Penetration (mm)',
           title='Static replay terrain penetration (not a dynamic rollout)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out) + '_terrain_penetration.png', dpi=180)
    plt.close(fig)
    print('Saved:', str(out) + '_candidate.npz',
          str(out) + '_marker_errors.png', str(out) + '_terrain_penetration.png')
    print('IMPORTANT: candidate NPZ is a diagnostic replay, NOT validated training reference.')


if __name__ == '__main__':
    main()
