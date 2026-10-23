"""Check Gait120 -> MyoLeg leg sign conventions using rigid marker consistency.

Place this file next to gait120_kinematic_compatibility.py and run inside
an environment where your complete MyoLeg model loads with MuJoCo.

Example:
python test_gait120_segment_rigidity.py \
  --model /path/to/myolegs.xml \
  --joint-mot 'step01(1).mot' \
  --trc Step01.trc

This does not modify the source files or the MyoLeg model.
"""
import argparse
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from inspection.inspection_utils.gait120_kinematic_compatibility import (
    LEG_JOINTS, T, mapped_angle_deg, os_pelvis_rotation,
    read_mot, read_trc, update_knee_dependencies,
)

# TRC marker names grouped by the rigid body on which they are attached.
BODY_MARKERS = {
    'pelvis': ('LASI', 'RASI', 'LPSI', 'RPSI'),
    'femur_r': ('RTHI', 'RKNE'),
    'femur_l': ('LTHI', 'LKNE'),
    'tibia_r': ('RTIB', 'RANK'),
    'tibia_l': ('LTIB', 'LANK'),
    'calcn_r': ('RHEE', 'RTOE'),
    'calcn_l': ('LHEE', 'LTOE'),
}

CASES = {
    'original': (),
    'flip left adduction': ('hip_adduction_l',),
    'flip left add + rotation': ('hip_adduction_l', 'hip_rotation_l'),
    'flip left add + rot + subtalar': (
        'hip_adduction_l', 'hip_rotation_l', 'subtalar_angle_l'
    ),
}


def measure_rigidity(model, joints, trc, names, flips):
    data = mujoco.MjData(model)
    rootadr = model.joint('root').qposadr[0]
    idx = {name: model.joint(name).qposadr[0] for name in LEG_JOINTS}
    angles = {
        name: np.deg2rad(mapped_angle_deg(joints, name))
        for name in LEG_JOINTS
    }
    rots = os_pelvis_rotation(joints, 'ZXY').as_matrix()
    trans = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()

    # Convert measured marker positions to MuJoCo WORLD coordinates.
    measured = {
        name: (T @ trc[:, names.index(name), :].T).T
        for markers in BODY_MARKERS.values() for name in markers
    }
    local = {body: {marker: [] for marker in markers}
             for body, markers in BODY_MARKERS.items()}

    for frame in range(len(joints)):
        data.qpos[:] = model.qpos0
        data.qpos[rootadr:rootadr+3] = T @ trans[frame]
        # pelvis body is fixed +90 degrees about X relative to root
        R_root = T @ rots[frame] @ T.T
        data.qpos[rootadr+3:rootadr+7] = as_quat_wxyz(Rotation.from_matrix(R_root))        

        for name in LEG_JOINTS:
            sign = -1 if name in flips else 1
            data.qpos[idx[name]] = sign * angles[name][frame]
        update_knee_dependencies(model, data, mujoco)
        mujoco.mj_forward(model, data)

        for body_name, marker_names in BODY_MARKERS.items():
            body = data.body(body_name)
            origin = body.xpos
            R_world_body = body.xmat.reshape(3, 3)
            for marker_name in marker_names:
                point_local = R_world_body.T @ (
                    measured[marker_name][frame] - origin
                )
                local[body_name][marker_name].append(point_local)

    results = {}
    for body_name, marker_dict in local.items():
        per_marker_error = []
        for samples in marker_dict.values():
            xyz = np.asarray(samples)
            mean_xyz = xyz.mean(axis=0)  # unknown constant marker placement
            per_marker_error.append(
                np.mean(np.sum((xyz - mean_xyz)**2, axis=1))
            )
        results[body_name] = 1000 * np.sqrt(np.mean(per_marker_error))
    return results


def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--joint-mot', required=True)
    parser.add_argument('--trc', required=True)
    args = parser.parse_args()

    joints = read_mot(args.joint_mot)
    times, names, trc = read_trc(args.trc)
    assert len(joints) == len(trc)
    assert np.allclose(joints.time.to_numpy(), times, atol=1e-8)
    assert all(name in names for group in BODY_MARKERS.values() for name in group)

    model = mujoco.MjModel.from_xml_path(args.model)
    results = {
        title: measure_rigidity(model, joints, trc, names, flips)
        for title, flips in CASES.items()
    }
    print('3D marker variation RMS (mm; lower is better)')
    print(f"{'Body':<12}" + ''.join(f'{title:>33}' for title in CASES))
    for body in BODY_MARKERS:
        print(f'{body:<12}' + ''.join(
            f'{results[title][body]:33.2f}' for title in CASES
        ))


if __name__ == '__main__':
    main()
