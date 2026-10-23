"""Gait120 -> MyoLeg: reproduce kinematic compatibility checks without editing source data.

Local preflight (NumPy/SciPy only):
 python gait120_kinematic_compatibility.py --joint-mot 'step01(1).mot' --trc Step01.trc \
    --chain myolegs_chain.xml --assets myolegs_assets.xml

Actual MuJoCo forward-kinematics test (run from your MyoSuite environment):
 python gait120_kinematic_compatibility.py --joint-mot 'step01(1).mot' --trc Step01.trc \
    --chain myolegs_chain.xml --assets myolegs_assets.xml --model /path/to/myolegs.xml

Requires numpy, pandas, scipy; MuJoCo is only required for the second test.
All angles in MOT are degrees; MJCF joint angles are radians.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

T = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
PELVIS_MARKERS = ('LASI', 'RASI', 'LPSI', 'RPSI')
LEG_JOINTS = (
    'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r',
    'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l',
    'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
)


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


def os_pelvis_rotation(joints, convention='ZXY'):
    return Rotation.from_euler(convention, joints[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].to_numpy(), degrees=True)


def pelvic_marker_constancy(joints, trc, names, convention='ZXY'):
    """RMS 3D marker variation after rotating measured markers into pelvis local frame.
    Allows each marker to have its own constant (subject-specific) anatomical offset.
    """
    measured = trc[:, [names.index(s) for s in PELVIS_MARKERS], :]
    root_xyz = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    R = os_pelvis_rotation(joints, convention).as_matrix()
    local = np.einsum('tij,tkj->tki', R.transpose(0, 2, 1), measured-root_xyz[:, None, :])
    rms_3d = 1000*np.sqrt(np.mean(np.sum((local-local.mean(axis=0))**2, axis=-1)))
    return rms_3d, local


def model_pelvis_markers(chain_xml):
    pelvis = ET.parse(chain_xml).getroot().find('.//body[@name="pelvis"]')
    return np.stack([np.fromstring(pelvis.find(f'.//site[@name="{s}"]').attrib['pos'], sep=' ')
                     for s in PELVIS_MARKERS])


def xml_joint_and_eq(chain_xml, assets_xml):
    chain = ET.parse(chain_xml).getroot()
    assets = ET.parse(assets_xml).getroot()
    joints = {x.attrib['name']: x for x in chain.findall('.//joint')}
    eqs = assets.findall('.//equality/joint')
    return joints, eqs


def mapped_angle_deg(joints, name):
    values = joints[name].to_numpy()
    return -values if name in ('knee_angle_l', 'knee_angle_r') else values


def check_joint_limits(joints, chain_xml, assets_xml):
    definitions, eqs = xml_joint_and_eq(chain_xml, assets_xml)
    print('\n=== Independent leg joint limit check (degrees) ===')
    count = 0
    for name in LEG_JOINTS:
        angles = mapped_angle_deg(joints, name)
        lo, hi = np.rad2deg(np.fromstring(definitions[name].attrib['range'], sep=' '))
        violations = int(np.count_nonzero((angles < lo-1e-6) | (angles > hi+1e-6)))
        count += violations
        print(f'{name:19} [{angles.min():7.2f}, {angles.max():7.2f}] vs [{lo:6.1f}, {hi:6.1f}], outside={violations}')
    print(f'Independent joint range violations: {count}')
    print('\n=== Dependent knee equality polynomials ===')
    for eq in eqs:
        dependent, source = eq.attrib['joint1'], eq.attrib['joint2']
        coef = np.fromstring(eq.attrib['polycoef'], sep=' ')
        q = np.deg2rad(mapped_angle_deg(joints, source))
        qdep = np.polynomial.polynomial.polyval(q, coef)  # XML default joint reference is zero
        lo, hi = np.fromstring(definitions[dependent].attrib['range'], sep=' ')
        over = np.maximum(np.maximum(lo-qdep, qdep-hi), 0.)
        if np.max(over) > 1e-7:
            unit = 'm' if definitions[dependent].attrib.get('type') == 'slide' else 'rad'
            print(f'{dependent:37} outside={int(np.count_nonzero(over > 1e-7)):3d}, '
                  f'max overrun={over.max():.6g} {unit}')
    print('These minor bounds overshoots are diagnostics; do not clamp a dependent joint.')


def update_knee_dependencies(model, data, mj):
    """Fill both knees' 14 dependent qpos values from the compiled equality constraints."""
    for i in range(model.neq):
        if model.eq_type[i] != mj.mjtEq.mjEQ_JOINT:
            continue
        dependent = int(model.eq_obj1id[i])
        source = int(model.eq_obj2id[i])
        if source < 0:
            continue
        dep_index = int(model.jnt_qposadr[dependent])
        src_index = int(model.jnt_qposadr[source])
        delta = data.qpos[src_index] - model.qpos0[src_index]
        data.qpos[dep_index] = model.qpos0[dep_index] + np.polynomial.polynomial.polyval(delta, model.eq_data[i, :5])

def as_quat_wxyz(rot):
    q_xyzw = rot.as_quat()
    if q_xyzw.ndim == 1:
        return np.roll(q_xyzw, 1)  # [x, y, z, w] -> [w, x, y, z]
    return np.roll(q_xyzw, 1, axis=-1)

def replay_fk(joints, trc, names, model_path):
    import mujoco as mj
    model = mj.MjModel.from_xml_path(str(model_path))
    data = mj.MjData(model)
    assert model.joint('root').type == mj.mjtJoint.mjJNT_FREE
    rootadr = model.joint('root').qposadr[0]
    # The pelvis body in myolegs_chain.xml has a fixed +90 deg X rotation
    # relative to 'root'. With the OS->MJ world basis change T, this gives:
    # R_root_MJ = T @ R_pelvis_OS @ T.T.
    rotations = os_pelvis_rotation(joints).as_matrix()
    positions = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    records = {key: [] for key in PELVIS_MARKERS + ('RTOE', 'LTOE', 'RCAL', 'LCAL')}
    for i in range(len(joints)):
        data.qpos[:] = model.qpos0
        data.qpos[rootadr:rootadr+3] = T @ positions[i]
        root_rot = T @ rotations[i] @ T.T
        data.qpos[rootadr+3:rootadr+7] = as_quat_wxyz(Rotation.from_matrix(root_rot))        
        for name in LEG_JOINTS:
            jointid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            assert jointid >= 0, f'Missing joint in compiled model: {name}'
            data.qpos[model.jnt_qposadr[jointid]] = np.deg2rad(mapped_angle_deg(joints, name)[i])
        update_knee_dependencies(model, data, mj)
        mj.mj_forward(model, data)
        for key in records:
            records[key].append(data.site(key).xpos.copy())
    print('\n=== Actual compiled MyoLeg FK vs measured TRC (mm) ===')
    for site, marker in [*[(x,x) for x in PELVIS_MARKERS], ('RTOE','RTOE'), ('LTOE','LTOE'),
                         ('RCAL','RHEE'), ('LCAL','LHEE')]:
        if marker not in names:
            print(f'{marker}: missing in TRC, skipped')
            continue
        expected = np.asarray(records[site])
        observed = (T @ trc[:, names.index(marker), :].T).T
        error = expected - observed
        raw = 1000*np.sqrt(np.mean(np.sum(error**2, axis=1)))
        temporal = 1000*np.sqrt(np.mean(np.sum((error-error.mean(axis=0))**2, axis=1)))
        print(f'{site:6} ↔ {marker:5}  raw RMS = {raw:7.2f}  motion-only RMS = {temporal:7.2f}')
    print('Raw RMS includes subject anatomy/scaling and marker-placement mismatch.')
    print('Motion-only RMS removes each marker’s constant laboratory offset; it is NOT a full IK accuracy measure.')
    print('RCAL/LHEE/RHEE are approximate matches only; their physical marker placements may differ.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint-mot', required=True)
    ap.add_argument('--trc', required=True)
    ap.add_argument('--chain', default='myolegs_chain.xml')
    ap.add_argument('--assets', default='myolegs_assets.xml')
    ap.add_argument('--model', default=None, help='Optional: actual loadable MyoSuite myolegs.xml path')
    args = ap.parse_args()
    joints = read_mot(args.joint_mot)
    time, names, trc = read_trc(args.trc)
    assert np.allclose(joints['time'].to_numpy(), time, atol=1e-8), 'Joint MOT and TRC timestamps differ!'
    assert np.isfinite(trc).all() and np.isfinite(joints.select_dtypes(include='number')).all().all()
    print('=== Original data ===')
    print('Frames:', len(time), 'time range:', (time[0], time[-1]))
    print('Time synchronization: PASS')
    print('Missing numerical data: NONE')

    print('\n=== Pelvis rotation: marker-rigidity test ===')
    for seq in ('ZXY', 'zxy', 'ZYX', 'zyx'):
        rms, _ = pelvic_marker_constancy(joints, trc, names, seq)
        print(f'{seq:4} (3D marker variation RMS): {rms:.3f} mm')
    _, local = pelvic_marker_constancy(joints, trc, names, 'ZXY')
    os_markers = model_pelvis_markers(args.chain)
    R = os_pelvis_rotation(joints).as_matrix()
    pos = joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    observed = trc[:, [names.index(s) for s in PELVIS_MARKERS], :]
    pred = pos[:, None, :] + np.einsum('tij,kj->tki', R, os_markers)
    dif = pred - observed
    print('Nominal (unscaled) MyoLeg pelvis-marker 3D RMS:',
          f'{1000*np.sqrt(np.mean(np.sum(dif**2,axis=-1))):.3f} mm')
    print('Per-marker motion-only RMS:',
          np.round(1000*np.sqrt(np.mean(np.sum((dif-dif.mean(0))**2, axis=-1), axis=0)), 3), 'mm')
    q = as_quat_wxyz(Rotation.from_matrix(T @ R[0] @ T.T))
    print('First frame root position (MJ):', np.round(T @ pos[0], 6))
    print('First frame root quaternion (wxyz):', np.round(q, 6))
    check_joint_limits(joints, args.chain, args.assets)
    if args.model:
        replay_fk(joints, trc, names, args.model)
    else:
        print('\nCompiled MuJoCo FK: NOT RUN. Pass --model /path/to/myolegs.xml in your MyoSuite environment.')

if __name__ == '__main__':
    main()
