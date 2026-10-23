import numpy as np
import mujoco

def update_knee_dependencies(model, data):
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue

        jdep = model.eq_obj1id[i]
        jsrc = model.eq_obj2id[i]

        dep = model.jnt_qposadr[jdep]
        src = model.jnt_qposadr[jsrc]

        x = data.qpos[src] - model.qpos0[src]
        c = model.eq_data[i, :5]

        data.qpos[dep] = (
            model.qpos0[dep]
            + np.polynomial.polynomial.polyval(x, c)
        )

model = mujoco.MjModel.from_xml_path(r"C:\Users\BoussofaraS\OneDrive - University of Twente\Documents\myosuite\myosuite\simhive\myo_sim\leg\myolegs.xml")
data = mujoco.MjData(model)

# Set knee flexion angles (in radians)
data.qpos[model.joint("knee_angle_r").qposadr[0]] = np.deg2rad(30)
data.qpos[model.joint("knee_angle_l").qposadr[0]] = np.deg2rad(45)

# Calculate all 14 dependent knee coordinates
update_knee_dependencies(model, data)

# Update forward kinematics
mujoco.mj_forward(model, data)
print(data.qpos)