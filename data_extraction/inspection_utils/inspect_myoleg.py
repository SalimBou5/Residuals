import mujoco

model = mujoco.MjModel.from_xml_path(r"C:\Users\BoussofaraS\OneDrive - University of Twente\Documents\myosuite\myosuite\simhive\myo_sim\leg\myolegs.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("Root qpos:", model.qpos0[:7])
print("Root orientation:", data.body("root").xquat)
print("Pelvis orientation:", data.body("pelvis").xquat)

for i in range(model.neq):
    if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
        j1 = int(model.eq_obj1id[i])
        j2 = int(model.eq_obj2id[i])

        print(
            model.joint(j1).name,
            model.joint(j2).name if j2 >= 0 else None,
            model.eq_data[i, :5],
        )