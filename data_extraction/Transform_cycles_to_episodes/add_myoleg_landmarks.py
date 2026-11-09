#!/usr/bin/env python3
"""STEP 2: add pelvis-relative MyoLeg landmark kinematics to STEP 1 loops.

Does NOT change the gait cycle, seam width, phase, displacement or events.
Only equality-dependent joint positions/velocities are projected to match the
supplied MuJoCo model. Then FK + Jacobians provide landmark states.

Required: numpy, scipy, mujoco.

PowerShell example (single line):
python add_myoleg_landmarks.py --input loopable_references --model "C:\\path\\to\\myolegs.xml" --output gait_with_landmarks
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_SITES = (
    "hip_r", "hip_l", "knee_r", "knee_l",
    "ankle_r", "ankle_l", "toe_r", "toe_l",
)


def wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return xyzw[[3, 0, 1, 2]]


def ordered_joint_equalities(model, mj):
    """Read active scalar joint polynomials, driver before dependent."""
    rules = []
    for eq in range(model.neq):
        if int(model.eq_type[eq]) != int(mj.mjtEq.mjEQ_JOINT):
            continue
        if hasattr(model, "eq_active0") and not bool(model.eq_active0[eq]):
            continue
        dep, src = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        if dep < 0:
            continue
        allowed = {int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)}
        if int(model.jnt_type[dep]) not in allowed or (src >= 0 and int(model.jnt_type[src]) not in allowed):
            raise ValueError("Expected hinge/slide joint equality")
        rules.append((dep, src, np.asarray(model.eq_data[eq, :5], dtype=float).copy()))
    dependent = {d for d, _, _ in rules}
    done, pending = set(), rules.copy()
    ordered = []
    while pending:
        can_apply = [i for i, (d, s, c) in enumerate(pending) if s not in dependent or s in done]
        if not can_apply:
            raise ValueError("Cyclic joint-equality dependencies")
        for i in reversed(can_apply):
            rule = pending.pop(i)
            ordered.append(rule)
            done.add(rule[0])
    return ordered


def project_equalities(qpos, qvel, model, rules):
    """Change dependent coordinates only; preserve all independent qpos/qvel."""
    q, v = qpos.copy(), qvel.copy()
    before, change = 0.0, 0.0
    for dep, src, c in rules:
        qadr = int(model.jnt_qposadr[dep])
        dadr = int(model.jnt_dofadr[dep])
        if src >= 0:
            xadr = int(model.jnt_qposadr[src])
            xvel = v[:, int(model.jnt_dofadr[src])]
            dx = q[:, xadr] - model.qpos0[xadr]
        else:
            dx = np.zeros(len(q))
            xvel = np.zeros(len(q))
        target = (model.qpos0[qadr] + c[0] + c[1]*dx + c[2]*dx**2
                  + c[3]*dx**3 + c[4]*dx**4)
        error = float(np.max(np.abs(q[:, qadr] - target)))
        before = max(before, error)
        change = max(change, error)
        q[:, qadr] = target
        v[:, dadr] = (c[1] + 2*c[2]*dx + 3*c[3]*dx**2 + 4*c[4]*dx**3) * xvel
    return q, v, before, change


def landmark_states(q, v, model, mj, pelvis_id, site_ids):
    """FK and exact Jacobian velocities, all expressed in the pelvis body frame.

    First landmark is the pelvis body origin: zero position/velocity,
    identity orientation, zero angular velocity.
    """
    frames = len(q)
    count = 1 + len(site_ids)
    pos = np.zeros((frames, count, 3))
    quat = np.zeros((frames, count, 4))
    quat[:, 0, 0] = 1.0
    vel = np.zeros((frames, count, 3))
    omega = np.zeros((frames, count, 3))

    data = mj.MjData(model)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    for frame in range(frames):
        data.qpos[:] = q[frame]
        data.qvel[:] = v[frame]
        mj.mj_forward(model, data)

        p0 = np.array(data.xpos[pelvis_id], copy=True)
        R0 = np.array(data.xmat[pelvis_id]).reshape(3, 3).copy()
        mj.mj_jacBody(model, data, jp, jr, pelvis_id)
        v0 = jp @ v[frame]
        w0 = jr @ v[frame]

        for k, sid in enumerate(site_ids, start=1):
            p_site = np.array(data.site_xpos[sid], copy=True)
            R_site = np.array(data.site_xmat[sid]).reshape(3, 3)
            mj.mj_jacSite(model, data, jp, jr, sid)
            v_site = jp @ v[frame]
            w_site = jr @ v[frame]
            arm = p_site - p0
            pos[frame, k] = R0.T @ arm
            quat[frame, k] = wxyz_from_matrix(R0.T @ R_site)
            vel[frame, k] = R0.T @ (v_site - v0 - np.cross(w0, arm))
            omega[frame, k] = R0.T @ (w_site - w0)

    # Quaternion signs are not physically different, but keep interpolation easy.
    for k in range(count):
        for i in range(1, frames):
            if np.dot(quat[i - 1, k], quat[i, k]) < 0:
                quat[i, k] *= -1
    return pos, quat, vel, omega


def process_file(source, target_dir, model, mj, pelvis_name, sites, rules):
    with np.load(source, allow_pickle=False) as f:
        arrays = {key: f[key].copy() for key in f.files}
    q0 = np.asarray(arrays["qpos"], dtype=float)
    v0 = np.asarray(arrays["qvel"], dtype=float)
    if q0.ndim != 2 or v0.ndim != 2 or q0.shape != (len(v0), model.nq) or v0.shape[1] != model.nv:
        raise ValueError(f"Dimension mismatch: input qpos={q0.shape}, qvel={v0.shape}; model nq={model.nq} nv={model.nv}")
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(v0)):
        raise ValueError("Nonfinite qpos or qvel")
    if not np.allclose(q0[-1, :2] - q0[0, :2], arrays["root_displacement_xyz_m"][:2], atol=1e-6):
        raise ValueError("Input root displacement does not match stride endpoints")
    q, v, before, correction = project_equalities(q0, v0, model, rules)
    pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, pelvis_name)
    if pelvis_id < 0:
        raise ValueError(f"Unknown pelvis body: {pelvis_name}")
    site_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, n) for n in sites]
    for name, sid in zip(sites, site_ids):
        if sid < 0:
            raise ValueError(f"Unknown site {name!r}; use --list-sites")
    P, R, V, W = landmark_states(q, v, model, mj, pelvis_id, site_ids)

    arrays.update({
        "qpos": q, "qvel": v, "qpos_ref": q, "qvel_ref": v,
        "site_names": np.array([pelvis_name, *sites], dtype="U"),
        "site_kinds": np.array(["pelvis_body_origin", *(["site"] * len(sites))], dtype="U"),
        "site_pos_rel": P, "site_quat_rel": R,
        "site_linvel_rel": V, "site_angvel_rel": W,
    })
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / source.name
    if source.resolve() == output.resolve():
        raise ValueError("--output must differ from --input; source files are kept intact")
    np.savez_compressed(output, **arrays)

    # Keep the original events/manifest unchanged; don't infer new gait events.
    for suffix in ("_events.csv", "_manifest.json"):
        companion = source.with_name(source.stem + suffix)
        if companion.is_file():
            shutil.copyfile(companion, target_dir / companion.name)
    diagnostic = {
        "source": str(source), "output": str(output),
        "pelvis_body": pelvis_name, "tracking_sites": list(sites),
        "num_joint_equalities": len(rules),
        "max_source_equality_mismatch_rad_or_m": before,
        "max_dependent_joint_adjustment_rad_or_m": correction,
        "max_pelvis_relative_site_position_loop_error_m": float(np.max(np.abs(P[-1] - P[0]))),
        "max_pelvis_relative_site_velocity_loop_error_m_s": float(np.max(np.abs(V[-1] - V[0]))),
        "max_pelvis_relative_site_angular_velocity_loop_error_rad_s": float(np.max(np.abs(W[-1] - W[0]))),
        "note": "Kinematics and FK only; does not prove contact or dynamic feasibility",
    }
    (target_dir / (source.stem + "_landmarks.json")).write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8")
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Loopable reference folder or one *_loopable.npz")
    parser.add_argument("--model", type=Path, required=True, help="Same MyoLeg XML/MJB used for the original kinematic mapping")
    parser.add_argument("--output", type=Path, default=Path("gait_with_landmarks"))
    parser.add_argument("--pelvis-body", default="pelvis")
    parser.add_argument("--sites", nargs="+", default=list(DEFAULT_SITES))
    parser.add_argument("--list-sites", action="store_true", help="Show model body/site names and quit")
    args = parser.parse_args()
    import mujoco as mj

    model = (mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower() == ".mjb"
             else mj.MjModel.from_xml_path(str(args.model)))
    if args.list_sites:
        print("BODY NAMES:")
        for i in range(model.nbody):
            print(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i))
        print("\nSITE NAMES:")
        for i in range(model.nsite):
            print(mj.mj_id2name(model, mj.mjtObj.mjOBJ_SITE, i))
        return

    if not args.input.exists():
        parser.error(f"No input: {args.input}")
    if args.input.is_file():
        files = [args.input]
        root = args.input.parent
    else:
        root = args.input
        files = sorted(root.rglob("*_loopable.npz"))
    if not files:
        parser.error("No *_loopable.npz files found")
    rules = ordered_joint_equalities(model, mj)
    summary = []
    for file in files:
        try:
            dest = args.output / file.parent.relative_to(root)
            entry = process_file(file, dest, model, mj, args.pelvis_body, args.sites, rules)
            summary.append(entry)
            print(f"OK   {file}")
        except Exception as e:
            summary.append({"source": str(file), "error": str(e)})
            print(f"ERROR {file}: {e}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary_landmarks.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed {sum('error' not in s for s in summary)}/{len(files)} strides")


if __name__ == "__main__":
    main()
