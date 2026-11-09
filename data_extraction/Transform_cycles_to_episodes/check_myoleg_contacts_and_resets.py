#!/usr/bin/env python3
"""Offline MyoLeg gait-reference check: foot/ground geometry and reset states.

Reads Step-2 *_loopable.npz (qpos, qvel, phase, stride duration) and its
companion *_manifest.json. Never steps physics, changes input files, infers
new gait events, or asserts that a reset is dynamically stable.

Requirements: pip install numpy matplotlib mujoco

Examples (PowerShell, one line):
    python check_myoleg_contacts_and_resets.py  --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml"  --output foot_ground_checks --reference "gait_with_landmarks\S045\Trial01\step01_loopable.npz"
    python check_myoleg_contacts_and_resets.py  --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml"  --output foot_ground_checks --reference "gait_with_landmarks" --max-files 969
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Only save offline plots; never open/update a GUI.
import matplotlib.pyplot as plt
import numpy as np
import mujoco as mj


DEFAULT_FOOT_BODIES = {
    "right": ("calcn_r", "toes_r", "talus_r", "foot_r"),
    "left": ("calcn_l", "toes_l", "talus_l", "foot_l"),
}


def name_id(model, obj, name):
    return int(mj.mj_name2id(model, obj, name))


def list_names(model):
    for heading, kind, n in (("GEOMS", mj.mjtObj.mjOBJ_GEOM, model.ngeom),
                             ("BODIES", mj.mjtObj.mjOBJ_BODY, model.nbody),
                             ("SITES", mj.mjtObj.mjOBJ_SITE, model.nsite)):
        print(f"\n{heading}:")
        for i in range(n):
            print(f"  {i:>3}  {mj.mj_id2name(model, kind, i)}")


def get_floor(model, data, requested):
    """Find one flat, world-fixed plane or box; use --ground-geom if ambiguous."""
    if requested:
        gid = name_id(model, mj.mjtObj.mjOBJ_GEOM, requested)
        if gid < 0:
            raise ValueError(f"No ground geom {requested!r}. Use --list-model-names.")
    else:
        planes = [i for i in range(model.ngeom)
                  if int(model.geom_bodyid[i]) == 0
                  and int(model.geom_type[i]) == int(mj.mjtGeom.mjGEOM_PLANE)]
        if len(planes) != 1:
            raise ValueError(f"Found {len(planes)} static plane geoms. "
                             "Specify --ground-geom NAME (see --list-model-names).")
        gid = planes[0]
    geom_type = int(model.geom_type[gid])
    if geom_type not in (int(mj.mjtGeom.mjGEOM_PLANE), int(mj.mjtGeom.mjGEOM_BOX)):
        raise ValueError("Ground geom must be a plane or a flat-topped box for site heights")
    if int(model.geom_bodyid[gid]) != 0:
        raise ValueError("Selected ground geom is not attached to the static world body")

    # The reference ground surface is the local +Z face/normal.
    data.qpos[:] = model.qpos0
    mj.mj_fwdPosition(model, data)
    R = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
    normal = R[:, 2].copy()
    origin = np.asarray(data.geom_xpos[gid]).copy()
    if geom_type == int(mj.mjtGeom.mjGEOM_BOX):
        origin += normal * float(model.geom_size[gid, 2])
    ground_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
    return gid, origin, normal, ground_name


def site_id(model, supplied, fallbacks, description):
    for name in [supplied, *fallbacks]:
        if name:
            sid = name_id(model, mj.mjtObj.mjOBJ_SITE, name)
            if sid >= 0:
                if name != supplied:
                    print(f"NOTE: using site {name!r} for {description}")
                return sid, name
    raise ValueError(f"No {description} marker site found. "
                     "Use --list-model-names and --right/left-heel/toe-site NAME.")


def foot_geom_ids(model, supplied_names, side):
    ids = {name_id(model, mj.mjtObj.mjOBJ_BODY, s) for s in supplied_names}
    ids.discard(-1)
    if not ids:
        raise ValueError(f"No foot body found for {side}. Use --list-model-names "
                         f"and --{side}-foot-bodies BODY [BODY ...].")

    def descends_from(body):
        while body > 0:
            if body in ids:
                return True
            body = int(model.body_parentid[body])
        return False

    result = {g for g in range(model.ngeom) if descends_from(int(model.geom_bodyid[g]))}
    if not result:
        raise ValueError(f"No collision geoms found in {side} foot bodies")
    print(f"  {side} foot: {len(ids)} named bodies, {len(result)} candidate geoms")
    return result


def load_events(reference, T):
    companion = reference.with_name(reference.stem + "_manifest.json")
    if not companion.is_file():
        print(f"WARNING: no event manifest {companion.name}; only right-HS boundary is known")
        return [(0.0, "R HS", "stride boundary", "black")], None, None
    manifest = json.loads(companion.read_text(encoding="utf-8"))
    src = manifest.get("events", {})
    events = [(0.0, "R HS", "dataset stride boundary", "black")]

    def append(phase, label, source, color):
        if phase is None:
            return None
        phase = float(phase)
        if not (0 <= phase < 1):
            print(f"WARNING: ignoring invalid event phase {label}={phase}")
            return None
        events.append((phase, label, source or "source unknown", color))
        return phase

    left_hs = src.get("left_hs_phase_in_right_stride")
    if left_hs is None and src.get("left_hs_time_s") is not None and manifest.get("t0_s") is not None:
        left_hs = (float(src["left_hs_time_s"]) - float(manifest["t0_s"])) / T
    append(left_hs, "L HS", src.get("left_hs_source"), "tab:green")
    right_to = append(src.get("right_toe_off_phase"), "R TO",
                      src.get("right_toe_off_source"), "tab:orange")
    left_to = append(src.get("left_toe_off_phase"), "L TO",
                     src.get("left_toe_off_source"), "tab:purple")
    return sorted(events), right_to, (None if left_hs is None else float(left_hs), left_to)


def joint_limit_excess(model, q):
    """Independent scalar joint limit excess only (dependent knee joints excluded)."""
    dependent = {int(model.eq_obj1id[e]) for e in range(model.neq)
                 if int(model.eq_type[e]) == int(mj.mjtEq.mjEQ_JOINT)
                 and (not hasattr(model, "eq_active0") or bool(model.eq_active0[e]))}
    worst = 0.0
    count = 0
    for jid in range(model.njnt):
        if jid in dependent or not bool(model.jnt_limited[jid]):
            continue
        if int(model.jnt_type[jid]) not in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
            continue
        pos = q[:, int(model.jnt_qposadr[jid])]
        lo, hi = model.jnt_range[jid]
        excess = np.maximum(np.maximum(lo - pos, pos - hi), 0)
        worst = max(worst, float(np.max(excess)))
        count += int(np.count_nonzero(excess > 1e-6))
    return worst, count


def equality_residual(model, q):
    worst = 0.0
    for e in range(model.neq):
        if int(model.eq_type[e]) != int(mj.mjtEq.mjEQ_JOINT):
            continue
        if hasattr(model, "eq_active0") and not bool(model.eq_active0[e]):
            continue
        j1, j2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if j1 < 0:
            continue
        y = q[:, int(model.jnt_qposadr[j1])]
        if j2 >= 0:
            adr = int(model.jnt_qposadr[j2])
            dx = q[:, adr] - model.qpos0[adr]
        else:
            dx = np.zeros(len(q))
        c = np.asarray(model.eq_data[e, :5])
        target = (model.qpos0[int(model.jnt_qposadr[j1])] + c[0] + c[1]*dx
                  + c[2]*dx**2 + c[3]*dx**3 + c[4]*dx**4)
        worst = max(worst, float(np.max(np.abs(y - target))))
    return worst


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def save_plot(path, phase, height, speed, contacts, penetration, pelvis_height, events, stem):
    fig, axs = plt.subplots(4, 2, figsize=(13, 13), sharex=True, constrained_layout=True)
    for column, side in enumerate(("right", "left")):
        h, s = height[side], speed[side]
        ax = axs[0, column]
        ax.plot(phase, h["heel"] * 1000, label="Heel marker")
        ax.plot(phase, h["toe"] * 1000, label="Toe marker")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(title=f"{side.capitalize()} foot — markers above ground",
               ylabel="Signed marker height (mm)")
        ax.legend(fontsize=8)
        ax = axs[1, column]
        ax.plot(phase, s["heel"], label="Heel marker")
        ax.plot(phase, s["toe"], label="Toe marker")
        ax.set(title="Tangential marker speed (world frame)", ylabel="Speed (m/s)")
        ax.legend(fontsize=8)
        ax = axs[2, column]
        ax.step(phase, contacts[side], where="mid")
        ax.set(title="Foot–ground collision pairs at/near contact", ylabel="Contact count")
        ax = axs[3, column]
        ax.plot(phase, penetration[side] * 1000, label="Foot ground-penetration depth")
        if column == 0:
            ax.plot(phase, pelvis_height * 1000, color="0.5", alpha=0.6,
                    label="Pelvis height above ground")
        ax.set(title="Geometry penetration (not contact force)", ylabel="Depth / height (mm)",
               xlabel="Stride phase (right HS = 0/1)")
        ax.legend(fontsize=8)

    for ax in axs.flat:
        for ph, label, source, color in events:
            ax.axvline(ph, color=color, lw=0.85, ls="--", alpha=0.7)
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 1)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=color, ls="--", label=f"{label} ({source})")
              for ph, label, source, color in events]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(stem + " — foot/ground plausibility (kinematics only)", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def check_one(reference, model, args, output):
    with np.load(reference, allow_pickle=False) as d:
        q = np.asarray(d["qpos"], dtype=float)
        v = np.asarray(d["qvel"], dtype=float)
        T = float(d["stride_duration_s"])
        t = np.asarray(d["time_since_start_s"], dtype=float)
        disp = np.asarray(d["root_displacement_xyz_m"], dtype=float)
    if q.shape != (len(t), model.nq) or v.shape != (len(t), model.nv):
        raise ValueError(f"qpos/qvel shapes incompatible with model: {q.shape}, {v.shape}")
    if len(t) < 5 or T <= 0 or not np.all(np.diff(t) > 0):
        raise ValueError("Invalid time grid or duration")
    if not np.isclose(t[0], 0) or not np.isclose(t[-1], T):
        raise ValueError("Expected t=0 to T inclusive")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
        raise ValueError("Nonfinite input qpos or qvel")
    phase = t / T
    events, right_to, left_events = load_events(reference, T)

    data = mj.MjData(model)
    ground_id, origin, normal, ground_name = get_floor(model, data, args.ground_geom)
    print(f"  Ground: {ground_name!r}")
    site_ids = {}
    site_used = {}
    for side, short in (("right", "r"), ("left", "l")):
        for kind in ("heel", "toe"):
            supplied = getattr(args, f"{side}_{kind}_site")
            fallbacks = ([f"heel_{short}", f"{short.upper()}CAL"] if kind == "heel" else
                         [f"toe_{short}", f"{short.upper()}TOE"])
            sid, name = site_id(model, supplied, fallbacks, f"{side} {kind}")
            site_ids[side, kind] = sid
            site_used[f"{side}_{kind}"] = name
    foot_geoms = {
        "right": foot_geom_ids(model, args.right_foot_bodies, "right"),
        "left": foot_geom_ids(model, args.left_foot_bodies, "left"),
    }
    pelvis_id = name_id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("No pelvis body. This script expects MyoLeg's pelvis body")

    N = len(t)
    height = {side: {kind: np.zeros(N) for kind in ("heel", "toe")}
              for side in ("right", "left")}
    speed = {side: {kind: np.zeros(N) for kind in ("heel", "toe")}
             for side in ("right", "left")}
    counts = {side: np.zeros(N, dtype=int) for side in ("right", "left")}
    penetration = {side: np.zeros(N) for side in ("right", "left")}
    any_ground_pen = np.zeros(N)
    pelvis_height = np.zeros(N)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))

    for i in range(N):
        data.qpos[:] = q[i]
        data.qvel[:] = v[i]
        # No mj_step: FK, collision geometry and Jacobians at the recorded state.
        mj.mj_fwdPosition(model, data)
        pelvis_height[i] = float(np.dot(np.asarray(data.xpos[pelvis_id]) - origin, normal))
        for side in ("right", "left"):
            for kind in ("heel", "toe"):
                sid = site_ids[side, kind]
                p = np.asarray(data.site_xpos[sid])
                height[side][kind][i] = float(np.dot(p - origin, normal))
                mj.mj_jacSite(model, data, jp, jr, sid)
                velocity = jp @ v[i]
                speed[side][kind][i] = float(np.linalg.norm(velocity - normal * np.dot(normal, velocity)))

        for j in range(data.ncon):
            c = data.contact[j]
            if hasattr(c, "geom"):
                a, b = map(int, c.geom[:2])
            else:  # Older MuJoCo bindings
                a, b = int(c.geom1), int(c.geom2)
            if ground_id not in (a, b):
                continue
            other = b if a == ground_id else a
            distance = float(c.dist)
            any_ground_pen[i] = max(any_ground_pen[i], -distance)
            for side in ("right", "left"):
                if other in foot_geoms[side]:
                    penetration[side][i] = max(penetration[side][i], -distance)
                    if distance <= args.contact_tolerance_mm / 1000.0:
                        counts[side][i] += 1

    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for i in range(N):
        diagnostics.append({
            "phase": float(phase[i]), "time_s": float(t[i]),
            "right_heel_marker_height_mm": float(1000*height["right"]["heel"][i]),
            "right_toe_marker_height_mm": float(1000*height["right"]["toe"][i]),
            "left_heel_marker_height_mm": float(1000*height["left"]["heel"][i]),
            "left_toe_marker_height_mm": float(1000*height["left"]["toe"][i]),
            "right_heel_tangential_speed_m_s": float(speed["right"]["heel"][i]),
            "right_toe_tangential_speed_m_s": float(speed["right"]["toe"][i]),
            "left_heel_tangential_speed_m_s": float(speed["left"]["heel"][i]),
            "left_toe_tangential_speed_m_s": float(speed["left"]["toe"][i]),
            "right_ground_contact_count": int(counts["right"][i]),
            "left_ground_contact_count": int(counts["left"][i]),
            "right_max_penetration_mm": float(1000*penetration["right"][i]),
            "left_max_penetration_mm": float(1000*penetration["left"][i]),
            "any_ground_max_penetration_mm": float(1000*any_ground_pen[i]),
        })
    write_csv(output / "signals.csv", diagnostics)

    resets = []
    excess, violations = joint_limit_excess(model, q)
    eq_residual = equality_residual(model, q)
    quat_err = float(np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)))
    for desired_phase in np.arange(0, 1, 0.125):
        i = int(np.argmin(np.abs(phase - desired_phase)))
        resets.append({
            "requested_phase": float(desired_phase),
            "actual_phase": float(phase[i]),
            "frame": i,
            "pelvis_height_above_ground_m": float(pelvis_height[i]),
            "right_heel_marker_height_mm": float(1000*height["right"]["heel"][i]),
            "right_toe_marker_height_mm": float(1000*height["right"]["toe"][i]),
            "left_heel_marker_height_mm": float(1000*height["left"]["heel"][i]),
            "left_toe_marker_height_mm": float(1000*height["left"]["toe"][i]),
            "right_ground_contacts": int(counts["right"][i]),
            "left_ground_contacts": int(counts["left"][i]),
            "max_ground_penetration_mm": float(1000*any_ground_pen[i]),
        })
    write_csv(output / "reset_states.csv", resets)

    stats = {}
    for side in ("right", "left"):
        stats[side] = {
            "max_foot_penetration_mm": float(1000*np.max(penetration[side])),
            "fraction_frames_with_near_contact": float(np.mean(counts[side] > 0)),
            "min_heel_marker_height_mm": float(1000*np.min(height[side]["heel"])),
            "min_toe_marker_height_mm": float(1000*np.min(height[side]["toe"])),
        }
    # Annotation-only stance/swing comparison: the TO values may be ASSUMED.
    if right_to is not None:
        stance = (phase < right_to)
        swing = (phase > right_to) & (phase < 1)
        stats["right"]["near_contact_fraction_annotated_stance"] = float(np.mean(counts["right"][stance] > 0)) if np.any(stance) else None
        stats["right"]["near_contact_fraction_annotated_swing"] = float(np.mean(counts["right"][swing] > 0)) if np.any(swing) else None
    if left_events is not None and all(x is not None for x in left_events):
        left_hs, left_to = left_events
        stance = (phase < left_to) | (phase >= left_hs)
        swing = (phase > left_to) & (phase < left_hs)
        stats["left"]["near_contact_fraction_annotated_stance"] = float(np.mean(counts["left"][stance] > 0)) if np.any(stance) else None
        stats["left"]["near_contact_fraction_annotated_swing"] = float(np.mean(counts["left"][swing] > 0)) if np.any(swing) else None

    seam_q = q[-1].copy()
    seam_q[:2] -= disp[:2]
    if np.dot(seam_q[3:7], q[0, 3:7]) < 0:
        seam_q[3:7] *= -1  # Equivalent quaternion, not an artificial overwrite
    report = {
        "reference": str(reference), "model": str(args.model),
        "ground_geom": ground_name, "foot_marker_sites": site_used,
        "foot_body_names": {"right": list(args.right_foot_bodies), "left": list(args.left_foot_bodies)},
        "stride_duration_s": T, "num_frames": N,
        "contact_tolerance_mm": args.contact_tolerance_mm,
        "events": [{"phase": ph, "label": label, "source": source} for ph, label, source, color in events],
        "geometry": stats,
        "max_any_ground_penetration_mm": float(1000*np.max(any_ground_pen)),
        "pelvis_height_above_ground_min_m": float(np.min(pelvis_height)),
        "pelvis_height_above_ground_max_m": float(np.max(pelvis_height)),
        "max_independent_joint_limit_excess_rad_or_m": excess,
        "num_independent_joint_limit_violating_samples": violations,
        "max_joint_equality_residual_rad_or_m": eq_residual,
        "max_root_quaternion_norm_error": quat_err,
        "max_seam_qpos_error_excluding_xy_drift": float(np.max(np.abs(seam_q - q[0]))),
        "max_seam_qvel_error": float(np.max(np.abs(v[-1] - v[0]))),
        "notes": [
            "No simulation stepping; collision and geometry checks only.",
            "Site markers are anatomical markers, NOT outsole collision points. Their height is not sole clearance.",
            "Contacts here refer to geom collision pairs, NOT measured ground reaction force or stable support.",
            "Toe-off phases may be assumed rather than measured; consult events[].source.",
            "Reset checks use nearest saved grid frames; arbitrary interpolated phase resets need later testing.",
            "Muscle activations, muscle forces, balance and dynamic stability are not verified.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plot(output / "foot_ground_check.png", phase, height, speed, counts,
              penetration, pelvis_height, events, reference.stem)
    print(f"  Max foot penetrations: R={stats['right']['max_foot_penetration_mm']:.1f} mm, "
          f"L={stats['left']['max_foot_penetration_mm']:.1f} mm")
    print(f"  Saved: {output / 'foot_ground_check.png'}")
    return {"reference": str(reference), "output": str(output), "geometry": stats,
            "max_any_ground_penetration_mm": report["max_any_ground_penetration_mm"],
            "max_joint_equality_residual": eq_residual}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="MyoLeg XML/MJB used for Step 2")
    p.add_argument("--reference", type=Path, required=True, help="One Step-2 NPZ or folder containing *_loopable.npz")
    p.add_argument("--output", type=Path, default=Path("foot_ground_checks"))
    p.add_argument("--max-files", type=int, default=6, help="Directory mode: first N files; 0 means all")
    p.add_argument("--list-model-names", action="store_true")
    p.add_argument("--ground-geom", help="Exact floor geom name; automatically selected if only one static plane")
    p.add_argument("--right-heel-site", default="RCAL")
    p.add_argument("--left-heel-site", default="LCAL")
    p.add_argument("--right-toe-site", default="RTOE")
    p.add_argument("--left-toe-site", default="LTOE")
    p.add_argument("--right-foot-bodies", nargs="+", default=list(DEFAULT_FOOT_BODIES["right"]))
    p.add_argument("--left-foot-bodies", nargs="+", default=list(DEFAULT_FOOT_BODIES["left"]))
    p.add_argument("--contact-tolerance-mm", type=float, default=2.0,
                   help="For displaying near-contact pairs only; NOT a validity threshold")
    args = p.parse_args()
    model = (mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower() == ".mjb"
             else mj.MjModel.from_xml_path(str(args.model)))
    if args.list_model_names:
        list_names(model)
        return
    if args.contact_tolerance_mm < 0 or args.max_files < 0:
        p.error("--contact-tolerance-mm and --max-files must be >= 0")
    if args.reference.is_file():
        refs = [args.reference]
        root = args.reference.parent
    else:
        if not args.reference.is_dir():
            p.error(f"No such reference: {args.reference}")
        root = args.reference
        refs = sorted(root.rglob("*_loopable.npz"))
        if args.max_files:
            refs = refs[:args.max_files]
    if not refs:
        p.error("No *_loopable.npz references found")
    reports = []
    for ref in refs:
        out = args.output / ref.parent.relative_to(root) / ref.stem
        try:
            print(f"CHECK {ref}")
            reports.append(check_one(ref, model, args, out))
        except Exception as exc:
            print(f"ERROR {ref}: {exc}")
            reports.append({"reference": str(ref), "error": str(exc)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Finished {len(reports)} checks; see {args.output / 'index.json'}")


if __name__ == "__main__":
    main()
