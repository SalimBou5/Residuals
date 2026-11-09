#!/usr/bin/env python3
"""Random-phase MyoLeg RSI test using the Step-2 loopable gait library.

Freeze at a randomly selected kinematic state and its precomputed muscle
activation state. SPACE releases the model into *unconstrained physics*, driven
only by the periodic bilateral MEP excitations. No tracking controller or RL.

The user's model, MEP CSV, lookup NPZ and reference are read locally.
This script has intentionally NOT been executed by its author.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


class PeriodicTable:
    """Vector-valued, piecewise-linear periodic sampler on phase [0,1)."""

    def __init__(self, phase, values, name):
        p = np.asarray(phase, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float)
        if y.ndim != 2 or len(y) != len(p) or len(p) < 3:
            raise ValueError(f"{name}: expected (samples, channels) table")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
            raise ValueError(f"{name}: NaN/inf values")
        if abs(p[0]) > 1e-6 or np.any(np.diff(p) <= 0):
            raise ValueError(f"{name}: phase must start at zero and increase")
        if p[-1] > 1.0 + 1e-6:
            raise ValueError(f"{name}: phase must be in [0, 1]")
        # Drop duplicated 100% row, if supplied. Interpolate the wrap explicitly.
        if abs(p[-1] - 1.0) < 1e-6:
            p, y = p[:-1], y[:-1]
        if p[-1] >= 1.0:
            raise ValueError(f"{name}: invalid phase end")
        self.phase, self.values = p, y

    def sample(self, phase):
        phi = float(phase) % 1.0
        idx = int(np.searchsorted(self.phase, phi, side="right") - 1)
        idx = max(idx, 0)
        j = (idx + 1) % len(self.phase)
        left = float(self.phase[idx])
        right = float(self.phase[j]) + (1.0 if j == 0 else 0.0)
        w = (phi - left) / (right - left)
        return self.values[idx] * (1.0 - w) + self.values[j] * w


def load_reference(path):
    """Step-2 outputs preserve all Step-1 qpos/qvel/time arrays."""
    with np.load(path, allow_pickle=False) as d:
        needed = ("qpos", "qvel", "time_since_start_s", "stride_duration_s")
        missing = [key for key in needed if key not in d]
        if missing:
            raise ValueError(f"Not a Step-2 reference ({missing} missing): {path}")
        qpos = np.asarray(d["qpos"], dtype=float)
        qvel = np.asarray(d["qvel"], dtype=float)
        times = np.asarray(d["time_since_start_s"], dtype=float)
        duration = float(d["stride_duration_s"])
    if (qpos.ndim != 2 or qvel.ndim != 2 or len(qpos) != len(qvel)
            or len(times) != len(qpos) or len(times) < 3
            or np.any(np.diff(times) <= 0) or duration <= 0
            or not np.isclose(times[0], 0, atol=1e-6)
            or not np.isclose(times[-1], duration, atol=1e-4)):
        raise ValueError("Invalid Step-2 reference arrays or timestamps")
    if not all(np.all(np.isfinite(x)) for x in (qpos, qvel, times)):
        raise ValueError("Reference contains NaN/inf")
    return qpos, qvel, times, duration


def sample_reference(qpos, qvel, times, duration, phase):
    """Interpolate qpos/qvel at arbitrary phase; handle free-root quaternion."""
    t = (float(phase) % 1.0) * duration
    idx = int(np.searchsorted(times, t, side="right") - 1)
    idx = min(max(0, idx), len(times) - 2)
    a = float((t - times[idx]) / (times[idx + 1] - times[idx]))
    q = (1.0 - a) * qpos[idx] + a * qpos[idx + 1]
    v = (1.0 - a) * qvel[idx] + a * qvel[idx + 1]
    qa = qpos[idx, 3:7]
    qb = qpos[idx + 1, 3:7].copy()
    if np.dot(qa, qb) < 0:
        qb *= -1
    root_quat = (1.0 - a) * qa + a * qb
    q[3:7] = root_quat / np.linalg.norm(root_quat)
    return q, v


def load_mep(path):
    """CSV: gait_cycle_pct, followed by 80 *actuator-named* muscle columns."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if "gait_cycle_pct" not in fields:
            raise ValueError("MEP CSV needs a 'gait_cycle_pct' column")
        names = [name for name in fields if name != "gait_cycle_pct"]
        if len(names) != 80 or len(set(names)) != 80:
            raise ValueError(f"Expected 80 unique MEP actuator columns, got {len(names)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty MEP CSV")
    phase = np.array([float(r["gait_cycle_pct"]) / 100.0 for r in rows])
    values = np.array([[float(r[name]) for name in names] for r in rows])
    return names, PeriodicTable(phase, values, "bilateral MEP")


def load_activation_lookup(path, duration, tolerance):
    """The lookup was made using the same MyoLeg activation dynamics and MEP."""
    with np.load(path, allow_pickle=False) as d:
        needed = ("durations_s", "phase_grid", "activation_profiles", "muscle_names")
        missing = [key for key in needed if key not in d]
        if missing:
            raise ValueError(f"Activation lookup missing keys: {missing}")
        durations = np.asarray(d["durations_s"], dtype=float).reshape(-1)
        phase = np.asarray(d["phase_grid"], dtype=float)
        profiles = np.asarray(d["activation_profiles"], dtype=float)
        names = [str(name) for name in d["muscle_names"].tolist()]
        saved_addresses = np.asarray(d["act_addresses"], dtype=int).reshape(-1) if "act_addresses" in d else None
    if profiles.ndim != 3 or profiles.shape[0] != len(durations) or profiles.shape[2] != len(names):
        raise ValueError("Lookup shapes disagree: expected (durations, phase, muscles)")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate lookup muscle names")
    closest = int(np.argmin(np.abs(durations - duration)))
    error = abs(float(durations[closest]) - duration)
    if error > tolerance:
        raise ValueError(
            f"No activation profile for T={duration:.6f}s. Nearest is "
            f"{durations[closest]:.6f}s (difference {error:.6f}s). "
            "Verify that the lookup covers this reference's duration."
        )
    print(f"Activation profile: T={durations[closest]:.6f}s "
          f"(reference {duration:.6f}s; difference {error:.6f}s)")
    return names, PeriodicTable(phase, profiles[closest], "activation lookup"), saved_addresses


def actuator_id(model, mujoco, name):
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise ValueError(f"Actuator {name!r} not in the loaded MyoLeg model")
    return idx


class RSIExperiment:
    def __init__(self, model, mujoco, qpos, qvel, times, duration, mep_names,
                 mep_table, act_names, act_table, lookup_addresses, phi0):
        self.model = model
        self.mj = mujoco
        self.sim = mujoco.MjData(model)  # self.sim.qpos, .qvel, .act, .ctrl
        self.duration = duration
        self.ref_qpos, self.ref_qvel, self.ref_times = qpos, qvel, times
        self.phi0 = phi0

        if qpos.shape[1] != model.nq or qvel.shape[1] != model.nv:
            raise ValueError(f"Model nq/nv={model.nq}/{model.nv}, "
                             f"reference={qpos.shape[1]}/{qvel.shape[1]}")
        if model.nu != 80:
            raise ValueError(f"Expected MyoLeg with 80 actuators; model.nu={model.nu}")
        if model.na < 80:
            raise ValueError(f"Model has only {model.na} activation states; expected at least 80")
        if set(mep_names) != set(act_names):
            raise ValueError("MEP and activation lookup muscle-name sets differ. "
                             f"MEP-only: {sorted(set(mep_names) - set(act_names))}; "
                             f"lookup-only: {sorted(set(act_names) - set(mep_names))}")

        self.mep = mep_table
        self.activations = act_table
        self.mep_actuator_ids = np.array([actuator_id(model, mujoco, n) for n in mep_names], dtype=int)
        self.activation_addresses = []
        for col, name in enumerate(act_names):
            aid = actuator_id(model, mujoco, name)
            adr = int(model.actuator_actadr[aid])
            nstate = int(model.actuator_actnum[aid])
            if adr < 0 or nstate != 1:
                raise ValueError(f"{name} must have one activation state (addr={adr}, n={nstate})")
            if lookup_addresses is not None and int(lookup_addresses[col]) != adr:
                raise ValueError(f"Activation address mismatch for {name}: lookup "
                                 f"{lookup_addresses[col]}, current model {adr}. "
                                 "Use the same MyoLeg model as for lookup generation.")
            self.activation_addresses.append(adr)
        if len(set(self.activation_addresses)) != 80:
            raise ValueError("Some muscles share an activation state unexpectedly")
        self.activation_addresses = np.asarray(self.activation_addresses, dtype=int)

        self.reset_qpos, self.reset_qvel = sample_reference(qpos, qvel, times, duration, phi0)
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_id < 0:
            raise ValueError("MyoLeg model missing 'pelvis' body")
        self.reset()

    def reset(self, phase=None):
        if phase is not None:
            self.phi0 = float(phase) % 1.0
            self.reset_qpos, self.reset_qvel = sample_reference(
                self.ref_qpos, self.ref_qvel, self.ref_times, self.duration, self.phi0)
        self.mj.mj_resetData(self.model, self.sim)
        self.sim.qpos[:] = self.reset_qpos
        self.sim.qvel[:] = self.reset_qvel
        a0 = self.activations.sample(self.phi0)
        if not np.all(np.isfinite(a0)) or np.min(a0) < -1e-5 or np.max(a0) > 1 + 1e-5:
            raise ValueError(f"Lookup activations outside [0,1]: {a0.min()} .. {a0.max()}")
        # The requested RSI muscle initialization: once, *before* physics stepping.
        self.sim.act[self.activation_addresses] = np.clip(a0, 0.0, 1.0)
        self.sim.ctrl[:] = 0.0  # Frozen display: commands begin only after SPACE.
        self.sim.time = 0.0
        self.mj.mj_forward(self.model, self.sim)
        self.pin_pelvis()
        print(f"RSI phase={self.phi0:.4f}, T={self.duration:.3f}s, "
              f"muscle act range={a0.min():.3f}..{a0.max():.3f}, "
              f"initial contacts={self.sim.ncon}")

    def configure_pelvis_tether(self, enabled):
        """Pin the free-root pose at each reset (kinematic tether)."""
        self.tether_pelvis = enabled
        if not enabled:
            return
        # Find the free joint on the pelvis or one of its ancestor bodies.
        body_id = self.pelvis_id
        free_joint = None
        while body_id > 0 and free_joint is None:
            for jid in range(int(self.model.body_jntadr[body_id]),
                             int(self.model.body_jntadr[body_id] + self.model.body_jntnum[body_id])):
                if self.model.jnt_type[jid] == self.mj.mjtJoint.mjJNT_FREE:
                    free_joint = jid
                    break
            body_id = int(self.model.body_parentid[body_id])
        if free_joint is None:
            raise ValueError("--tether-pelvis requires a free root joint on the pelvis/ancestor")
        self.root_qadr = int(self.model.jnt_qposadr[free_joint])
        self.root_vadr = int(self.model.jnt_dofadr[free_joint])
        self.pin_pelvis()

    def pin_pelvis(self):
        """Hold the root's 7 qpos and 6 qvel fixed; legs remain free."""
        if not getattr(self, "tether_pelvis", False):
            return
        self.sim.qpos[self.root_qadr:self.root_qadr + 7] = self.reset_qpos[self.root_qadr:self.root_qadr + 7]
        self.sim.qvel[self.root_vadr:self.root_vadr + 6] = 0.0
        self.mj.mj_forward(self.model, self.sim)

    def apply_mep(self):
        phase = (self.phi0 + self.sim.time / self.duration) % 1.0
        values = self.mep.sample(phase)
        if np.any(~np.isfinite(values)) or np.min(values) < -1e-5 or np.max(values) > 1 + 1e-5:
            raise ValueError(f"MEP excitation outside [0,1] at phase {phase:.4f}")
        self.sim.ctrl[:] = 0.0
        self.sim.ctrl[self.mep_actuator_ids] = np.clip(self.mep_scale * values, 0.0, 1.0)
        return phase


def plot_actual_activations(trace, muscle_names, activation_addresses, output):
    """Plot actual sim.act separately for left and right MyoLeg muscles."""
    import matplotlib.pyplot as plt

    t = np.asarray(trace["time_s"], dtype=float)
    actual = np.asarray(trace["act"], dtype=float)[:, activation_addresses]

    for side, suffix in (("left", "_l"), ("right", "_r")):
        indices = [i for i, name in enumerate(muscle_names)
                   if str(name).lower().endswith(suffix)]
        if not indices:
            print(f"WARNING: No {side} muscles found (expected names ending in {suffix}).")
            continue

        # One figure per leg, with both an overview and readable curves.
        ncols = 2
        nrows = int(np.ceil(len(indices) / 10))
        fig, axes = plt.subplots(nrows, ncols, figsize=(17, max(9, 3.5 * nrows)),
                                 sharex=True, sharey=True, squeeze=False)
        for group in range(nrows):
            ax = axes[group, 0]
            subset = indices[group * 10:(group + 1) * 10]
            for i in subset:
                ax.plot(t, actual[:, i], linewidth=1.2, label=muscle_names[i])
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel("Activation")
            ax.set_title(f"Muscles {group * 10 + 1}–{group * 10 + len(subset)}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            ax.grid(alpha=0.25)
            axes[group, 1].axis("off")

        # Use the right half of the figure for a heatmap of this leg.
        # Create it as a separate full-height axes so names remain readable.
        for row in range(nrows):
            axes[row, 1].remove()
        fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07,
                            wspace=0.5, hspace=0.3)
        pos_top = axes[0, 0].get_position()
        pos_bottom = axes[-1, 0].get_position()
        heat_ax = fig.add_axes([0.63, pos_bottom.y0, 0.30,
                                pos_top.y1 - pos_bottom.y0])
        image = heat_ax.imshow(actual[:, indices].T, origin="lower", aspect="auto",
                               interpolation="nearest",
                               extent=[t[0], max(t[-1], t[0] + 1e-6),
                                       -0.5, len(indices) - 0.5], vmin=0, vmax=1)
        heat_ax.set_yticks(range(len(indices)))
        heat_ax.set_yticklabels([muscle_names[i] for i in indices], fontsize=7)
        heat_ax.set_xlabel("Simulation time (s)")
        heat_ax.set_title("Actual activation heatmap")
        fig.colorbar(image, ax=heat_ax, label="Activation", fraction=0.04, pad=0.02)
        axes[-1, 0].set_xlabel("Simulation time (s)")
        fig.suptitle(f"{side.capitalize()} leg — actual MuJoCo sim.act ({len(indices)} muscles)",
                     fontsize=15)
        path = output.with_name(output.stem + f"_activations_{side}.png")
        fig.savefig(path, dpi=180)
        print(f"Saved {side} muscle activation plot: {path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Exact MuJoCo MyoLeg XML/MJB")
    parser.add_argument("--reference", type=Path, required=True, help="Step-2 *_loopable.npz")
    parser.add_argument("--mep", type=Path, required=True, help="myoleg_bilateral_mep_80.csv")
    parser.add_argument("--lookup", type=Path, required=True, help="activation_lookup.npz")
    parser.add_argument("--tether-pelvis", action="store_true",
                        help="Keep pelvis/root pose fixed while allowing leg joints to move")
    parser.add_argument("--mep-scale", type=float, default=1.0,
                        help="Multiply all MEP excitations by this factor, then clip to [0,1] (default: 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Reproducible random phase")
    parser.add_argument("--phase", type=float, default=None, help="Explicit phase in [0,1), overrides random")
    parser.add_argument("--duration-tolerance", type=float, default=0.005,
                        help="Maximum difference to selected lookup stride duration (seconds)")
    parser.add_argument("--seconds", type=float, default=5.0, help="Seconds of PHYSICS after Space")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation time / wall time")
    parser.add_argument("--viewer-fps", type=int, default=30)
    parser.add_argument("--log-hz", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("rsi_mep_trace.npz"))
    args = parser.parse_args()
    if args.phase is not None and not (0 <= args.phase < 1):
        parser.error("--phase must be in [0,1)")
    if args.seconds <= 0 or args.speed <= 0 or args.viewer_fps < 1 or args.log_hz <= 0:
        parser.error("--seconds, --speed, --viewer-fps and --log-hz must be positive")
    if not np.isfinite(args.mep_scale) or args.mep_scale < 0:
        parser.error("--mep-scale must be finite and nonnegative")
    if args.duration_tolerance < 0:
        parser.error("--duration-tolerance must be nonnegative")

    import glfw
    import mujoco as mj
    import mujoco.viewer

    model = (mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower() == ".mjb"
             else mj.MjModel.from_xml_path(str(args.model)))
    qpos, qvel, times, duration = load_reference(args.reference)
    mep_names, mep = load_mep(args.mep)
    act_names, activation, saved_addresses = load_activation_lookup(
        args.lookup, duration, args.duration_tolerance)
    rng = np.random.default_rng(args.seed)
    phi0 = args.phase if args.phase is not None else float(rng.random())
    experiment = RSIExperiment(model, mj, qpos, qvel, times, duration,
                               mep_names, mep, act_names, activation, saved_addresses, phi0)
    experiment.mep_scale = args.mep_scale
    experiment.configure_pelvis_tether(args.tether_pelvis)
    print(f"MEP excitation scale: {args.mep_scale:g} (clipped to [0, 1])")
    sim = experiment.sim

    flags = {"release": False, "reset": False, "quit": False}

    def on_key(key):
        if key == glfw.KEY_SPACE:
            flags["release"] = True
        elif key == glfw.KEY_R:
            flags["reset"] = True
        elif key == glfw.KEY_Q:
            flags["quit"] = True

    # Log actual simulation output, NOT a replay of the target kinematics.
    trace = {name: [] for name in ("time_s", "phase", "qpos", "qvel", "act", "ctrl")}
    last_log = -float("inf")

    def record():
        nonlocal last_log
        now = float(sim.time)
        if now - last_log < (1.0 / args.log_hz) - 1e-9:
            return
        last_log = now
        trace["time_s"].append(now)
        trace["phase"].append((experiment.phi0 + now / duration) % 1.0)
        for name in ("qpos", "qvel", "act", "ctrl"):
            trace[name].append(getattr(sim, name).copy())

    released = False
    finished = False
    sim_target = 0.0
    frame_period = 1.0 / args.viewer_fps

    print("\nFROZEN: random-phase qpos + qvel + periodic lookup activations are initialized.")
    print("SPACE: release | R: reset at a NEW random phase | Q: close viewer")
    print("There is NO leg kinematic tracking, balance controller, phase feedback, or RSI correction.")
    if args.tether_pelvis:
        print("PELVIS TETHER ON: root pose pinned after each physics step; leg joints remain dynamic.")
    with mj.viewer.launch_passive(model, sim, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = sim.xpos[experiment.pelvis_id]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -12
            viewer.cam.azimuth = 90

        while viewer.is_running() and not flags["quit"]:
            frame_start = time.perf_counter()
            if flags["reset"]:
                flags["reset"] = False
                flags["release"] = False
                released = False
                finished = False
                sim_target = 0.0
                last_log = -float("inf")
                # Keep the output trace associated with the most recent trial.
                trace = {name: [] for name in trace}
                with viewer.lock():
                    experiment.reset(phase=float(rng.random()))
                    viewer.cam.lookat[:] = sim.xpos[experiment.pelvis_id]
                print("RESET: frozen at new random phase; press SPACE to release.")
            if flags["release"] and not released:
                released = True
                with viewer.lock():
                    experiment.apply_mep()
                    # Update initial physics with the actual baseline excitation.
                    mj.mj_forward(model, sim)

                    # 1. Initial root velocities
                    root_lin_vel = np.linalg.norm(sim.qvel[:3])
                    root_ang_vel = np.linalg.norm(sim.qvel[3:6])

                    # 2. Maximum floor penetration
                    floor_id = mj.mj_name2id(
                        model, mj.mjtObj.mjOBJ_GEOM, "floor"
                    )

                    penetrations = []

                    for i in range(sim.ncon):
                        c = sim.contact[i]

                        if c.geom1 == floor_id or c.geom2 == floor_id:
                            penetrations.append(max(0.0, -c.dist))

                    max_penetration = max(penetrations, default=0.0)

                    # 3. Initial root acceleration
                    root_lin_acc = np.linalg.norm(sim.qacc[:3])

                    print(
                        f"RSI phase={experiment.phi0:.3f} | "
                        f"root speed={root_lin_vel:.3f} m/s | "
                        f"root ang. speed={root_ang_vel:.3f} rad/s | "
                        f"penetration={max_penetration*1000:.1f} mm | "
                        f"root acceleration={root_lin_acc:.2f} m/s²"
                    )
                    record()  # Capture the exact RSI state before the first physics step.
                print(f"RELEASE at reference phase {experiment.phi0:.4f}. Physics stepping begins.")
            if released and not finished:
                sim_target = min(args.seconds, sim_target + args.speed * frame_period)
                with viewer.lock():
                    while float(sim.time) + 1e-10 < sim_target:
                        experiment.apply_mep()  # MEP is evaluated at the current PHYSICAL gait phase.
                        mj.mj_step(model, sim)
                        experiment.pin_pelvis()
                        record()
                        if not np.all(np.isfinite(sim.qpos)) or not np.all(np.isfinite(sim.qvel)):
                            print("ERROR: Nonfinite simulation state; stopping physics.")
                            finished = True
                            break
                    if experiment.pelvis_id >= 0:
                        viewer.cam.lookat[:] = sim.xpos[experiment.pelvis_id]
                if float(sim.time) >= args.seconds - 1e-8 and not finished:
                    finished = True
                    print("Finished prescribed simulation duration; final state held until Q/close.")
                if finished:
                    print(f"Final physics time={sim.time:.3f}s, "
                          f"pelvis world position={sim.xpos[experiment.pelvis_id]}")
            viewer.sync()
            time.sleep(max(0.0, frame_period - (time.perf_counter() - frame_start)))

    if trace["time_s"]:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output,
                            time_s=np.asarray(trace["time_s"]),
                            phase=np.asarray(trace["phase"]),
                            qpos=np.asarray(trace["qpos"]),
                            qvel=np.asarray(trace["qvel"]),
                            act=np.asarray(trace["act"]),
                            ctrl=np.asarray(trace["ctrl"]),
                            reset_phase=np.array(experiment.phi0),
                            mep_scale=np.array(args.mep_scale),
                            tether_pelvis=np.array(args.tether_pelvis),
                            stride_duration_s=np.array(duration),
                            model_path=np.array(str(args.model)),
                            reference_path=np.array(str(args.reference)))
        print(f"Saved actual simulated trajectory: {args.output}")
        plot_actual_activations(trace, act_names, experiment.activation_addresses, args.output)
    else:
        print("No simulation was released; no physics trace saved.")


if __name__ == "__main__":
    main()
