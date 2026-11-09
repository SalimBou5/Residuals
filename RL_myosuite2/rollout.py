"""Step 3: policy decisions at 200 Hz, physics/MEP/phase at 1000 Hz.

This module deliberately has NO reward or complete observation implementation.
It can run scripted actions and feeds the future Gymnasium environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from phase.contacts import ContactConfig, ContactEventDetector
from control import ActionConfig, ResidualActionController
from phase.phase import PhaseConfig, PhaseController, PhaseStep
from RSI.reference import LoopableReference
from RSI.rsi import RSIInitializer, ResetResult



@dataclass(frozen=True)
class RolloutConfig:
    physics_dt_s: float = 0.001
    frame_skip: int = 5
    record_substeps: bool = False

    def __post_init__(self):
        if not np.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0 or self.frame_skip < 1:
            raise ValueError("Invalid physics timestep/frame skip")

    @property
    def control_dt_s(self) -> float:
        return self.physics_dt_s * self.frame_skip


class PhysicsControlLoop:
    """Simulator-agnostic control loop using a real MjModel, MjData and mj_step.

    The only code that writes `data.act` is the RSI reset; during rollout
    MuJoCo itself integrates the activation state each 1 ms physics step.
    """
    def __init__(self, rsi: RSIInitializer, *,
                 rollout_config: RolloutConfig = RolloutConfig(),
                 action_config: ActionConfig = ActionConfig(),
                 phase_config: PhaseConfig = PhaseConfig(),
                 contact_config: ContactConfig = ContactConfig(),
                 contact_reader: Callable[[Any, Any], np.ndarray] | None = None):
        self.rsi = rsi
        self.model, self.mj, self.data = rsi.model, rsi.mj, rsi.data
        self.config = rollout_config
        model_dt = float(self.model.opt.timestep)
        if not np.isclose(model_dt, self.config.physics_dt_s, rtol=0, atol=1e-9):
            raise ValueError(f"MyoLeg timestep {model_dt} does not match desired "
                             f"{self.config.physics_dt_s}. Keep model unchanged, "
                             "or explicitly configure a matching physics_dt_s.")
        if action_config.enable_unmapped != rsi.config.enable_unmapped:
            raise ValueError("RSI and action controller must use the same enable_unmapped flag")
        self.actions = ResidualActionController(rsi.binding, action_config)
        self.clock = PhaseController(phase_config)
        self.contact_reader = contact_reader
        self.detector = ContactEventDetector(contact_config) if contact_reader else None
        self.reset_result: ResetResult | None = None
        self.last_contact_forces = np.zeros(4, dtype=float)
        self.control_steps = 0
        self.physics_steps = 0
        self.last_substeps: list[dict] = []
        self.last_events: list[dict] = []
        self.last_phase_step: PhaseStep | None = None

    def reset(self, *, reference: LoopableReference | str | None = None,
              phase: float | None = None, seed: int | None = None) -> ResetResult:
        result = self.rsi.reset(reference=reference, phase=phase, seed=seed)
        self.reset_result = result
        self.clock.reset(result.phase_state, result.reference.duration, result.reference.events)
        self.actions.set_policy_action(np.zeros(self.actions.action_dim))
        self.control_steps = 0
        self.physics_steps = 0
        self.last_substeps = []
        self.last_events = []
        self.last_phase_step = None
        if self.detector is not None:
            self.last_contact_forces = self.contact_reader(self.model, self.data)
            self.detector.seed(self.last_contact_forces, time_s=float(self.data.time))
        else:
            self.last_contact_forces[:] = 0
        return result

    def step(self, action: np.ndarray) -> dict:
        """One RL action; exactly frame_skip internal physics integrations.

        Substep order:
          sample MEP at theta_j -> write (MEP+held residual) to data.ctrl
          -> mj_step(1ms) -> integrate phase -> detect events at new time.
        A newly detected event starts influencing phase on the NEXT substep.
        """
        if self.reset_result is None:
            raise RuntimeError("Call reset before step")
        self.actions.set_policy_action(action)
        events = []
        trace = []
        for sub in range(self.config.frame_skip):
            theta_before = self.clock.theta
            u = self.actions.excitation_at(theta_before)
            self.data.ctrl[:] = u
            if self.config.record_substeps:
                trace.append({"time_before_s": float(self.data.time),
                              "theta_before": theta_before,
                              "mep": self.actions.last_baseline.copy(),
                              "ctrl": u.copy(),
                              "residual_action": self.actions.actuator_action.copy(),
                              "policy_action": self.actions.action.copy()})
            self.mj.mj_step(self.model, self.data)
            self.physics_steps += 1
            for x_name in ("qpos", "qvel", "act"):
                if not np.all(np.isfinite(getattr(self.data, x_name))):
                    raise FloatingPointError(f"Nonfinite MyoLeg {x_name} at {self.data.time:.6f}s")
            self.last_phase_step = self.clock.advance(
                self.config.physics_dt_s, self.actions.phase_action)
            if self.detector is not None:
                self.last_contact_forces = self.contact_reader(self.model, self.data)
                detected = self.detector.update(self.last_contact_forces,
                                                time_s=float(self.data.time))
                for name in detected:
                    accepted, reason, error = self.clock.observe_event(name, time_s=float(self.data.time))
                    events.append({"event": name, "time_s": float(self.data.time),
                                   "phase": self.clock.phase, "phase_error": error,
                                   "accepted": accepted, "reason": reason})
            if self.config.record_substeps:
                trace[-1].update({"theta_after": self.clock.theta,
                                  "phase_rate_hz": self.last_phase_step.phase_rate_hz,
                                  "event_correction": self.last_phase_step.effective_event_step,
                                  "events": [e.copy() for e in events if e["time_s"] == float(self.data.time)]})
        self.control_steps += 1
        self.last_events = events
        self.last_substeps = trace
        # IMPORTANT: data.ctrl holds the LAST applied substep command. The
        # next control-step observation should sample baseline at CURRENT phi.
        return {
            "phase": self.clock.phase, "theta": self.clock.theta,
            "theta_nominal": self.clock.theta_nominal,
            "phase_rate_hz": self.clock.phase_rate_hz,
            "phase_drift": self.clock.phase_drift,
            "event_remaining": self.clock.remaining_error,
            "event_active": self.clock.active,
            "events": events,
            "physics_steps": self.physics_steps,
            "control_steps": self.control_steps,
            "time_s": float(self.data.time),
        }
