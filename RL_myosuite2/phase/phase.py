"""Step 3: gait clock at physics frequency, with buffered event synchronization.

The event error is measured ONLY when a physical contact transition is accepted.
We then discharge that fixed error over multiple physics steps, accounting for
phase-rate clipping. We never repeatedly chase a static event target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np

import sys
from pathlib import Path

# Add the root 'RL_myosuite2' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from RSI.reference import ReferenceEvents
from RSI.rsi import PhaseReset


def wrap_half_cycle(error: float) -> float:
    """Signed shortest phase difference in [-0.5, 0.5)."""
    return (float(error) + 0.5) % 1.0 - 0.5


@dataclass(frozen=True)
class PhaseConfig:
    kappa_max: float = 0.30
    event_gain: float = 0.50  # requested correction limited to gain * dt / T
    activation_tolerance: float = 0.03  # cycles: trigger only at a new event
    completion_tolerance: float = 1e-4  # cycles: finish an existing correction
    min_rate_ratio: float = 0.20  # final increment lower bound relative to dt/T
    max_rate_ratio: float = 2.00  # final increment upper bound relative to dt/T
    event_phase_gate: float = 0.30  # ignore wildly implausible detected timing
    use_assumed_toe_off: bool = False
    min_same_event_interval_fraction: float = 0.35
    min_any_event_interval_s: float = 0.02

    def __post_init__(self) -> None:
        if (not 0 <= self.kappa_max < 1 or self.event_gain <= 0 or
                not 0 <= self.completion_tolerance < self.activation_tolerance < .5 or
                not 0 < self.min_rate_ratio < self.max_rate_ratio or
                self.event_phase_gate <= self.activation_tolerance or
                self.event_phase_gate > .5 or
                self.min_same_event_interval_fraction < 0 or
                self.min_any_event_interval_s < 0):
            raise ValueError("Invalid phase-controller configuration")


@dataclass(frozen=True)
class PhaseStep:
    theta_before: float
    theta_after: float
    nominal_step: float
    rl_step: float
    requested_event_step: float
    effective_event_step: float
    actual_step: float
    phase_rate_hz: float  # cycles per second, for time-warped reference velocities
    correction_remaining: float
    correction_active: bool


class PhaseController:
    def __init__(self, config: PhaseConfig = PhaseConfig()):
        self.config = config
        self.duration = None
        self.events = None
        self.theta = 0.0
        self.theta_nominal = 0.0
        self.remaining_error = 0.0
        self.active = False
        self.phase_rate_hz = None
        self.last_event = None
        self.event_accept_count = 0
        self.event_reject_count = 0
        self._last_valid_event_time = {}
        self._last_any_event_time = -np.inf

    @property
    def phase(self) -> float:
        return self.theta % 1.0

    @property
    def phase_drift(self) -> float:
        return self.theta - self.theta_nominal

    def reset(self, state: PhaseReset, duration: float, events: ReferenceEvents) -> None:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Reference duration must be finite and positive")
        self.duration = float(duration)
        self.events = events
        self.theta = float(state.theta)
        self.theta_nominal = float(state.theta_nominal)
        self.remaining_error = float(state.remaining_event_correction)
        self.active = bool(state.event_correction_active)
        self.phase_rate_hz = 1.0 / self.duration
        self.last_event = None
        self.event_accept_count = 0
        self.event_reject_count = 0
        self._last_valid_event_time = {}
        self._last_any_event_time = -np.inf

    def observe_event(self, name: str, *, time_s: float | None = None) -> tuple[bool, str, float]:
        """Register a detected event at the CURRENT (post-physics) phase.

        Returns (accepted, reason, signed phase error). A new accepted event
        replaces a previous pending correction rather than summing incompatible
        timing measurements. Event availability and provenance are checked here.
        """
        if self.duration is None:
            raise RuntimeError("Phase clock has not been reset")
        if name not in self.events.phases:
            self.event_reject_count += 1
            return False, "missing_reference_event", 0.0
        if name.endswith("toe_off") and not self.config.use_assumed_toe_off:
            source = self.events.sources.get(name, "unspecified").lower()
            if "assumed" in source or "inferred" in source or source == "unspecified":
                self.event_reject_count += 1
                return False, "unverified_toe_off_phase", 0.0
        if time_s is not None:
            if not np.isfinite(time_s) or time_s < 0:
                raise ValueError("Event time must be nonnegative and finite")
            last_same = self._last_valid_event_time.get(name, -np.inf)
            if time_s - last_same < self.config.min_same_event_interval_fraction * self.duration:
                self.event_reject_count += 1
                return False, "repeated_event_too_soon", 0.0
            if time_s - self._last_any_event_time < self.config.min_any_event_interval_s:
                self.event_reject_count += 1
                return False, "events_too_close", 0.0
        error = wrap_half_cycle(self.events.phases[name] - self.phase)
        if abs(error) > self.config.event_phase_gate:
            self.event_reject_count += 1
            return False, "implausible_phase_error", error
        self.last_event = name
        if time_s is not None:
            # Count a correctly timed physical event even if it does not
            # require a phase correction, so contact chatter cannot re-arm it.
            self._last_valid_event_time[name] = time_s
            self._last_any_event_time = time_s
        if abs(error) <= self.config.activation_tolerance:
            # A valid new contact event says the clocks already agree. Do not
            # cancel an existing correction solely due to the activation deadband.
            return False, "within_activation_tolerance", error
        self.remaining_error = error
        self.active = True
        self.event_accept_count += 1
        return True, "correction_started", error

    def advance(self, dt: float, rl_phase_action: float) -> PhaseStep:
        """Advance once for *one physics step*, not one PPO action interval."""
        if self.duration is None:
            raise RuntimeError("Phase clock has not been reset")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be positive")
        action = float(rl_phase_action)
        if not np.isfinite(action) or abs(action) > 1 + 1e-8:
            raise ValueError("Phase action must be in [-1,1]")
        action = float(np.clip(action, -1, 1))
        delta_nominal = dt / self.duration
        delta_rl = self.config.kappa_max * delta_nominal * action
        delta_event = (float(np.clip(self.remaining_error,
                                     -self.config.event_gain * delta_nominal,
                                     self.config.event_gain * delta_nominal))
                       if self.active else 0.0)
        lower = self.config.min_rate_ratio * delta_nominal
        upper = self.config.max_rate_ratio * delta_nominal
        d_normal = float(np.clip(delta_nominal + delta_rl, lower, upper))
        d_actual = float(np.clip(delta_nominal + delta_rl + delta_event, lower, upper))
        effective_event = d_actual - d_normal
        old_theta = self.theta
        self.theta += d_actual
        self.theta_nominal += delta_nominal
        if self.active:
            # IMPORTANT: subtract the *applied* correction after final clipping.
            self.remaining_error -= effective_event
            if abs(self.remaining_error) <= self.config.completion_tolerance:
                self.remaining_error = 0.0
                self.active = False
        self.phase_rate_hz = d_actual / dt
        return PhaseStep(old_theta, self.theta, delta_nominal, delta_rl,
                         delta_event, effective_event, d_actual,
                         self.phase_rate_hz, self.remaining_error, self.active)
