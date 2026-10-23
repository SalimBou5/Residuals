"""Event-corrected, normalized gait phase estimator (one reference leg).

Phase is in [0, 1), and frequency is in cycles/second, NOT radians/second.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhaseConfig:
    reference_stride_s: float
    toe_off_phase: float   # From the reference gait, not an assumed universal constant.
    alpha: float = 0.3    # Smoothed stride-duration update; initial working value.
    k_hs: float = 0.5     # Partial heel-strike correction; tune later.
    k_to: float = 0.5     # Partial toe-off correction; tune later.
    min_stride_s: float = 0.4  # Sanity bounds: CONFIGURATION PLACEHOLDERS.
    max_stride_s: float = 2.5

    def __post_init__(self):
        if self.reference_stride_s <= 0:
            raise ValueError("reference_stride_s must be positive")
        if not 0.0 <= self.toe_off_phase < 1.0:
            raise ValueError("toe_off_phase must be in [0, 1)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not (0 <= self.k_hs <= 1 and 0 <= self.k_to <= 1):
            raise ValueError("Event-correction gains must be in [0, 1]")
        if not (0 < self.min_stride_s < self.max_stride_s):
            raise ValueError("Invalid stride-time sanity bounds")


def circular_error(target: float, current: float) -> float:
    """Shortest signed phase difference measured in gait cycles."""
    return (target - current + 0.5) % 1.0 - 0.5


class GaitPhaseEstimator:
    def __init__(self, config: PhaseConfig):
        self.cfg = config
        self.reset()

    def reset(self, initial_phase: float = 0.0):
        self.phase = initial_phase % 1.0
        self.elapsed_s = 0.0
        self.last_hs_s = None
        self.smoothed_stride_s = self.cfg.reference_stride_s
        self.valid_strides = 0
        self.confidence = 0.0
        self.last_stride_s = None

    @property
    def frequency_hz(self) -> float:
        f_ref = 1.0 / self.cfg.reference_stride_s
        f_est = 1.0 / self.smoothed_stride_s
        return (1.0 - self.confidence) * f_ref + self.confidence * f_est

    def _update_confidence(self):
        n = self.valid_strides
        c_strides = 0.0 if n < 2 else (0.2 if n == 2 else (0.3 if n == 3 else 0.5))

        e_t = abs(self.smoothed_stride_s - self.cfg.reference_stride_s)
        if e_t > 0.5:
            c_timing = 0.0
        elif e_t > 0.3:
            c_timing = 0.2
        elif e_t > 0.15:
            c_timing = 0.3
        else:
            c_timing = 0.5
        self.confidence = min(1.0, c_strides + c_timing)

    def update(self, dt: float, heel_strike: bool = False, toe_off: bool = False) -> float:
        """Advance the phase then correct it with newly observed contact events.

        heel_strike and toe_off should be *event edges*, not continuous contact.
        Both events refer to the same (reference) leg.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if heel_strike and toe_off:
            raise ValueError("Heel strike and toe off cannot be simultaneous")

        self.phase = (self.phase + self.frequency_hz * dt) % 1.0
        self.elapsed_s += dt

        if heel_strike:
            if self.last_hs_s is not None:
                stride_s = self.elapsed_s - self.last_hs_s
                if self.cfg.min_stride_s <= stride_s <= self.cfg.max_stride_s:
                    self.last_stride_s = stride_s
                    a = self.cfg.alpha
                    self.smoothed_stride_s = ((1.0 - a) * self.smoothed_stride_s
                                              + a * stride_s)
                    self.valid_strides += 1
                else:
                    # Implausible event timing: do not train the cadence estimate on it.
                    self.valid_strides = 0
                    self.last_stride_s = None
                self._update_confidence()
            self.last_hs_s = self.elapsed_s
            self.phase = (self.phase
                          + self.cfg.k_hs * circular_error(0.0, self.phase)) % 1.0

        elif toe_off:
            self.phase = (self.phase + self.cfg.k_to * circular_error(
                self.cfg.toe_off_phase, self.phase)) % 1.0

        return self.phase


class FootContactEvents:
    """Simple *placeholder* contact-edge detection for a chosen leg.

    The model-specific extraction of the boolean contact state is left to the user.
    """

    def __init__(self):
        self.previous_contact = None

    def reset(self):
        self.previous_contact = None

    def update(self, in_contact: bool) -> tuple[bool, bool]:
        contact = bool(in_contact)
        if self.previous_contact is None:
            self.previous_contact = contact
            return False, False  # Initial state is not a detected event.
        hs = contact and not self.previous_contact
        to = not contact and self.previous_contact
        self.previous_contact = contact
        return hs, to
