"""Step 3: one 81-D policy action, recomputed MEP excitation at each physics step."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from RSI.rsi import MyoLegMuscleBinding


@dataclass(frozen=True)
class ActionConfig:
    # This is a CALIBRATION parameter, not a scientifically validated value.
    residual_limit: float | np.ndarray = 0.25  # scalar OR 80 actuator-ordered limits
    enable_unmapped: bool = False

    def __post_init__(self):
        limits = np.asarray(self.residual_limit, dtype=float)
        if (limits.shape not in [(), (80,)] or not np.all(np.isfinite(limits))
                or np.any(limits < 0)):
            raise ValueError("Residual limits must be a nonnegative scalar or 80-vector")


class ResidualActionController:
    """Expand compact policy actions into MyoLeg's full 80-actuator command.

    The last action is always phase-rate modulation. This mapping is fixed
    at construction time: do not toggle enable_unmapped in a running policy.
    """
    def __init__(self, binding: MyoLegMuscleBinding, config: ActionConfig):
        self.binding = binding
        self.config = config
        self.residual_limits = np.broadcast_to(
            np.asarray(config.residual_limit, dtype=float), (80,)).copy()

        # Policy muscle coordinates are ordered by actual actuator ID; the
        # unmapped muscles have NO policy coordinate when disabled.
        self.policy_to_actuator_ids = (np.arange(80, dtype=int) if config.enable_unmapped
                                       else np.flatnonzero(binding.mapped_mask))
        self.muscle_action_dim = int(self.policy_to_actuator_ids.size)
        self.action_dim = self.muscle_action_dim + 1
        self.policy_muscle_names = tuple(binding.names_by_id[i]
                                         for i in self.policy_to_actuator_ids)
        self.action = np.zeros(self.action_dim, dtype=float)
        # Expanded muscle action, in actuator-ID order, for logging/control.
        self.actuator_action = np.zeros(80, dtype=float)
        self.last_baseline = np.zeros(80, dtype=float)
        self.last_excitation = np.zeros(80, dtype=float)

    def set_policy_action(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=float)
        if (a.shape != (self.action_dim,) or not np.all(np.isfinite(a))
                or np.any(np.abs(a) > 1.0 + 1e-8)):
            raise ValueError(f"Policy action must have shape ({self.action_dim},) "
                             "with entries in [-1,1]")
        self.action[:] = np.clip(a, -1, 1)
        self.actuator_action[:] = 0.0
        self.actuator_action[self.policy_to_actuator_ids] = self.action[:-1]

    @property
    def phase_action(self) -> float:
        return float(self.action[-1])

    def excitation_at(self, theta: float) -> np.ndarray:
        # Interpolate MEP once per physics step; residual remains held between
        # RL decisions. data.ctrl remains an 80-D actuator-ordered vector.
        self.last_baseline[:] = self.binding.baseline(theta)
        self.last_excitation[:] = self.last_baseline
        mapped = self.binding.mapped_mask
        unmapped = self.binding.unmapped_mask
        self.last_excitation[mapped] = np.clip(
            self.last_baseline[mapped] +
            self.residual_limits[mapped] * self.actuator_action[mapped], 0., 1.)
        if self.config.enable_unmapped:
            self.last_excitation[unmapped] = (self.actuator_action[unmapped] + 1.) / 2.
        else:
            self.last_excitation[unmapped] = 0.0
        return self.last_excitation

    @property
    def applied_residual(self) -> np.ndarray:
        delta = self.last_excitation - self.last_baseline
        # Directly controlled unmapped muscles are not MEP residuals.
        delta[self.binding.unmapped_mask] = 0.0
        return delta
