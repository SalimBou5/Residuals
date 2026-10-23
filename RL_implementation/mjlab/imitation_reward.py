"""Gait-phase aligned *KINEMATIC* imitation rewards for MyoSuite 3 / mjlab.

Muscle activation profiles provide the base controller excitation, NOT an
imitation target. There is deliberately no muscle-tracking reward in this file.
Module is algorithm-neutral; reward computations stay in PyTorch on device.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ImitationRewardConfig:
    # Provisional weights/gains for software tests; they need validation.
    q_weight: float = 1.0
    dq_weight: float = 0.1
    q_gain: float = 2.0
    dq_gain: float = 0.1

    def __post_init__(self):
        if any(value < 0 for value in vars(self).values()):
            raise ValueError('Imitation weights and gains must be nonnegative')


class ImitationReward:
    """Compare named MyoLeg joint states to phase-normalized kinematics only."""

    def __init__(self, config: ImitationRewardConfig, reference, mapper, sim):
        self.cfg = config
        self.reference = reference
        self.sim = sim
        if (reference.joint_angles is not None or reference.joint_velocities is not None) and not reference.joint_names:
            raise ValueError('Reference must have explicit joint_names')
        if config.q_weight and reference.joint_angles is None:
            raise ValueError('Joint-angle imitation needs joint_angles')
        if config.dq_weight and reference.joint_velocities is None:
            raise ValueError('Joint-velocity imitation needs joint_velocities')
        self.joint_names = reference.joint_names

    def compute(self, phase: torch.Tensor, targets: dict | None = None):
        """Returns one imitation reward and component diagnostics per walker."""
        if targets is None:
            targets = self.reference.at_phase(phase)
        rewards = torch.zeros_like(phase)
        terms = {}
        if targets['joint_angles'] is not None or targets['joint_velocities'] is not None:
            sim_q, sim_dq = self.sim.get_joint_state(self.joint_names)
            if targets['joint_angles'] is not None:
                q_mse = ((sim_q - targets['joint_angles']) ** 2).mean(dim=-1)
                r_q = torch.exp(-self.cfg.q_gain * q_mse)
                rewards = rewards + self.cfg.q_weight * r_q
                terms.update(q_mse=q_mse, q_reward=r_q)
            if targets['joint_velocities'] is not None:
                dq_mse = ((sim_dq - targets['joint_velocities']) ** 2).mean(dim=-1)
                r_dq = torch.exp(-self.cfg.dq_gain * dq_mse)
                rewards = rewards + self.cfg.dq_weight * r_dq
                terms.update(dq_mse=dq_mse, dq_reward=r_dq)
        terms['total_imitation_reward'] = rewards
        return rewards, terms
