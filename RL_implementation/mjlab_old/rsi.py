"""RSI staging interface, deliberately *without* unverified MyoLeg resets.

Stage 1: 'standard' resets use MyoSuite's native reset. Stage 2/3 samplers are
prepared, but applying qpos/qvel/act to mjlab's manager reset must be written
and tested before enabling them. We never pretend that reference kinematics
alone specify a dynamically consistent muscle state.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RSIConfig:
    mode: str = 'standard'  # standard | kinematic | physical
    initial_phase: float = 0.0  # must be calibrated to default MyoLeg keyframe

    def __post_init__(self):
        if self.mode not in ('standard', 'kinematic', 'physical'):
            raise ValueError('Unknown RSI mode')
        if not 0 <= self.initial_phase < 1:
            raise ValueError('initial_phase must be in [0, 1)')


class RSISampler:
    def __init__(self, config: RSIConfig, reference, device):
        self.config = config
        self.reference = reference
        self.device = torch.device(device)

    def sample_kinematic(self, n_envs: int):
        if self.reference.q is None or self.reference.dq is None:
            raise ValueError('Kinematic RSI requires joint angles AND joint velocities')
        if not self.reference.joint_names:
            raise ValueError('Kinematic RSI requires explicit joint_names')
        phase = torch.rand(n_envs, device=self.device)
        targets = self.reference.at_phase(phase)
        return {'phase': phase, 'joint_names': self.reference.joint_names,
                'q': targets['joint_angles'], 'dq': targets['joint_velocities']}


class PhysicalStateBuffer:
    """Prototype candidate buffer: stores complete verified states and their phases.

    Only add states drawn from actual successful MyoLeg rollouts. Physical RSI
    also needs a backend-specific restore operation; this class does not do it.
    """

    def __init__(self, capacity: int = 50000):
        if capacity < 1:
            raise ValueError('capacity must be >= 1')
        self.capacity = capacity
        self._rows: list[dict[str, torch.Tensor]] = []

    def add_batch(self, qpos, qvel, act, phase, approved):
        if approved.dtype != torch.bool or phase.ndim != 1:
            raise ValueError('Need approved[B] bool and phase[B]')
        n = len(phase)
        if any(x.shape[0] != n for x in (qpos, qvel, act, approved)):
            raise ValueError('Incompatible state batch dimensions')
        # Explicit approval is required to avoid storing falling/corrupt states.
        for i in torch.nonzero(approved).flatten().tolist():
            self._rows.append({
                'qpos': qpos[i].detach().cpu().clone(),
                'qvel': qvel[i].detach().cpu().clone(),
                'act': act[i].detach().cpu().clone(),
                'phase': phase[i].detach().cpu().clone(),
            })
        if len(self._rows) > self.capacity:
            del self._rows[:len(self._rows) - self.capacity]

    def sample(self, n: int, device='cpu'):
        if not self._rows:
            raise RuntimeError('No approved states in physical RSI buffer')
        indices = torch.randint(len(self._rows), (n,))
        return {key: torch.stack([self._rows[i][key] for i in indices.tolist()]).to(device)
                for key in ('qpos', 'qvel', 'act', 'phase')}

    def __len__(self):
        return len(self._rows)
