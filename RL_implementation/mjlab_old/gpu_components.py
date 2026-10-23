"""GPU-only gait reference, muscle mapping and event-corrected phase tracking.

Torch tensors stay on the simulation device. One row = one mjlab environment.
The original reference files are gait-cycle normalized; the phase just looks up
which sample is relevant to each walker. No time-rescaling of the dataset occurs.
"""
from __future__ import annotations

import numpy as np
import torch

from gait_phase import PhaseConfig
from gait_reference import CHANNELS, GaitReference
from muscle_mapping import RIGHT_MAPPING


class GPUReference:
    """Periodic lookup in an already cycle-normalized reference: phase -> values."""

    def __init__(self, reference: GaitReference, device: torch.device | str):
        self.device = torch.device(device)
        self.joint_names = reference.joint_names
        self.phase = torch.as_tensor(reference.phase, device=self.device, dtype=torch.float32)
        self.activations = torch.as_tensor(reference.muscle_activations, device=self.device, dtype=torch.float32)
        self.q = (None if reference.joint_angles is None else
                  torch.as_tensor(reference.joint_angles, device=self.device, dtype=torch.float32))
        self.dq = (None if reference.joint_velocities is None else
                   torch.as_tensor(reference.joint_velocities, device=self.device, dtype=torch.float32))
        # The first and last samples may not be at 0/1. Extend circularly.
        self.ext_phase = torch.cat((self.phase[-1:] - 1.0, self.phase, self.phase[:1] + 1.0))

    def _lookup(self, query: torch.Tensor, table: torch.Tensor | None) -> torch.Tensor | None:
        if table is None:
            return None
        if query.ndim != 1 or query.device != self.device:
            raise ValueError("Query must have shape [num_envs] on the reference device")
        x = torch.remainder(query, 1.0)
        idx = torch.searchsorted(self.ext_phase.contiguous(), x.contiguous(), right=True)
        idx = idx.clamp(1, self.ext_phase.numel() - 1)
        i0, i1 = idx - 1, idx
        x0, x1 = self.ext_phase[i0], self.ext_phase[i1]
        ext_y = torch.cat((table[-1:], table, table[:1]), dim=0)
        w = ((x - x0) / (x1 - x0)).unsqueeze(-1)
        return ext_y[i0] + (ext_y[i1] - ext_y[i0]) * w

    def at_phase(self, phase: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return {
            "muscles_15": self._lookup(phase, self.activations),
            "joint_angles": self._lookup(phase, self.q),
            "joint_velocities": self._lookup(phase, self.dq),
        }


class GPUMuscleMapper:
    """Complete provisional right/left mapping: feedforward excitation [B,80].

    Right side uses phase phi; left side uses phi+0.5 cycles (a *provisional*
    symmetric-gait assumption). We never take an RL reward against this drive.
    Unknown actuator names raise at startup to avoid silently zeroed muscles.
    """

    def __init__(self, actuator_names: tuple[str, ...], device: torch.device | str):
        from muscle_mapping import mapping_source

        self.device = torch.device(device)
        self.actuator_names = tuple(actuator_names)
        bases = [name.rsplit('/', 1)[-1] for name in actuator_names]
        if len(set(bases)) != len(bases):
            raise ValueError('Duplicate actuator names after namespace removal')
        right_w = np.zeros((len(actuator_names), len(CHANNELS)), dtype=np.float32)
        left_w = np.zeros_like(right_w)
        self.provisional = []
        self.unresolved = []  # kept for backward-compatible reporting; now empty.
        for row, (full, base) in enumerate(zip(actuator_names, bases)):
            channels, side, provisional = mapping_source(base)
            matrix = right_w if side == 'r' else left_w
            for ch in channels:
                matrix[row, CHANNELS.index(ch)] = 1.0 / len(channels)
            if provisional:
                self.provisional.append(full)
        self.right_weights = torch.as_tensor(right_w, device=self.device)
        self.left_weights = torch.as_tensor(left_w, device=self.device)
        self.valid = torch.ones(len(actuator_names), dtype=torch.bool, device=self.device)

    def expand(self, profiles_right: torch.Tensor,
               profiles_left: torch.Tensor | None = None) -> torch.Tensor:
        if profiles_right.ndim != 2 or profiles_right.shape[1] != len(CHANNELS):
            raise ValueError('Expected [num_envs, 15] for right profile')
        if profiles_right.device != self.device:
            raise ValueError('Reference profiles must be on mapper device')
        if profiles_left is None:
            # Do not silently use the same phase for both legs.
            if bool(torch.any(self.left_weights).item()):
                raise ValueError('Bilateral mapping requires the left-leg phase-shifted profile')
            profiles_left = torch.zeros_like(profiles_right)
        if profiles_left.shape != profiles_right.shape or profiles_left.device != self.device:
            raise ValueError('Left/right profile shapes and devices must match')
        return profiles_right @ self.right_weights.T + profiles_left @ self.left_weights.T


class GPUPhaseEstimator:
    """Event-corrected phase, stride timing, and confidence for N walkers.

    Phase is in cycles [0,1), not radians. Only right-foot events currently.
    Missing or invalid stride observations never set cadence estimates.
    """

    def __init__(self, cfg: PhaseConfig, num_envs: int, device: torch.device | str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.phase = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.elapsed = self.phase.clone()
        self.smoothed_stride = torch.full_like(self.phase, cfg.reference_stride_s)
        self.last_stride = torch.full_like(self.phase, float('nan'))
        self.last_hs_time = torch.full_like(self.phase, float('nan'))
        self.count = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.confidence = self.phase.clone()
        self.previous_contact = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.initial_phase = 0.0
        self.reset()

    def _check_mask(self, mask: torch.Tensor, name: str) -> torch.Tensor:
        if mask.shape != (self.num_envs,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError(f"{name} must be bool tensor [num_envs] on {self.device}")
        return mask

    def reset(self, mask: torch.Tensor | None = None,
              initial_contact: torch.Tensor | None = None,
              initial_phase: float = 0.0) -> None:
        """Reset all or selected walkers. The first contact is a baseline, NOT HS."""
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        mask = self._check_mask(mask, "reset mask")
        if initial_contact is None:
            initial_contact = torch.zeros_like(mask)
        self._check_mask(initial_contact, "initial_contact")
        self.phase[mask] = initial_phase % 1.0
        self.elapsed[mask] = 0.0
        self.smoothed_stride[mask] = self.cfg.reference_stride_s
        self.last_stride[mask] = float('nan')
        self.last_hs_time[mask] = float('nan')
        self.count[mask] = 0
        self.confidence[mask] = 0.0
        self.previous_contact = torch.where(mask, initial_contact, self.previous_contact)

    def update(self, dt: float, contact: torch.Tensor,
               done: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-physics sensor update; done envs are already auto-reset by mjlab."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        contact = self._check_mask(contact, "contact")
        if done is None:
            done = torch.zeros_like(contact)
        done = self._check_mask(done, "done")

        heel_strike = contact & ~self.previous_contact & ~done
        toe_off = ~contact & self.previous_contact & ~done
        frequency = ((1 - self.confidence) / self.cfg.reference_stride_s
                     + self.confidence / self.smoothed_stride)
        self.phase = torch.remainder(self.phase + frequency * dt, 1.0)
        self.elapsed += dt

        previous_hs_exists = torch.isfinite(self.last_hs_time)
        stride = self.elapsed - self.last_hs_time
        within = ((stride >= self.cfg.min_stride_s) &
                  (stride <= self.cfg.max_stride_s))
        valid = heel_strike & previous_hs_exists & within
        invalid = heel_strike & previous_hs_exists & ~within
        updated = ((1.0 - self.cfg.alpha) * self.smoothed_stride +
                   self.cfg.alpha * stride)
        self.smoothed_stride = torch.where(valid, updated, self.smoothed_stride)
        self.last_stride = torch.where(valid, stride, self.last_stride)
        self.last_stride = torch.where(invalid, float('nan'), self.last_stride)
        self.count = torch.where(valid, self.count + 1, self.count)
        self.count = torch.where(invalid, 0, self.count)
        self.last_hs_time = torch.where(heel_strike, self.elapsed, self.last_hs_time)

        # Confidence schedule from original skeleton; provisional thresholds.
        c_strides = torch.where(self.count >= 4, 0.5,
                    torch.where(self.count < 4, self.count * 0.1))
        dt_err = torch.abs(self.smoothed_stride - self.cfg.reference_stride_s)
        c_timing = torch.where(dt_err > 0.4, 0.0,
                   torch.where(dt_err > 0.3, 0.2,
                   torch.where(dt_err > 0.15, 0.3, 0.5)))
        self.confidence = torch.where(valid, torch.clamp(c_strides + c_timing, 0, 1),
                            torch.where(invalid, 0.0, self.confidence))

        # Shortest signed distance modulo one cycle, with HS target=0.
        d_hs = torch.remainder(-self.phase + 0.5, 1.0) - 0.5
        d_to = torch.remainder(self.cfg.toe_off_phase - self.phase + 0.5, 1.0) - 0.5
        phase = torch.where(heel_strike, self.phase + self.cfg.k_hs * d_hs, self.phase)
        phase = torch.where(toe_off, phase + self.cfg.k_to * d_to, phase)
        self.phase = torch.remainder(phase, 1.0)
        self.previous_contact = contact.clone()

        # mjlab resets completed walkers internally; their reported sensors now
        # correspond to a fresh episode. Do not create false HS/TO transitions.
        self.reset(mask=done, initial_contact=contact, initial_phase=self.initial_phase)
        return heel_strike, toe_off
