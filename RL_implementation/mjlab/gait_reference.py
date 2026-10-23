"""Gait-cycle-normalized reference tables and ONE periodic lookup algorithm.

Preserves the user's NPZ schema, channel reordering, joint-name validation,
and scalar at_phase() result. The same PyTorch interpolation also handles
parallel [N] phase tensors without host-device transfers in the hot path.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

CHANNELS = (
    'TA', 'SOL', 'Per', 'VastLat', 'VastMed', 'RFem', 'Sar', 'Add',
    'GlutMed', 'TFL', 'GastLat', 'GastMed', 'BFem', 'Semi', 'GlutMax',
)


@dataclass
class GaitReference:
    phase: np.ndarray
    muscle_activations: np.ndarray
    joint_angles: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
    joint_names: tuple[str, ...] | None = None

    def __post_init__(self):
        self.phase = np.asarray(self.phase, dtype=float)
        self.muscle_activations = np.asarray(self.muscle_activations, dtype=float)
        if self.phase.ndim != 1 or len(self.phase) < 2:
            raise ValueError('phase must be a 1D array with at least two samples')
        if not np.all(np.diff(self.phase) > 0) or self.phase[0] < 0 or self.phase[-1] >= 1:
            raise ValueError('phase must be strictly increasing in [0, 1)')
        if self.muscle_activations.shape != (len(self.phase), len(CHANNELS)):
            raise ValueError('muscle_activations must have shape (N, 15)')
        if not (np.isfinite(self.phase).all() and np.isfinite(self.muscle_activations).all()):
            raise ValueError('Nonfinite phase/activation values')
        if self.joint_names is not None:
            self.joint_names = tuple(str(n) for n in self.joint_names)
            if len(self.joint_names) != len(set(self.joint_names)):
                raise ValueError('Duplicate joint_names')
        for name in ('joint_angles', 'joint_velocities'):
            array = getattr(self, name)
            if array is not None:
                array = np.asarray(array, dtype=float)
                if array.ndim != 2 or array.shape[0] != len(self.phase):
                    raise ValueError(f'{name} must have shape (N, D)')
                if not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite {name}')
                if self.joint_names is not None and array.shape[1] != len(self.joint_names):
                    raise ValueError(f'{name} columns must match joint_names')
                setattr(self, name, array)
        self._torch_device = None
        self._extended_phase = None
        self._extended_tables = None

    @classmethod
    def from_npz(cls, path: str | Path) -> 'GaitReference':
        with np.load(path, allow_pickle=False) as d:
            if 'muscle_names' not in d:
                raise ValueError("NPZ needs 'muscle_names' to prevent wrong channel ordering")
            actual = [str(x) for x in d['muscle_names']]
            if len(actual) != 15 or set(actual) != set(CHANNELS):
                raise ValueError(f'Expected exactly 15 channels {CHANNELS}; found {actual}')
            indices = [actual.index(name) for name in CHANNELS]
            return cls(
                phase=d['phase'],
                muscle_activations=d['muscle_activations'][:, indices],
                joint_angles=d['joint_angles'] if 'joint_angles' in d else None,
                joint_velocities=d['joint_velocities'] if 'joint_velocities' in d else None,
                joint_names=tuple(str(x) for x in d['joint_names']) if 'joint_names' in d else None,
            )

    def to(self, device):
        """Cache periodically extended tables on the requested CPU/GPU device."""
        device = torch.device(device)
        if self._torch_device == device and self._extended_phase is not None:
            return self
        p = torch.as_tensor(self.phase, dtype=torch.float32, device=device)
        self._extended_phase = torch.cat((p[-1:] - 1.0, p, p[:1] + 1.0))
        self._extended_tables = {}
        for key, array in (
            ('muscles_15', self.muscle_activations),
            ('joint_angles', self.joint_angles),
            ('joint_velocities', self.joint_velocities),
        ):
            if array is None:
                self._extended_tables[key] = None
                continue
            values = torch.as_tensor(array, dtype=torch.float32, device=device)
            self._extended_tables[key] = torch.cat((values[-1:], values, values[:1]), dim=0)
        self._torch_device = device
        return self

    def at_phase(self, phase: float | torch.Tensor) -> dict:
        """
        Interpolate the reference at the given phase.
        Scalar input retains the user's original ``muscles_15`` dictionary API.
        A vector of phases returns a [B,15] tensor and [B,D] joint tensors.
        The reference is already normalized; this interpolates, not resamples.
        """
        scalar = not isinstance(phase, torch.Tensor) or phase.ndim == 0
        if not isinstance(phase, torch.Tensor):
            phases = torch.as_tensor(phase, dtype=torch.float32,
                                     device=self._torch_device or torch.device('cpu'))
        else:
            phases = phase
        if phases.ndim > 1:
            raise ValueError('phase must be scalar or a 1D tensor [B]')
        if self._torch_device != phases.device or self._extended_phase is None:
            self.to(phases.device)

        x = torch.remainder(phases.float().reshape(-1), 1.0).contiguous()
        xp = self._extended_phase
        hi = torch.searchsorted(xp, x, right=True).clamp(1, len(xp)-1)
        lo = hi-1
        w = ((x-xp[lo]) / (xp[hi]-xp[lo])).unsqueeze(-1)
        outputs = {
            key: None if table is None else table[lo] + w * (table[hi]-table[lo])
            for key, table in self._extended_tables.items()
        }
        if not scalar:
            return outputs
        activation = outputs['muscles_15'][0].detach().cpu().numpy()
        q = None if outputs['joint_angles'] is None else outputs['joint_angles'][0].detach().cpu().numpy()
        dq = None if outputs['joint_velocities'] is None else outputs['joint_velocities'][0].detach().cpu().numpy()
        return {'muscles_15': dict(zip(CHANNELS, activation.tolist())),
                'joint_angles': q, 'joint_velocities': dq}
