"""Phase-indexed 15-channel reference gait data.

Accepted input: NPZ with arrays 'phase', 'muscle_activations', 'muscle_names';
optional 'joint_angles', 'joint_velocities' (all indexed by the same phase rows).
This defines a template, not an assumption about the user's existing file layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

CHANNELS = (
    "TA", "SOL", "Per", "VastLat", "VastMed", "RFem", "Sar", "Add",
    "GlutMed", "TFL", "GastLat", "GastMed", "BFem", "Semi", "GlutMax",
)


def _periodic_interp(x, phase, y):
    """Linear interpolation across phase=0/1 as well as within the stride."""
    p = np.asarray(phase)
    x_ext = np.concatenate(([p[-1] - 1.0], p, [p[0] + 1.0]))
    y_ext = np.concatenate((y[-1:], y, y[:1]), axis=0)
    at = float(x) % 1.0
    if y.ndim == 1:
        return float(np.interp(at, x_ext, y_ext))
    return np.array([np.interp(at, x_ext, y_ext[:, j])
                     for j in range(y.shape[1])], dtype=float)


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
            raise ValueError("phase must be a 1D array with at least two samples")
        if not np.all(np.diff(self.phase) > 0) or self.phase[0] < 0 or self.phase[-1] >= 1:
            raise ValueError("phase must be strictly increasing in [0, 1)")
        if self.muscle_activations.shape != (len(self.phase), len(CHANNELS)):
            raise ValueError("muscle_activations must have shape (N, 15)")
        if not (np.isfinite(self.phase).all() and np.isfinite(self.muscle_activations).all()):
            raise ValueError("Nonfinite phase/activation values")
        if self.joint_names is not None:
            self.joint_names = tuple(str(n) for n in self.joint_names)
            if len(self.joint_names) != len(set(self.joint_names)):
                raise ValueError("Duplicate joint_names")
        for name in ("joint_angles", "joint_velocities"):
            array = getattr(self, name)
            if array is not None:
                array = np.asarray(array, dtype=float)
                if array.ndim != 2 or array.shape[0] != len(self.phase):
                    raise ValueError(f"{name} must have shape (N, D)")
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite {name}")
                if self.joint_names is not None and array.shape[1] != len(self.joint_names):
                    raise ValueError(f"{name} columns must match joint_names")
                setattr(self, name, array)

    @classmethod
    def from_npz(cls, path: str | Path) -> "GaitReference":
        with np.load(path, allow_pickle=False) as d:
            if "muscle_names" not in d:
                raise ValueError("NPZ needs 'muscle_names' to prevent wrong channel ordering")
            actual = [str(x) for x in d["muscle_names"]]
            if len(actual) != 15 or set(actual) != set(CHANNELS):
                raise ValueError(f"Expected exactly 15 channels {CHANNELS}; found {actual}")
            indices = [actual.index(name) for name in CHANNELS]
            return cls(
                phase=d["phase"],
                muscle_activations=d["muscle_activations"][:, indices],
                joint_angles=d["joint_angles"] if "joint_angles" in d else None,
                joint_velocities=d["joint_velocities"] if "joint_velocities" in d else None,
                joint_names=tuple(str(x) for x in d["joint_names"]) if "joint_names" in d else None,
            )

    def at_phase(self, phase: float) -> dict:
        activations = _periodic_interp(phase, self.phase, self.muscle_activations)
        q = None if self.joint_angles is None else _periodic_interp(
            phase, self.phase, self.joint_angles)
        dq = None if self.joint_velocities is None else _periodic_interp(
            phase, self.phase, self.joint_velocities)
        return {
            "muscles_15": dict(zip(CHANNELS, activations.tolist())),
            "joint_angles": q,
            "joint_velocities": dq,
        }
