"""Provisional 15-channel EMG-derived feedforward drive -> MyoLeg 80 muscles.

This is a *command mapping*, NOT a target for muscle imitation.
The PDF supplies a subset of right-leg assignments. Missing assignments are
explicit temporary anatomical approximations; every left-leg muscle shares
its right-leg mapping but samples the profile half a gait cycle later.
No signal is divided among muscles in a group: each muscle receives its mapped
channel value or the specified mean of channels.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from gait_reference import CHANNELS

# Sources are names among the 15 González-Vargas-style channels.
# Entries labelled TEMP were not resolved by the supplied PDF. All 40 right
# muscles are assigned, so the complete bilateral model can receive a baseline.
RIGHT_MAPPING = {
    "addbrev_r": "Add",             # TEMP: measured adductor group
    "addlong_r": "Add",             # TEMP
    "addmagDist_r": "Add",          # TEMP
    "addmagIsch_r": "Add",          # TEMP
    "addmagMid_r": "Add",           # TEMP
    "addmagProx_r": "Add",          # TEMP
    "bflh_r": "BFem",
    "bfsh_r": "BFem",
    "edl_r": "TA",                # TEMP: toe extensor / dorsiflexor
    "ehl_r": "TA",
    "fdl_r": "SOL",               # TEMP: toe flexor / plantarflexor
    "fhl_r": "SOL",               # TEMP
    "gaslat_r": "GastLat",
    "gasmed_r": "GastMed",
    "glmax1_r": "GlutMax",
    "glmax2_r": "GlutMax",
    "glmax3_r": "GlutMax",
    "glmed1_r": "GlutMed",
    "glmed2_r": "GlutMed",
    "glmed3_r": "GlutMed",
    "glmin1_r": ("GlutMed", "GlutMax"),
    "glmin2_r": ("GlutMed", "GlutMax"),
    "glmin3_r": ("GlutMed", "GlutMax"),
    "grac_r": "Add",              # TEMP: adductor group
    "iliacus_r": "RFem",           # TEMP: hip flexor approximation
    "perbrev_r": "Per",
    "perlong_r": "Per",
    "piri_r": "GlutMax",          # TEMP: posterior hip rotator approximation
    "psoas_r": "RFem",             # TEMP: hip flexor approximation
    "recfem_r": "RFem",
    "sart_r": "Sar",
    "semimem_r": "Semi",
    "semiten_r": "Semi",           # TEMP: PDF unresolved
    "soleus_r": "SOL",
    "tfl_r": "TFL",
    "tibant_r": "TA",
    "tibpost_r": "SOL",
    "vasint_r": ("VastLat", "VastMed"),
    "vaslat_r": "VastLat",
    "vasmed_r": "VastMed",
}

PROVISIONAL_MUSCLES = frozenset({
    "addbrev_r", "addlong_r", "addmagDist_r", "addmagIsch_r",
    "addmagMid_r", "addmagProx_r", "edl_r", "fdl_r", "fhl_r",
    "grac_r", "iliacus_r", "piri_r", "psoas_r", "semiten_r",
})


def mapping_source(base_actuator_name: str):
    """Return (reference channels, leg, provisional), failing on unknown muscles."""
    side = base_actuator_name[-1:]
    if side not in ("l", "r") or not base_actuator_name.endswith("_" + side):
        raise ValueError(f"Not a bilateral MyoLeg muscle: {base_actuator_name!r}")
    right = base_actuator_name[:-1] + "r"
    if right not in RIGHT_MAPPING:
        raise ValueError(f"Unknown MyoLeg actuator {base_actuator_name!r}; update mapping explicitly")
    src = RIGHT_MAPPING[right]
    channels = (src,) if isinstance(src, str) else tuple(src)
    return channels, side, side == "l" or right in PROVISIONAL_MUSCLES


class MuscleMapper:
    """One 15→80 mapping, usable with CPU dictionaries or batched GPU tensors.

    Maps names to the current installed MyoLeg actuator order. The user may
    replace the provisional RIGHT_MAPPING above at any time.
    """

    def __init__(self, actuator_names: Sequence[str], device=None):
        self.actuator_names = tuple(str(s) for s in actuator_names)
        if not self.actuator_names or len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("Actuator names must be nonempty and unique")
        self.provisional = []
        self.unresolved = []
        self._right_weights = np.zeros((len(self.actuator_names), len(CHANNELS)), dtype=np.float32)
        self._left_weights = np.zeros_like(self._right_weights)
        for i, full in enumerate(self.actuator_names):
            channels, side, provisional = mapping_source(full.rsplit('/', 1)[-1])
            weights = self._right_weights if side == 'r' else self._left_weights
            for channel in channels:
                weights[i, CHANNELS.index(channel)] = 1.0 / len(channels)
            if provisional:
                self.provisional.append(full)
        self.device = None
        if device is not None:
            self.to(device)

    def to(self, device):
        """Copy the SAME mapping matrix to GPU once, not each control step."""
        import torch
        self.device = torch.device(device)
        self.right_weights = torch.as_tensor(self._right_weights, device=self.device)
        self.left_weights = torch.as_tensor(self._left_weights, device=self.device)
        return self

    def expand(self, muscles_right, muscles_left=None):
        """One matrix-multiply path for scalar dicts and batched GPU tensors.

        Dict input: returns an 80-element NumPy array for compatibility.
        Tensor input: returns [B,80] on the input device, with no CPU copies.
        """
        import torch
        scalar = not isinstance(muscles_right, torch.Tensor)
        if scalar:
            if set(muscles_right) != set(CHANNELS):
                raise ValueError('Exactly the 15 right reference channels are required')
            if muscles_left is not None and set(muscles_left) != set(CHANNELS):
                raise ValueError('Exactly the 15 left reference channels are required')
            device = self.device if self.device is not None else torch.device('cpu')
            right = torch.tensor([[muscles_right[ch] for ch in CHANNELS]],
                                 dtype=torch.float32, device=device)
            left = (torch.tensor([[muscles_left[ch] for ch in CHANNELS]],
                                 dtype=torch.float32, device=device)
                    if muscles_left is not None else None)
        else:
            right, left = muscles_right, muscles_left
            if right.ndim != 2 or right.shape[1] != len(CHANNELS):
                raise ValueError('Expected batched [num_envs,15] right data')
            if left is not None and (left.shape != right.shape or left.device != right.device):
                raise ValueError('Incompatible left/right batched profiles')
        if self._left_weights.any() and left is None:
            raise ValueError('Left-leg phase-shifted profiles must be passed explicitly')
        if left is None:
            left = torch.zeros_like(right)
        if self.device is None or self.device != right.device:
            self.to(right.device)

        # The ONLY mapping calculation for CPU and GPU.
        mapped = right @ self.right_weights.T + left @ self.left_weights.T
        return mapped[0].detach().cpu().numpy() if scalar else mapped

    @staticmethod
    def unused_reference_channels() -> list[str]:
        used = {ch for v in RIGHT_MAPPING.values()
                for ch in ((v,) if isinstance(v, str) else v)}
        return sorted(set(CHANNELS) - used)
