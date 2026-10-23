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
    """CPU helper; GPU hot path uses GPUMuscleMapper with the same mapping."""

    def __init__(self, actuator_names: Sequence[str]):
        self.actuator_names = tuple(str(s) for s in actuator_names)
        if not self.actuator_names or len(set(self.actuator_names)) != len(self.actuator_names):
            raise ValueError("Actuator names must be nonempty and unique")
        self.provisional = []
        for name in self.actuator_names:
            _channels, _side, provisional = mapping_source(name.rsplit("/", 1)[-1])
            if provisional:
                self.provisional.append(name)

    def expand(self, muscles_right: dict[str, float],
               muscles_left: dict[str, float] | None = None) -> np.ndarray:
        if set(muscles_right) != set(CHANNELS):
            raise ValueError("Exactly the 15 reference channels are required")
        if any(name.endswith('_l') for name in self.actuator_names) and muscles_left is None:
            raise ValueError("Supply the left-leg phase-shifted 15 channels")
        if muscles_left is not None and set(muscles_left) != set(CHANNELS):
            raise ValueError("Exactly the 15 left-leg channels are required")
        commands = []
        for name in self.actuator_names:
            channels, side, _provisional = mapping_source(name.rsplit("/", 1)[-1])
            profile = muscles_right if side == 'r' else muscles_left
            commands.append(np.mean([profile[ch] for ch in channels]))
        return np.asarray(commands, dtype=float)

    @staticmethod
    def unused_reference_channels() -> list[str]:
        used = {ch for v in RIGHT_MAPPING.values()
                for ch in ((v,) if isinstance(v, str) else v)}
        return sorted(set(CHANNELS) - used)
