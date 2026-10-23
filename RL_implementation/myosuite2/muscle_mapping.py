"""PROVISIONAL González-Vargas -> right-side MyoLeg mapping from supplied PDF.

Every listed mapping yields the same *profile value* for each corresponding muscle.
We do NOT divide one profile by the number of target muscles.
Unknown/left-side actuators remain NaN and are returned explicitly as unresolved.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from gait_reference import CHANNELS

# None = unresolved by the PDF; do NOT silently treat as zero activation.
# Tuple(str, ...) = mean of the named reference profiles.
# Source: 'MYOLEG vs Gonzales-Vargas(1).pdf', pages 1-4.
RIGHT_MAPPING = {
    "addbrev_r": None,       # PDF: Adductor group not assigned (channel 'Add' unused)
    "addlong_r": None,
    "addmagDist_r": None,
    "addmagIsch_r": None,
    "addmagMid_r": None,
    "addmagProx_r": None,
    "bflh_r": "BFem",
    "bfsh_r": "BFem",       # Long/short head provisional group
    "edl_r": None,
    "ehl_r": "TA",
    "fdl_r": None,
    "fhl_r": None,
    "gaslat_r": "GastLat",
    "gasmed_r": "GastMed",
    "glmax1_r": "GlutMax",
    "glmax2_r": "GlutMax",
    "glmax3_r": "GlutMax",
    "glmed1_r": "GlutMed",
    "glmed2_r": "GlutMed",
    "glmed3_r": "GlutMed",
    "glmin1_r": ("GlutMed", "GlutMax"),  # PDF: average channels 8 & 14
    "glmin2_r": ("GlutMed", "GlutMax"),
    "glmin3_r": ("GlutMed", "GlutMax"),
    "grac_r": None,         # PDF: neglect OR obtain from Camargo
    "iliacus_r": None,
    "perbrev_r": "Per",
    "perlong_r": "Per",
    "piri_r": None,         # PDF suggests neglect
    "psoas_r": None,
    "recfem_r": "RFem",
    "sart_r": "Sar",
    "semimem_r": "Semi",
    "semiten_r": None,      # PDF: semimem OR Camargo; not yet decided
    "soleus_r": "SOL",
    "tfl_r": "TFL",
    "tibant_r": "TA",
    "tibpost_r": "SOL",
    "vasint_r": ("VastLat", "VastMed"),  # PDF: average channels 4 & 5
    "vaslat_r": "VastLat",
    "vasmed_r": "VastMed",
}


class MuscleMapper:
    def __init__(self, actuator_names: Sequence[str]):
        # Names and order must come from the ACTUAL MyoLeg simulation.
        self.actuator_names = tuple(str(s) for s in actuator_names)
        if not self.actuator_names:
            raise ValueError("Need actuator names in simulator actuator order")
        if len(self.actuator_names) != len(set(self.actuator_names)):
            raise ValueError("Duplicate actuator names")

    def expand(self, muscles_15: dict[str, float]) -> tuple[np.ndarray, list[str]]:
        if set(muscles_15) != set(CHANNELS):
            raise ValueError("Exactly the 15 named reference channels are required")
        u = np.full(len(self.actuator_names), np.nan, dtype=float)
        unresolved = []
        for i, name in enumerate(self.actuator_names):
            source = RIGHT_MAPPING.get(name)
            if source is None:
                unresolved.append(name)
            elif isinstance(source, tuple):
                u[i] = np.mean([muscles_15[channel] for channel in source])
            else:
                u[i] = float(muscles_15[source])
        return u, unresolved

    @staticmethod
    def unused_reference_channels() -> list[str]:
        used = set()
        for source in RIGHT_MAPPING.values():
            if isinstance(source, tuple):
                used.update(source)
            elif isinstance(source, str):
                used.add(source)
        return sorted(set(CHANNELS) - used)
