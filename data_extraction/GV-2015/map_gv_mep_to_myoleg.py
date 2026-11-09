from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

gv = pd.read_csv(HERE / "outputs/data/mep_1p2ms.csv")
ips = pd.read_csv(HERE / "outputs/data/iliopsoas_from_TA.csv")

if not np.allclose(gv["gait_cycle_pct"], ips["gait_cycle_pct"], atol=1e-5):
    raise ValueError("The two files have different gait-cycle grids.")

# None = zero placeholder for muscles without available data
mapping = {
    "addbrev_r": "Add",
    "addlong_r": "Add",
    "addmagDist_r": "Add",
    "addmagIsch_r": "Add",
    "addmagMid_r": "Add",
    "addmagProx_r": "Add",
    "bflh_r": "BFem",
    "bfsh_r": "BFem",
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
    "glmin1_r": ("GlutMed", "GlutMax"),
    "glmin2_r": ("GlutMed", "GlutMax"),
    "glmin3_r": ("GlutMed", "GlutMax"),
    "grac_r": None,          # Camargo data not yet provided
    "iliacus_r": "ips:iliacus",
    "perbrev_r": "Per",
    "perlong_r": "Per",
    "piri_r": None,          # Neglect
    "psoas_r": "ips:psoas",
    "recfem_r": "RFem",
    "sart_r": "Sar",
    "semimem_r": "Semi",
    "semiten_r": "Semi",
    "soleus_r": "Sol",
    "tfl_r": "TFL",
    "tibant_r": "TA",
    "tibpost_r": "Sol",
    "vasint_r": ("VastLat", "VastMed"),
    "vaslat_r": "VastLat",
    "vasmed_r": "VastMed",
}

assert len(mapping) == 40

mep = pd.DataFrame({"gait_cycle_pct": gv["gait_cycle_pct"]})

for muscle, source in mapping.items():
    if source is None:
        mep[muscle] = 0.0
    elif isinstance(source, tuple):
        mep[muscle] = gv[list(source)].mean(axis=1)
    elif source.startswith("ips:"):
        mep[muscle] = ips[source.split(":", 1)[1]]
    else:
        mep[muscle] = gv[source]

# Save the right-leg signals
mep.to_csv(HERE / "outputs/data/myoleg_right_mep_40.csv", index=False)

# Generate left-leg signals with a 50% gait-cycle shift
phase = mep["gait_cycle_pct"].to_numpy(dtype=float)
shifted_phase = (phase + 50.0) % 100.0

for right_muscle in mapping:
    left_muscle = right_muscle.removesuffix("_r") + "_l"

    mep[left_muscle] = np.interp(
        shifted_phase,
        phase,
        mep[right_muscle].to_numpy(dtype=float)
    )

# Save all 80 muscles
mep.to_csv(HERE / "myoleg_bilateral_mep_80.csv", index=False)

print(f"Saved {len(mep)} samples x {len(mapping)} muscles")