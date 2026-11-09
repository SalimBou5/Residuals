import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load original CSV files
gv = pd.read_csv("mep_1p2ms.csv")
ips = pd.read_csv("iliopsoas_from_TA.csv")

# Check gait-cycle alignment
assert np.allclose(gv["gait_cycle_pct"], ips["gait_cycle_pct"])

phase = gv["gait_cycle_pct"]

plt.figure(figsize=(11, 5))

plt.plot(
    phase, gv["TA"],
    label="Tibialis anterior (original)",
    color="tab:blue",
    linewidth=2
)

plt.plot(
    phase, ips["psoas"],
    label="Psoas (derived from TA)",
    color="tab:orange",
    linewidth=2
)

plt.xlabel("Gait cycle (%)")
plt.ylabel("MEP amplitude")
plt.title("Tibialis Anterior vs Psoas")
plt.xlim(0, 100)
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 101, 10))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()