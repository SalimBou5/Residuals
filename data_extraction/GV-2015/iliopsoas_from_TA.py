#!/usr/bin/env python3
"""Build an iliacus/psoas baseline from the TA column of mep_1p2ms.csv.

Keeps TA on [0, first_end] % and [swing_start, 100] % of the gait cycle, fades everything
else to 0 with C2-continuous (smootherstep) ramps. Ramps lie OUTSIDE the kept regions:
  first_end   -> first_end + ramp_out   : 1 -> 0
  swing_start - ramp_in -> swing_start  : 0 -> 1
"""
import argparse
import numpy as np


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (x * (6.0 * x - 15.0) + 10.0)


def gate_window(phase, first_end, swing_start, ramp_out, ramp_in):
    if ramp_out <= 0 or ramp_in <= 0:
        raise ValueError("ramp widths must be > 0 (0 would be a hard cut)")
    if first_end < 0 or swing_start > 100:
        raise ValueError("thresholds must lie in [0, 100] %")
    if first_end + ramp_out > swing_start - ramp_in:
        raise ValueError("ramps overlap: need first_end + ramp_out <= swing_start - ramp_in")

    p = np.asarray(phase, dtype=float)
    w = np.zeros_like(p)
    w[p <= first_end] = 1.0
    m = (p > first_end) & (p < first_end + ramp_out)
    w[m] = 1.0 - smootherstep((p[m] - first_end) / ramp_out)
    m = (p >= swing_start - ramp_in) & (p < swing_start)
    w[m] = smootherstep((p[m] - (swing_start - ramp_in)) / ramp_in)
    w[p >= swing_start] = 1.0
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="mep_1p2ms.csv", help="input CSV (gait_cycle_pct, TA, ...)")
    ap.add_argument("--out", default="outputs/data/iliopsoas_from_TA.csv")
    ap.add_argument("--first-end", type=float, default=10.0, help="keep TA from 0 %% to here")
    ap.add_argument("--swing-start", type=float, default=60.0, help="keep TA from here to 100 %%")
    ap.add_argument("--ramp-out", type=float, default=5.0, help="fade-out width after first-end (%% cycle)")
    ap.add_argument("--ramp-in", type=float, default=5.0, help="fade-in width before swing-start (%% cycle)")
    a = ap.parse_args()

    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    phase, ta = d["gait_cycle_pct"], d["TA"]

    w = gate_window(phase, a.first_end, a.swing_start, a.ramp_out, a.ramp_in)
    proxy = ta * w

    out = np.column_stack([phase, ta, w, proxy, proxy])   # same profile for iliacus and psoas
    np.savetxt(a.out, out, delimiter=",", header="gait_cycle_pct,TA,window,iliacus,psoas",
               comments="", fmt="%.6g")
    print(f"wrote {a.out}  (wrap check: window[0]={w[0]:.3f}, window[-1]={w[-1]:.3f})")


if __name__ == "__main__":
    main()
