# MyoLeg Step 4 — fixed observation normalization (refined Gait120)

This update changes **only the RL observation processing and diagnostic scripts**.
It does **not** change reference NPZs, the data-refinement pipeline, action
control, the physics/control frequencies, the phase controller, or rewards.

## Installation

Extract the patch ZIP directly into your existing `RL_myosuite2` project root,
replacing the supplied `gait_rl/*.py`, inspection/plotting scripts, and
`try_step4.py`. The `data_refining/` folder is **not** included or modified.

For the **485-D `enable_unmapped=False`** configuration, use the supplied
parameter file:

`normalization_params/observation_normalization_refined_944.json`

It was fitted from the supplied `features(2).csv` / `metadata(2).json`, covering
**944 retained references and 23,600 kinematic samples**. Source CSV and JSON
snapshots are included for reproducibility. The 505-D action/observation option
is still supported by the environment but requires a new **505-D inspection and
normalizer fit**; reusing 485-D parameters raises an error.

## Using the normalized observation

In the environment constructor:

```python
from gait_rl import MyoLegResidualEnv, ObservationConfig

env = MyoLegResidualEnv(
    model_path, assets,
    observation_config=ObservationConfig(
        normalization_file="normalization_params/observation_normalization_refined_944.json"
    ),
)
```

In the Step-4 smoke test (PowerShell):

```powershell
python try_step4.py --model "MODEL" --references "gait_training_clean" --mep "MEP_CSV" --lookup "LOOKUP_NPZ" --normalization "normalization_params/observation_normalization_refined_944.json"
```

Passing **no** `normalization_file` leaves the observations unnormalized
(including contact forces in raw Newtons); this is for debugging, not the
intended PPO configuration. `info['observation_normalized']` at reset reports
which configuration was used.

## Chosen rules

* **Kinematic joint, pelvis, site and target features:** `(x - median) / max((P99 - P01)/2, physical_floor)`; actual and corresponding reference features **share exactly the same parameters** (including future joint lookahead). Floor values are documented per feature in the JSON.
* **Four force sensor channels:** take **raw** force in Newtons into the observation builder and map `asinh(max(F,0)/200 N)` inside the fixed normalizer. This is continuous, monotonic, and retains both contact onset and support-load information. The 20/10 N HS/TO event thresholds still operate on the **raw** sensor values.
* **70 mapped muscle activations and 70 MEP excitations:** leave in physical [0,1] range; **not** normalized by their approximately 0.46 healthy peaks.
* **Quaternion components:** leave as canonical unit-quaternion components (`w >= 0` as already implemented); phase sine/cosine and correction-active flag remain unchanged.
* **Phase drift:** divide by 0.5 cycles. **Pending event correction:** divide by 0.3 cycles. Desired forward speed: reference-median centering with a minimum scale of 0.4 m/s.
* **Clip transformed continuous channels at ±10**, NOT to healthy reference percentiles. This leaves room for early falls/large tracking errors. The actual and target observations are identically scaled; clipping could still obscure very extreme states, so saturation should be monitored.

These values are **initial design choices**, not estimates of physical maximum
speeds. The reference dataset gives *baseline typical centers and variation*;
early RL rollouts will be out-of-distribution.

The Step-4 `info` dictionary reports:

- `normalization_clipped_features`: number of channels clipped on the latest observation
- `normalization_clip_fraction_total`: cumulative fraction clipped across returned observations

And `env.observations.last_blocks` retains **raw** block values for debugging;
`last_normalized_blocks` contains the actual post-normalization vectors.

## Refit parameters later

```powershell
python fit_observation_normalization.py --features "normalization_params/source_features_refined_944.csv" --metadata "normalization_params/source_metadata_refined_944.json" --output "normalization_params/observation_normalization_refined_944.json"
```

The previously produced distribution plots showed foot features bounded by
`F/(F+100 N)`, despite having kN axis labels. **Those contact percentiles are
not used in the fit.** The updated inspection tool outputs **raw Newton forces**,
and its plots now label their actual units. Do not feed already normalized
observations to the fixed normalizer a second time.

## Validation

The test suite checks layout, matching actual/reference scales, physically
bounded channels, contact-force continuity, OOD falling inputs, clipping
monitoring, and parameter/muscle-layout mismatch. The supplied source plots
and samples are based on `mj_forward` of imposed reference states; training
rollouts were not available to test dynamic distribution or clipping rates.
