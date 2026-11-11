# MyoLeg Step 5 — Reward recalibration V2

This is a **small patch over Step 5**, designed after analyzing the user's 944 refined gait references (23,600 imposed poses) and 2,400 static MyoLeg FK perturbation cases. It is **not** a trained-policy validation.

## Apply

Extract this ZIP into the existing RL project root, replacing the paths that already exist. `gait_rl/rewards.py` has the changed **default** `RewardConfig`, so creating the environment without passing a custom `reward_config` uses V2 automatically.

**If your script explicitly loads the old JSON, change its path** to `reward_config/reward_calibrated_v2.json`; otherwise it will override these new defaults with V1.

No modification to `env.py`, `observations.py`, `reference.py`, `phase.py`, `control.py`, `normalization.py`, or `data_refining/` is required.

## What changed

| Term | V1 | V2 | Rationale |
|---|---:|---:|---|
| Joint-position half-score RMS | 0.15 rad | 0.15 rad | Retain |
| Joint-velocity half-score RMS | 2 rad/s | 2 rad/s | Retain |
| Site-position half-score RMS | 0.06 m | **0.12 m** | Prevent saturation for modest joint errors |
| Site-orientation half-score RMS | 0.20 rad | **0.30 rad** | Less brittle early learning |
| Site-linear-velocity half-score RMS | 0.50 m/s | **1.50 m/s** | Preserve useful differentiation of deviations |
| Site-angular-velocity half-score RMS | 2 rad/s | **3 rad/s** | Less brittle |
| Upright half-score tilt | 30° | **18°** | Stronger posture discrimination |
| Upright additive weight | 0.20 | **0.45** | More costly leaning and falling |
| Activation L1 coefficient | 0.03 | **0.15** | Fivefold stronger regularization |
| Activation L2 coefficient | 0.02 | **0.10** | Fivefold stronger regularization |

The six imitation **weights** (`0.20, 0.10, 0.28, 0.14, 0.18, 0.10`) still sum to 1. The speed term is **unchanged**: `w_speed=0.20`, half-score error `0.35 m/s`. Residual and phase coefficients remain `0.50` and `0.05`, respectively.

### Reference speed: already implemented correctly

The actual signed pelvis forward velocity is projected onto the reference's **own horizontal displacement direction**. The target is `ref.desired_speed`, loaded from `forward_speed_m_s` in the reference NPZ. The observation builder also exposes this same target as `gait_phase[0]`. **It is not hard-coded to 1.3 m/s**, and it is not replaced with a phase-dependent instantaneous velocity target. No further speed-code change was needed.

### Upright convention

The reward projects **pelvis-body local +Y** onto **world +Z**. This matches `myolegs_chain.xml`'s pelvis quaternion. The +Y world heading of the standardized reference library is a separate convention. This calculation was not altered.

### Small bug fix

`rewards.py` and `inspect_reward_perturbations.py` now normalize forward-direction arrays *out of place*. This avoids the `ValueError: output array is read-only` reported when accessing readonly reference arrays.

## Measured results (same 23,600 imposed reference poses)

| Statistic | V1 | V2 |
|---|---:|---:|
| Ideal score, no penalties | 1.400 | **1.650** |
| Mean reference-only score | 1.370 | **1.573** |
| Mean upright score | 0.968 | **0.917** |
| Mean L1+L2 activation penalty | 0.00411 | **0.02055** |

V2 total is not directly comparable to V1 total without accounting for the higher maximum. Healthy reference walking remains high reward, but normal within-stride pelvic orientation is penalized somewhat more strongly.

### V1–V2 comparison for *exactly the same* 2,400 stored static FK cases

| Perturbation | V1 reward drop | V2 reward drop |
|---|---:|---:|
| Independent joint angle σ = 0.05 rad | 0.258 | **0.114** |
| Independent joint angle σ = 0.10 rad | 0.465 | **0.309** |
| Independent joint velocity σ = 1 rad/s | 0.254 | **0.157** |
| Independent joint velocity σ = 2 rad/s | 0.324 | **0.294** |
| Pelvis tilt 15° | 0.043 | **0.195** |
| Pelvis tilt 30° | 0.108 | **0.362** |
| Forward speed change −0.35 m/s | 0.086 | **0.086** |

These values were recalculated **exactly from archived physical error quantities and old upright scores**, not by rerunning MuJoCo. The rescore script independently verifies that the V1 configuration reproduces the saved old total reward within numerical tolerance. All differences are relative to the **original-pose condition of the corresponding version**.

Scaling the 70 saved mapped activation channels uniformly by ×2 (10 inactive unmapped channels left zero) raises the mean V2 activation penalty from **0.02055** to **0.04640**. This is a static sensitivity estimate, not an active muscle-dynamics simulation.

**Scientific limitation:** absolute activation penalties encourage lower activation, *not* specifically agreement with the MEP prior. That is intentional preservation of our existing L1/L2 objective, but if the policy chronically suppresses activity the regularizer should be reassessed. Residual penalties separately discourage large corrections to the MEP baseline.

## Files to overwrite / add

**Overwrite:**
- `gait_rl/rewards.py` — V2 default coefficients; readonly forward-array fix.
- `inspect_reference_rewards.py` — correct half-score labels in the sensitivity report.
- `inspect_reward_perturbations.py` — readonly fix; extra static activation sensitivity cases.

**Add:**
- `reward_config/reward_calibrated_v2.json` — explicit V2 values for CLI scripts and reproducibility.
- `rescore_saved_reward_probes.py` — recompute old archived FK probe rewards under V2 without MyoLeg simulation.
- `tests/test_rewards_step5_v2.py` — V2 regression tests.
- `reports/reward_reference_v2/` and `reports/reweighted_perturbations_v2/` — analysis results.
- `reports/activation_sensitivity_v2.csv` — static increase/decrease activation sensitivity.

The older `reward_config/reward_provisional_v1.json` is also included in the ZIP strictly for the historical comparison. Do not load it for training.

## Run checks on your machine

```powershell
python -m pytest -q tests/test_rewards_step5.py tests/test_rewards_step5_v2.py

python inspect_reference_rewards.py `
  --samples "PATH_TO_observation_samples(1).npz" `
  --normalization "normalization_params/observation_normalization_refined_944.json" `
  --reward-config "reward_config/reward_calibrated_v2.json" `
  --output reward_reference_v2

python inspect_reward_perturbations.py `
  --model "PATH_TO_myolegs.xml" `
  --references "PATH_TO_gait_training_clean" `
  --mep "PATH_TO_myoleg_bilateral_mep_80.csv" `
  --lookup "PATH_TO_activation_lookup.npz" `
  --reward-config "reward_config/reward_calibrated_v2.json" `
  --output reward_perturbation_v2
```

The exact saved FK probe re-score was executed here without MuJoCo. **The updated perturbation script and dynamic rollouts have not been run against your local full model**; we therefore cannot yet claim suitable reward balance during actual training.
