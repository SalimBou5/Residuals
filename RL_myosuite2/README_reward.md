# MyoLeg MEP-informed residual gait RL — Step 5 (reward + reference diagnostics)

**Scope:** small patch on top of the Step-4 fixed-observation normalization release.
No files in `data_refining/`, `reference.py`, `observations.py`, `control.py`, or `phase.py` were changed. This is not a PPO training implementation.

## Files

Replaced:
- `gait_rl/env.py` — evaluates reward after each physics/control step and returns termination/truncation info.
- `gait_rl/__init__.py` — exports reward classes.

Added:
- `gait_rl/rewards.py` — raw-physics-unit reward and conservative termination logic.
- `try_step5.py` — short MEP-only reward smoke test (a fall is expected).
- `inspect_reference_rewards.py` — 944-reference baseline scan from the already-saved Step-4 RAW observation NPZ (does not need MuJoCo).
- `inspect_reward_perturbations.py` — actual MuJoCo forward-kinematic perturbation scan to assess relative reward sensitivities (requires your local model and full prepared reference library).
- `tests/test_rewards_step5.py` — reward invariance, time-warp, phase penalty, dimensions and geometry tests.
- `reward_config/reward_provisional_v1.json` and `reward_config/termination_provisional_v1.json` — editable **provisional** hyperparameters.
- `reports/*` — baseline analysis already computed from your 23,600 uploaded raw observation samples.

## Reward formula

\[
 r = \underbrace{\sum_{x\in\{q,\dot q,p,R,v,\omega\}} w_x\exp(-\beta_xE_x)}_{r_{\rm imitation}}
     +w_s\exp[-\beta_s(v_{\rm forward}-v_{\rm desired})^2]
     +w_u\exp[-\beta_u(1-\hat z_{\rm body}\cdot\hat z_{\rm world})]
     -\lambda_{\rm res} C_{\rm res}-\lambda_\phi C_\phi
     -\lambda_1C_{L1}-\lambda_2C_{L2}.
\]

- `E_q`: mean squared error over **28 nonroot joints** (rad^2).
- `E_dq`: mean squared error over **28 nonroot joint velocities** ((rad/s)^2). Target velocity is time-warped by `reference.duration * clock.phase_rate_hz`.
- `E_p`: mean of squared **3-D Euclidean position errors** for 6 sites: knees, ankles, toes. **No hip positions.**
- `E_R`: mean squared geodesic relative orientation error (radians squared) at 8 sites: hips, knees, ankles, toes. Quaternion sign is immaterial.
- `E_v`: mean squared **3-D Euclidean velocity errors** for the same 6 sites. **No hip linear velocities.**
- `E_omega`: mean squared 3-D angular velocity errors at the same 8 sites.
- `C_res`: weighted mean square of **requested** mapped muscle excitation residuals `lambda_res_i * action_i`, not raw action, not accidentally reduced by excitation clipping. Unmapped muscles do not have an MEP prior, so are excluded here.
- `C_phi`: `(kappa_max * policy_phase_action)^2` — event-feedback correction is **not** penalized.
- `C_L1/L2`: mean abs activation and mean square activation over all 80 actual MuJoCo muscle activations (`data.act` in actuator ID order). 10 unmapped remain zero when disabled.
- All parts logged as `info['reward_terms']`, including raw errors, scores, weighted contributions, costs, applied residual RMS and termination metrics.
- Reward uses raw physical-unit blocks **before** Step-4 normalization. No action/residual smoothness penalty was introduced.

### Important axis conventions discovered using *your* refined reference samples

**MyoLeg's pelvis BODY local +Y points up**, whereas the global heading of the provided references is **world +Y**. The default upright body is `pelvis` with local axis `(0,1,0)`; the online signed speed is projected along each reference's `root_displacement` (not hard-coded to +Y). Applying generic pelvis local +Z/world +X conventions would have made healthy gait score nearly zero on both terms. Do not change these without a model-specific check.

### Initial (NOT validated) temperatures and weights

The six imitation weights sum to 1:

| Score | Weight | RMS deviation giving half-score |
|---|---:|---|
| Joint position | 0.20 | 0.15 rad over joints |
| Joint velocity | 0.10 | 2 rad/s over joints |
| Root-relative site position | 0.28 | 6 cm Euclidean per site |
| Site orientation | 0.14 | 0.20 rad geodesic |
| Root-relative site linear velocity | 0.18 | 0.50 m/s Euclidean per site |
| Site angular velocity | 0.10 | 2 rad/s Euclidean per site |
| Signed forward speed | +0.20 | 0.35 m/s deviation |
| Upright | +0.20 | 30-degree tilt |

`lambda_residual=0.50`, `lambda_phase=0.05`, `lambda_activation_l1=0.03`, `lambda_activation_l2=0.02`. They are **starting values**, not inferred from the near-perfect reference imitation score. Adjust only after reviewing sensitivity results, and do not treat near-perfect imposed replay as a simulated walking result.

### Default termination checks (also provisional)

Immediate if pelvis height < 0.55 m or torso/pelvis local-up points below horizontal. After 5 control-step grace, terminate if mean site position error > 0.50 m, world root error > 1 m, or free root orientation error > 1.5 rad. Time limit is a Gymnasium *truncation*, not a termination. Disable tracking ET during sensitivity scans; it should be re-evaluated during first real PPO experiments.

## Baseline from the refined training library (already computed)

`reports/reward_reference_baseline.json` summarizes **23,600 uniformly selected phases over 944 refined references**.

| Quantity | Mean | P05 | P95 |
|---|---:|---:|---:|
| Total nominal reward | 1.3699 | 1.3223 | 1.3932 |
| Six imitation scores | ~1.0000 | ~1.0000 | ~1.0000 |
| Signed speed score | 0.9022 | 0.6695 | 0.9993 |
| Upright score | 0.9681 | 0.8986 | 0.9981 |
| Actual muscle activation mean abs (80 muscles) | 0.1194 | 0.1059 | 0.1344 |
| Actual muscle activation mean square | 0.0265 | 0.0201 | 0.0316 |

Interpretation:
- Perfect joint/site imitation is **almost tautological** during imposed replay (`qpos/qvel` initialized from reference and the matching FK targets). It cannot determine useful `beta`/`w` alone.
- Speed term is an approximation here: estimated via world +Y reconstructed from pelvis-local velocity; in online reward, the actual direction comes from the *individual* reference root displacement.
- Phase and requested muscle residual costs are zero for neutral policy and therefore **cannot be calibrated** from the passive reference replay.
- Muscle activation costs use the periodic RSI lookup, **not** actual activations after a learned policy deviates.
- Contact sensors are used in observations / phase-event detection, **not directly in this reward**; no force calibration claim is made.
- Use the MuJoCo FK perturbation diagnostic below for realistic cross-coupling of joint and site errors; then inspect first true dynamic rollouts. No stable PPO policy is required to perform this static perturbation test.

## Commands

1. Baseline from the uploaded NPZ (only needed if you want to reproduce):

```powershell
python inspect_reference_rewards.py --samples "PATH_TO_observation_samples.npz" --normalization "normalization_params/observation_normalization_refined_944.json" --reward-config "reward_config/reward_provisional_v1.json" --output reward_reference_report
```

2. **Recommended next calibration test**: static FK perturbations across 100 randomly selected references and two RSI phases per reference; no integration and no policy needed:

```powershell
python inspect_reward_perturbations.py --model "PATH_TO_myolegs.xml" --references "PATH_TO_gait_training_clean" --mep "PATH_TO_myoleg_bilateral_mep_80.csv" --lookup "PATH_TO_activation_lookup.npz" --reward-config "reward_config/reward_provisional_v1.json" --output reward_perturbation_report
```

Look at `perturbed_reward_summary.csv`: it shows how each reward term responds to small/moderate joint-angle errors, velocity errors, speed errors, and pelvis tilt after rerunning *real MyoLeg FK*.

3. Runtime smoke test (no policy, falling expected):

```powershell
python try_step5.py --model "PATH_TO_myolegs.xml" --references "PATH_TO_gait_training_clean" --mep "PATH_TO_myoleg_bilateral_mep_80.csv" --lookup "PATH_TO_activation_lookup.npz" --normalization "normalization_params/observation_normalization_refined_944.json" --steps 100
```

You may pass a reward config to the smoke test; for later PPO training, instantiate `MyoLegResidualEnv(..., reward_config=RewardConfig(...), termination_config=TerminationConfig(...))`.

## Validation and limitations

- 61 tests passed (Step-4 tests plus Step-5 unit tests).
- Real offline baseline executed from your attached **refined** observation archive.
- The online MyoLeg reward and physical perturbation scan still need a run with your local model and actual NPZ training library; neither has been executed against the full model here.
- No `mj_step` is used in either offline diagnostic; timing and contact-triggered phase corrections must be assessed during dynamic rollouts.
