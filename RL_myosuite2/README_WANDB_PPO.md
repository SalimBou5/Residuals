# Step 6 — Dynamic MEP validation, PPO training and online W&B monitoring

**Scope:** An additive training patch for the **existing, tested Step-5 V2** `gait_rl` package. We do **not** change actions, rewards, phase dynamics, MEP, RSI, data refinement or normalization. Only one existing production file is touched: `gait_rl/env.py`, which adds the selected reference path to the `step()` info dictionary for diagnostics.

This is the current **485-observation / 71-action** configuration: 70 MEP-mapped muscles with a residual plus one phase-rate action. Ten unmapped muscles remain disabled. Training the 81-action variant is supported but requires a separately fitted **505-D** observation normalizer.

## Installation

Extract the ZIP **at the root of the same project that already contains `gait_rl/`**, keeping folder structure. Use the **same Python environment** as your working Step 5 setup:

```powershell
python -m pip install -r requirements_step6.txt
wandb login
```

W&B login is interactive and stores your API credentials on **your** computer; do not put API keys into a script. Training uses the online mode by default. Use `--wandb-mode offline` when not connected; run `wandb sync PATH_TO_OFFLINE_RUN` later. `--wandb-mode disabled` disables W&B.

**Paths:** In the examples replace model, reference, MEP, and lookup paths with your actual local locations. These are not included in this patch. `myolegs.xml` must still be loadable with all its original `<include>` and mesh assets.

## 1. Validate real dynamics without a policy

```powershell
python smoke_test_mep.py  --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\\Users\\BoussofaraS\\Documents\\residuals\\RL_myosuite2\\data_refining\\gait_training_clean --mep "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\myoleg_bilateral_mep_80.csv" --lookup "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\activation_profiles\\activation_lookup.npz"  --normalization "C:\Users\BoussofaraS\Documents\residuals\RL_myosuite2\normalization_parameters\observation_normalization_refined_944.json"  --reward-config "C:\Users\BoussofaraS\Documents\residuals\RL_myosuite2\reward_config\reward_calibrated_v2.json"  --episodes 8 --seconds 2 --output "runs\mep_smoke"
```

This initializes random references/phases via RSI (including the activation lookup), then repeatedly applies **MEP excitation with no residual and no phase policy action** while MuJoCo integrates at **1,000 Hz**. The policy loop runs at **200 Hz**. The script verifies finite states/rewards/observations, bounded excitations, and that commands match the MEP. It records pelvis/contact forces, phase, speed, tracking errors, normalization clipping and reasons for termination.

Outputs:
- `runs/mep_smoke/dynamic_steps.csv` — each dynamic control step
- `runs/mep_smoke/episodes.csv` — one summary per episode
- `runs/mep_smoke/summary.json` — aggregate results and settings

**Falling is expected and is not treated as a numerical failure.** This is not replay: `qpos`, `qvel` and `act` are **not imposed** after the initial reset.

## 2. Train PPO with online Weights & Biases

```powershell
python train_ppo_wandb.py  --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\\Users\\BoussofaraS\\Documents\\residuals\\RL_myosuite2\\data_refining\\gait_training_clean --mep "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\myoleg_bilateral_mep_80.csv" --lookup "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\activation_profiles\\activation_lookup.npz"  --normalization "C:\Users\BoussofaraS\Documents\residuals\RL_myosuite2\normalization_parameters\observation_normalization_refined_944.json"  --reward-config "C:\Users\BoussofaraS\Documents\residuals\RL_myosuite2\reward_config\reward_calibrated_v2.json" --wandb-project "myoleg-residual-gait" --n-envs 8 --total-steps 2000000
```

The training command automatically runs **3 MEP-only dynamic preflight episodes** first and records them alongside the run. A fall is reported, not used as an automatic reason to stop PPO. Any invalid model, reference, nonfinite state, unexpected control bounds or inconsistent activation will stop the run with an exception. Explicitly use `--preflight-episodes 0` if you want to skip this safety check.

Initial PPO settings (provisional; compare checkpoints before tuning):

| Parameter | Initial value |
|---|---|
| Algorithm | Stable-Baselines3 PPO, separate 256×256 actor and critic MLPs |
| Workers | 8 independent MuJoCo processes (`spawn` on Windows) |
| Observation / action | 485 / 71 |
| Initial policy action standard deviation | 0.10 (MEP residual action is near 0 initially) |
| Physics / control | 1 ms / 5 ms |
| Rollout per PPO update | 1,024 transitions/worker, 8,192 total |
| PPO minibatch / epochs | 512 / 10 |
| Learning rate / γ / GAE λ | 3×10⁻⁴ / 0.99 / 0.95 |
| PPO clip / entropy coefficient | 0.20 / 0.0 |
| Total transitions | 2,000,000 initially |
| Episode limit | 4 seconds (time-limit = truncation) |
| Checkpoints | Every ~100,000 total transitions |
| Evaluation | Every ~100,000 transitions; same 8 valid RSI reset cases per run |
| Reference loading | Lazy cache of at most 16 cycles **per worker**, not full preloading of 944 |

### W&B graphs

W&B uploads from the **parent process only**, with no TensorBoard dependency. Relevant graphs include:

- `reward/step`, `reward_terms/imitation`, individual joint/site tracking scores, speed and uprightness scores, and every penalty/contribution.
- `reward_terms/mean_site_position_error_m`, `reward_terms/world_root_position_error_m`, `reward_terms/actual_forward_speed_m_s`, `reward_terms/desired_forward_speed_m_s`, requested residual costs and applied residual RMS.
- `state/phase_drift`, `state/event_remaining`, `state/phase_rate_hz`, `events/accept_fraction` and raw event counts.
- `normalization/clipped_fraction`, `action/muscle_abs_mean`, `action/phase_abs`.
- `references/unique_total`, `references/unique_window`, `episodes/mean_return`, `episodes/mean_length_steps`, `episodes/ended_<reason>` and `episodes/fraction_<reason>`: distinguish falls/tracking failures from time-limit truncations.
- `sb3/train/*` (policy/value losses, approximate KL, entropy, learning rate, explained variance), collected through SB3's native logger, and `eval/mean_episode_return`.

All plots use **total environment transitions** on the horizontal axis. Each PPO update summarizes one rollout; individual control-step data are not uploaded. `VecMonitor` additionally records episode returns and lengths to a local CSV.

### Run outputs

Each run has its own `runs/ppo_myoleg/<name>/` directory containing:

- `mep_preflight/` — dynamic baseline CSV/JSON, if enabled
- `checkpoints/myoleg_ppo_*_steps.zip` — intermediate SB3 checkpoints
- `evaluation/best_model.zip` — highest evaluation mean return so far
- `final_model.zip` — final PPO checkpoint
- `episode_monitor.csv`, `sb3_logs/`, `run_manifest.json` (local paths and hashes)

The W&B configuration contains only hyperparameters and file hashes, **not your local file paths**. To also upload model checkpoints to W&B, pass `--upload-final-model` (final and best only).

### Continue training

```powershell
python train_ppo_wandb.py <SAME SIX INPUT PATH ARGUMENTS> `
  --resume "runs\ppo_myoleg\YOUR_RUN\final_model.zip" `
  --total-steps 1000000 --wandb-name "continued_run"
```

`--total-steps` means *additional* steps when resuming. The loaded checkpoint retains its original optimizer and PPO hyperparameter schedule; new learning-rate arguments are not applied silently. Keep the same observation normalizer, reward, MEP setup and action layout for a meaningful continuation. A new W&B run is created, linked by run name rather than falsely merged into an old timeline.

## 3. Evaluate a checkpoint with full dynamic physics

```powershell
python evaluate_ppo.py  --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\\Users\\BoussofaraS\\Documents\\residuals\\RL_myosuite2\\data_refining\\gait_training_clean --mep "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\myoleg_bilateral_mep_80.csv" --lookup "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\activation_profiles\\activation_lookup.npz"  --normalization "C:\Users\BoussofaraS\Documents\residuals\RL_myosuite2\normalization_parameters\observation_normalization_refined_944.json"  --reward-config "C:\Users\BoussofaraS\Documents\residuals\RL_myosuite2\reward_config\reward_calibrated_v2.json" --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\evaluation\best_model.zip"  --episodes 16 --seconds 4 --output "runs\evaluate_best"
```

The evaluation script uses the deterministic policy by default and writes the same dynamic per-step/episode CSVs as the MEP baseline. Use `--stochastic` to evaluate sampled actions. **Do not interpret improving training reward as sufficient evidence of stable walking**: check time-limit completion rates, speed, uprightness, tracking errors, contact patterns, action magnitudes and phase corrections.

## Precautions and limitations

1. **Not yet executed against your complete local setup:** this package was checked by Python compilation, pure-Python unit tests and the existing Step-5 reward/phase/RSI regression tests. MuJoCo, Gymnasium, SB3 and W&B are not installed in the artifact environment, and your full dataset/linked model asset tree is not mounted here. Your local run is the first real dynamics + multiprocess PPO integration test.
2. Synthetic joint perturbations and replay analyses are not dynamic training results. The Step-5 V2 reward weights remain provisional; change them only after observing real policy dynamics.
3. The reference selection uses the existing **subject→trial→stride** hierarchical sampling. It does not sample all 944 references uniformly, preserving the original project's sampling semantics.
4. Evaluation uses a fixed set of **valid random reference/phase resets**, not a subject-held-out test set. It helps compare checkpoints within one run; it does **not** establish out-of-subject generalization.
5. The MEP baseline has 10 unmapped actuators held at zero in the default 71-action mode. Optional 81-action mode controls those actuators directly and needs its own 505-D normalizer.
6. The existing model runs at **1000 Hz / 200 Hz** according to the Step-5 rollout code. This takes precedence over older notes proposing 500 Hz / 100 Hz.

## Exact files added or modified

**Modified:**
- `gait_rl/env.py`: adds `info['reference']` to each `step()` for episode attribution.

**Added:**
- `gait_rl/step6.py`: safe worker environment factory, MEP-only action, dynamic validation and W&B diagnostics aggregator.
- `smoke_test_mep.py`: random-phase, MEP-only, full-physics preflight with CSV/JSON.
- `train_ppo_wandb.py`: PPO + Windows parallelism, W&B, evaluation, checkpoints and resume.
- `evaluate_ppo.py`: deterministic/stochastic dynamic checkpoint evaluation.
- `requirements_step6.txt`, this README, `tests/test_step6_training.py`.

**Included unchanged for convenience:** current V2 `reward_config/reward_calibrated_v2.json` and fitted 944-reference `normalization_params/observation_normalization_refined_944.json`.

**Untouched:** `gait_rl/rewards.py`, `gait_rl/observations.py`, `gait_rl/control.py`, `gait_rl/phase.py`, `gait_rl/rsi.py`, `gait_rl/rollout.py`, `gait_rl/reference.py` and the entire `data_refining/` tree.
