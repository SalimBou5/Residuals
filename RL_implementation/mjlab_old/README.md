# MyoSuite 3 — gait imitation using muscle-feedforward + policy residual

**Current status:** Algorithm-agnostic mjlab/PyTorch simulation and control/reward **smoke-test skeleton**. The actual RSL-RL/PPO manager-task registration is **not implemented yet**, and live MyoSuite GPU physics has not been tested in this environment.

## Control design (IMPORTANT)

The **15 gait-cycle-normalized activation profiles are the feedforward muscle command, not imitation targets**. At each control step for walker `i`:

1. Update its right-foot gait phase `phi_i` from the phase estimator.
2. Retrieve the 15 right-leg profiles at `phi_i`. **Provisionally**, use the same profiles at `(phi_i + 0.5) % 1` for the left leg.
3. Map those values to all **80 MyoLeg muscle actuator commands** in the real model actuator order.
4. Add the learned policy residual, with a configurable amplitude:

   `u_i = clip(u_feedforward_80(phi_i) + residual_scale * tanh(policy_action_i), 0, 1)`

5. Pass `u_i` to MyoSuite 3's `myoLegWalk-v0` mjlab environment, where these `[0,1]` values are muscle **excitation commands**. The simulator's filtered muscle-activation state `data.act` is different from commanded excitation.

**Zero policy output gives the feedforward command exactly** (provided its values are already within `[0,1]`). The RL output is *never* a full direct-muscle command. The default `residual_scale=0.05` is provisional and configurable.

## Imitation reward — kinematics only

The **joint kinematics** are the imitation targets. Muscle profiles do not appear in `ImitationReward`:

- `q_mse = mean((q_sim - q_ref(phi))**2)`
- `dq_mse = mean((dq_sim - dq_ref(phi))**2)`
- `r = w_q*exp(-k_q*q_mse) + w_dq*exp(-k_dq*dq_mse)`

Weights and gains are merely software-test defaults and should be validated later. No muscle MSE/reward term is computed. The policy may change muscle activations *away* from the feedforward baseline if that improves imitation of the **kinematics**.

## Provisional complete 15 → 80 mapping

`muscle_mapping.py` contains the 40 right-side muscle assignments; the corresponding 40 left muscles use the same muscle-to-channel mapping at the **contralateral phase**. The names and action ordering come from MyoSuite, not an assumed numeric ordering.

Assignments in the supplied **MYOLEG vs Gonzales-Vargas** PDF are retained where supplied. Previously unresolved muscle classes are now filled temporarily:

| Unresolved muscles in PDF | Temporary channels |
| --- | --- |
| Adductor brevis/longus/magnus (6 portions), gracilis | `Add` |
| Extensor digitorum longus | `TA` |
| Flexor digitorum / hallucis longus | `SOL` |
| Iliacus / psoas | `RFem` |
| Piriformis | `GlutMax` |
| Semitendinosus | `Semi` |

Other existing allocations and two-channel averages are unchanged. **These temporary choices are not validated physiological estimates**. Left-leg shift of 50% is also provisional; it assumes left/right gait symmetry. Unrecognized muscle names raise an explicit error instead of producing silently missing commands. Profiles are copied to each recipient muscle rather than divided by recipient count.

## Data input

`GaitReference.from_npz(...)` expects:

- `phase [N]`: increasing normalized cycle coordinate in `[0,1)`, right heel strike at phase 0.
- `muscle_activations [N,15]`, `muscle_names [15]`: `TA SOL Per VastLat VastMed RFem Sar Add GlutMed TFL GastLat GastMed BFem Semi GlutMax` (channels are automatically reordered by name).
- `joint_angles [N,D]` and `joint_velocities [N,D]`: right/left named MyoLeg joints when available; angles in radians, velocities in radians per **physical second**.
- `joint_names [D]`: required for kinematic reward, explicitly matching MyoLeg model names and conventions.

The phase lookup is periodic interpolation **inside existing gait-normalized data**. The reference is not normalized or resampled over simulation time again.

## Testing and setup

Use the [MyoSuite 3 setup instructions](https://github.com/MyoHub/myosuite/blob/main/docs/source/install.rst). Install from source with `[mjlab]` using a compatible PyTorch/CUDA stack. Run from outside the cloned MyoSuite repository's parent directory.

```bash
python test_skeleton.py              # runs CPU-only unit/mock tests
python inspect_myosuite3.py --num-envs 4 --steps 5  # requires MyoSuite 3/mjlab
python inspect_imitation.py \
  --reference path/to/prepared_reference.npz \
  --stride YOUR_STRIDE_SECONDS \
  --toe-off YOUR_TOE_OFF_FRACTION \
  --reset-phase YOUR_KEYFRAME_REFERENCE_PHASE \
  --num-envs 4 --steps 5
```

The imitation inspection uses an **all-zero policy action** (pure feedforward command). It is an interface test, not a claim that feedforward alone can produce stable walking. You still need to align the initial gait phase with MyoSuite's reset keyframe and verify kinematic conventions.

## RSI staging

- **Stage 1 (current)**: use MyoSuite default reset, initialize phase to configured `initial_phase`. No RSI.
- **Stage 2 (future)**: sample reference kinematic poses/velocities (muscle state may not be consistent).
- **Stage 3 (future)**: sample validated physical states saved from successful simulations, including muscle activation state, kinematics and phase.

The `rsi.py` sampler/buffer interfaces exist. Nonstandard physics-state restoration intentionally raises `NotImplementedError`; no fake RSI is performed.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `gait_reference.py` / `gpu_components.py` | Read already normalized reference; GPU periodic lookup, 80-channel mapping and per-walker phase estimator |
| `muscle_mapping.py` | Fully populated provisional bilateral mapping |
| `rl_env.py` | Map feedforward + learned residual to MuJoCo excitation; simulate and return kinematic reward |
| `imitation_reward.py` | **Kinematic imitation only**, algorithm-independent |
| `myosuite3_adapter.py` | GPU MyoLeg simulation, foot touch signals, joint access and actuator order |
| `rsi.py` | Future kinematic/physical RSI scaffolding |
| `inspect_imitation.py` | Real mjlab feedforward smoke test (needs your NPZ) |
| `test_skeleton.py` | Pure CPU tests with synthetic references/mock physics |

**Still needed for training:** integration into native mjlab manager-based terms for the RSL-RL runner, real prepared data format/convention validation, reset-phase alignment and selected reward design. Nothing here commits us to PPO permanently.
