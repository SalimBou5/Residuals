# MyoLeg MEP-informed residual gait RL — Steps 1–4

This package **extends** the tested `test_rsi_mep_from_library.py` through a cached reference loader (Step 1), randomized state initialization (Step 2), a physics-rate action/phase controller (Step 3), and the **complete Gymnasium observation vector** (Step 4).

**Important:** This is not ready for PPO training: the full observation is implemented and validated with unit tests, but `reward=0.0` until Step 5. Failure termination and PPO training are also still pending. Do not train yet.


## Step 4 — observation specification (default 505 values)

The policy receives **actual and reference signals separately**, as agreed. There are no tracking-error features, no absolute pelvis XY coordinates, and no hip root-relative positions or linear velocities.

| Block | Dimensions | Ordering / physical meaning |
|---|---:|---|
| Joint proprioception | 56 | `qpos[7:]` (28), `qvel[6:]` (28) |
| Root state | 11 | pelvis height (1), world quaternion `wxyz` (4), pelvis-local linear and angular velocity (3+3) |
| Foot contacts | 4 | `r_foot`, `r_toes`, `l_foot`, `l_toes`, divided by `contact_force_scale_n=1000` (configurable), **not binary** |
| Muscle activations | 80 | Actual `data.act` reordered by **actuator ID** via activation addresses |
| Actual landmarks | 92 | 6 pelvis-relative positions (18); 6 pelvis-relative linear velocities (18); 8 relative `wxyz` orientations (32); 8 pelvis-relative angular velocities (24) |
| Reference landmarks | 92 | Corresponding trial-specific reference at current adapted phase; velocity tables scaled by actual phase rate |
| Target joints | 28 | Current reference nonroot joint positions |
| Future joints | 56 | Reference nonroot joint positions at **0.1 s and 0.2 s** lookahead (nominal reference duration) |
| MEP baseline | 80 | 80 actuator-ordered baseline excitations at current phase; unmapped entries zero |
| Gait phase | 6 | target speed, `sin(2πφ)`, `cos(2πφ)`, wrapped nominal-phase drift, remaining event correction, event active flag |
| **Total** | **505** | `float32`, fixed order and named slices |

Position/linear-velocity sites: `knee_r`, `knee_l`, `ankle_r`, `ankle_l`, `toe_r`, `toe_l`. Orientation/angular-velocity sites additionally include `hip_r`, `hip_l`. Reference site names are **looked up by name**, not by assumed array order. The reference file includes a pelvis entry but this is represented by the root block, not duplicated in the landmarks.

All actual site features use the same **moving pelvis frame** as the processed references:

```text
p_rel = R_pelvis.T @ (p_site - p_pelvis)
v_rel = R_pelvis.T @ (v_site - v_pelvis - omega_pelvis × (p_site - p_pelvis))
R_rel = R_pelvis.T @ R_site
omega_rel = R_pelvis.T @ (omega_site - omega_pelvis)
```

MuJoCo site and pelvis Jacobians produce world-frame velocities; they are transformed as above before comparison. To avoid equivalent `q` and `-q` representations producing artificial differences, actual relative quaternions are sign-aligned with the reference. **No reward implementation** is hidden in observations. The Step-5 reward will omit hip position/linear velocity errors consistently.

The nominal future lookahead uses `theta + horizon / reference_duration`, and loops across the RHS seam. Target velocity at the current phase uses `reference_duration × actual_phase_rate` (including RL and event corrections). Because the episode's phase can drift relative to the nominal clock, no absolute world root XY is exposed to the policy.

### Run the Step-4 actual-model smoke test

From the extracted project root, using the same model/data paths as your working Step-3 run:

```powershell
python try_step4.py --model "PATH_TO_MYOLEG_XML" --references "PATH_TO_GAIT_WITH_LANDMARKS" --mep "PATH_TO_BILATERAL_MEP_CSV" --lookup "PATH_TO_ACTIVATION_LOOKUP_NPZ" --steps 5 --seed 7 --no-event-sync
```

The test prints each block's `[start:stop]` slice and its value ranges, **plus RMS/max errors between actual and reference site quantities immediately after RSI**. It then checks 5 scripted policy intervals. Compare site parity carefully: orientation/position mismatches would indicate a frame/site mismatch, while smaller velocity differences can also reflect the preparation pipeline's finite-difference conventions. For your already-verified sensors, omit `--no-event-sync` to also enable phase synchronization. The seed `7` is intentionally reproducible.

To inspect individual blocks interactively:

```python
obs, info = env.reset(seed=7)
root = obs[env.observation_slices["root_state"]]
reference_sites = obs[env.observation_slices["reference_sites"]]
print(env.action_space.shape, env.observation_space.shape)  # (71,), (505,) by default
```

**Validation boundary:** the actual MyoLeg XML, 80-column MEP CSV and duration lookup were not provided here. The automated geometry/observation tests use the uploaded real 226-frame reference with a synthetic forward-kinematics engine. Run `try_step4.py` on your machine before trusting the real-model match.

## Agreed timing

| Component | When |
|---|---|
| MyoLeg MuJoCo physics | **0.001 s / 1000 Hz** — leave the original model timestep unchanged |
| Phase-clock progression | Every 0.001 s |
| MEP sampling (piecewise-linear periodic interpolation) | Every 0.001 s, at the *current* adapted phase |
| `data.ctrl` = interpolated MEP + held policy correction | Every 0.001 s |
| Sensor readings, HS/TO detection and queued event correction | Every 0.001 s |
| **71-D** RL action (default) or **81-D** (unmapped enabled) | Every **0.005 s / 200 Hz**, `frame_skip=5` |
| `data.act` internal muscle state | MuJoCo evolves this every physics step; initialized from lookup **once** at RSI |

The MEP has approximately 200 samples per original ~1-second gait cycle. Sampling the interpolator at 1000 Hz retains its original information and tracks phase warping; it does **not** create extra measured physiological information. Muscle and phase-rate policy actions are held over the five physics steps. We don't freeze the MEP between control steps.

## Action dimensions and indexing (configurable at environment creation)

| `enable_unmapped` | Muscle actions | Phase action index (0-based) | Total policy actions | Unmapped muscle excitation |
|---|---:|---:|---:|---|
| **False (default)** | **70**, mapped muscle actuator IDs only | **70** | **71** | Exactly **0**, no policy actions allocated |
| True | 80, all muscle actuator IDs | 80 | 81 | Direct control `(a + 1) / 2` |

Muscle actions appear in **ascending MyoLeg actuator-ID order** after filtering out unmapped actuators (when disabled). There is *no* assumption that unmapped muscles are grouped at the end. The mapping is available from `core.actions.policy_to_actuator_ids` and `core.actions.policy_muscle_names`. The final policy coordinate is always phase-rate control. `data.ctrl` remains **80-D in full actuator-ID order** in both modes. Similarly, `data.act` always remains **80-D**, because disabling policy outputs does not alter the physical model.

The Gymnasium environment sets `action_space.shape` automatically from `ActionConfig(enable_unmapped=...)`. To change mode, create a new environment and train a matching policy; do **not** reuse an old 81-output checkpoint with a 71-output environment. When unmapped muscles are enabled, a zero action means **0.5 excitation** on those channels; the proposed `-0.9` initial policy mean (about 0.05 excitation) will be set during PPO implementation, not during RSI.

For example:

```python
from gait_rl import ActionConfig, MyoLegResidualEnv
# Default: env.action_space.shape == (71,)
env = MyoLegResidualEnv(model_path, assets, action_config=ActionConfig())
# To create a separate full-action experiment:
full_env = MyoLegResidualEnv(model_path, assets, action_config=ActionConfig(enable_unmapped=True))
# full_env.action_space.shape == (81,)
```

### Per-step ordering

At reset: choose one reference / phase; set `qpos`, `qvel`, and `data.act` from the duration-matched periodic activation lookup; set initial `data.ctrl` to the MEP baseline for `mj_forward`; seed the contact detector (existing ground contacts are not new heel strikes); set `theta = theta_nominal = phi0`.

At each **policy** step: obtain 71 actions by default (or 81 when unmapped control is enabled). Repeat **five physics substeps**:

1. Compute the MEP at current unwrapped phase `theta` (internally uses `theta % 1`).
2. Apply mapped-muscle `clip(MEP + lambda_res * last_action, 0, 1)`. The ten unmapped muscles default to **exactly zero**. If enabled, they receive direct excitation `(action+1)/2`.
3. Assign `data.ctrl` and call `mj_step` once (1 ms). **Never rewrite `data.act` during the episode.**
4. Advance **unwrapped** phase using your exact nominal + RL + event increment equation, with clipping. The RL scalar stays fixed for the five substeps.
5. Read the latest verified touch forces. Debounce contact transitions, detect HS/TO, and set a **pending** event correction for subsequent physics steps. A new event does **not** jump phase instantly.

No pretrained policy or learned action is required to run the smoke test; scripted actions are supported.

## Source structure

```text
myoleg_rl_starter/
  gait_rl/
    reference.py      # Step 1: preload 969 refs; per-reference interpolation; MEP & lookup
    rsi.py            # Step 2: qpos/qvel/data.act initialization, model index validation
    phase.py          # Step 3: physics-step phase equation and buffered event correction
    contacts.py       # Step 3: verified touch sensor reader and event detector
    control.py        # Step 3: compact 70/80 muscle actions + 1 phase-rate action
    rollout.py        # Step 3: 5 physics steps per action, MEP updates each substep
    observations.py   # Step 4: 10-block, 505-D pelvis-relative observation
    env.py            # Gymnasium observation + Step 1–3 physics (reward still zero)
    __init__.py
  try_step4.py        # Step 4: actual-model observation and FK parity smoke test
  try_rsi.py          # Earlier validated RSI smoke test, preserved
  try_step3.py        # New headless scripted-action smoke test, no Gym required
  tests/              # Reference/RSI, phase, contact and integration mock tests
```

## Run Step 3 on your machine

Run from `myoleg_rl_starter/` in your **existing MyoSuite/MuJoCo environment** (Python, NumPy, MuJoCo; install `gymnasium` for `MyoLegResidualEnv`, `pytest` for tests).

**A. Inspect actual touch sensor names:**

```bash
python try_step3.py --model /path/to/myoleg.xml --list-sensors
```

We do **not** guess which sensor is which. You need four verified **scalar touch force** sensors corresponding to **right heel, right toe, left heel, left toe**, in exactly that order. If your model lacks these four sensors, we must first inspect its contact geometry or add an appropriately validated contact reader. We cannot claim event synchronization works in the real model without this verification.

**B. Run the phase/MyoLeg action loop WITHOUT event synchronization** (first smoke test):

```bash
python try_step3.py \
  --model /path/to/myoleg.xml \
  --references /path/to/gait_with_landmarks \
  --mep /path/to/myoleg_bilateral_mep_80.csv \
  --lookup /path/to/activation_lookup.npz \
  --steps 400 --print-every 20 --seed 7
```

`--muscle-action 0` means mapped muscles follow MEP exactly (no RL residual); unmapped muscles are zero in the default disabled mode. MEP alone is known not to sustain walking: falling during this test is **expected**, not evidence that the controller is broken. `--phase-action 0` means nominal phase speed; you can test `--phase-action 1` or `-1`. `--residual-limit 0.25` is an explicit **provisional calibration default**, not a validated physiological value.

**C. Run WITH contact-event synchronization**, after confirming the four sensor names:

```bash
python try_step3.py \
  --model /path/to/myoleg.xml \
  --references /path/to/gait_with_landmarks \
  --mep /path/to/myoleg_bilateral_mep_80.csv \
  --lookup /path/to/activation_lookup.npz \
  --contact-sensors RIGHT_HEEL_SENSOR RIGHT_TOE_SENSOR LEFT_HEEL_SENSOR LEFT_TOE_SENSOR \
  --steps 400
```

Replace `RIGHT_*` and `LEFT_*` **placeholders with real sensor names**, not arbitrary text. Check the force thresholds (`--contact-on-n`, `--contact-off-n`) against the model's normal-force magnitudes; defaults 20 N/10 N are **initial engineering settings**. If no sensors are specified, physics, MEP and the RL phase clock work, but **event sync is deliberately disabled**, with a warning.

**D. Run tests:**

```bash
python -m pytest -q
```

## Important phase details

At every **physics** step `j`, for the currently selected reference of duration `T`:

```text
delta_nominal = 0.001 / T
delta_rl      = kappa_max * delta_nominal * held_phase_action
delta_event   = clip(remaining_event_error, -eta*delta_nominal, +eta*delta_nominal)
step_normal   = clip(delta_nominal + delta_rl, min_step, max_step)
step_actual   = clip(delta_nominal + delta_rl + delta_event, min_step, max_step)
theta        += step_actual
theta_nominal += delta_nominal
remaining_event_error -= (step_actual - step_normal)
phi           = theta % 1
```

`min_step` and `max_step` are **positive** nominal-step multiples. The controller defaults to `kappa_max=0.3`, `eta=0.5`, activation tolerance=0.03 cycle, completion tolerance=0.0001 cycle, and rate bounds 0.2–2.0 times nominal. These should be tuned empirically.

An event is compared with its reference phase **only once when detected**. A discrepancy beyond the activation tolerance starts buffered correction. Once correcting, the completion tolerance controls when to stop; crossing back below the activation tolerance **does not** prematurely cancel it. A later valid event can replace the older pending correction. `phase_drift = theta - theta_nominal` and `phase_rate_hz` are available for the final observation/reward and time-warped reference velocities.

Contact detection uses hysteresis, debouncing, a refractory period, minimum event intervals, and a reference phase plausibility gate. The manifest's event timing provenance is preserved. **Assumed or inferred toe-off phases are skipped for synchronization by default**; `--use-assumed-toe-off` enables them experimentally. Right HS is the cycle origin; observed LHS is used when available. This avoids silently treating assumed toe-off timings as ground-truth measurements.

## Gymnasium environment (observation complete, reward pending)

```python
from gait_rl import ReferenceAssets, MyoLegResidualEnv
assets = ReferenceAssets('/path/to/gait_with_landmarks',
                         '/path/to/myoleg_bilateral_mep_80.csv',
                         '/path/to/activation_lookup.npz')
env = MyoLegResidualEnv('/path/to/myoleg.xml', assets)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

It uses **direct MuJoCo `data.ctrl` writes**, bypassing MyoSuite action normalization, because excitation must be updated at *every physics step*. The environment defaults to your VERIFIED touch sensor names: `("r_foot", "r_toes", "l_foot", "l_toes")`. Set `event_sync=False` to disable phase synchronization while **still providing real contact observations**. A missing sensor is a hard error, not silently replaced by zero.

The agreed **full ten-block observation** is now implemented, excluding hip root-relative positions/linear velocities while retaining hip relative orientation/angular velocity. Reward is **exactly zero** and episode termination is not finalized. Imitation rewards, physical-failure/early termination, and PPO training follow in Steps 5–7. **Do not train yet.**

Policy initialization for unmapped-muscle action means near `-0.9` is **deferred to PPO**, as agreed; the default disabled mode sends exactly zero regardless of action.

## Validation boundary

The test suite uses the **genuine uploaded 226-frame reference** plus synthetic MEP/activation lookup and a lightweight MuJoCo stand-in. The sample reference and its manifest are bundled under `tests/fixtures/`, so these tests work outside this conversation's workspace. They verify phase wrap/forward root displacement, RSI, 1-ms interpolation with held 5-ms residual, event hysteresis, deadband/completion, effective correction after clipping, assumed TO filtering, and model-dt validation.

**We have not run this against your actual MyoLeg XML or the real MEP/activation lookup**, which were not available in the attachments for this turn. In particular, real sensor naming, contact thresholds and gym runtime need verification on your machine. An actual training policy is not yet implemented.

## Step 4b — Inspect observation ranges before choosing normalization

The new `inspect_observation_ranges.py` calls the **same 505-dimensional**
`ObservationBuilder` used by the environment. It evaluates imposed reference
kinematics (MyoLeg `qpos` and `qvel`), the duration-matched RSI activation lookup,
and the phase-indexed experimental MEP. It does **not** integrate physics or
change observation normalization. It uses the existing MyoLeg model and all
verified touch sensors.

From this directory (PowerShell; replace paths with your own):

```powershell
python inspect_observation_ranges.py --model "C:\path\myolegs.xml" --references "C:\path\gait_with_landmarks" --mep "C:\path\myoleg_bilateral_mep_80.csv" --lookup "C:\path\activation_lookup.npz" --output observation_range_report --samples-per-reference 25
```

The default checks **25 evenly spaced samples from each of 969 references**
(about 24,225 poses); the script loads and caches all references once. For an
exhaustive reference scan, rerun with `--samples-per-reference 0`. Exact
minima, maxima, means and standard deviations are computed across **all poses
actually scanned**; percentiles are exact up to 40,000 samples and then
estimated from a bounded random reservoir.

Optional `--max-penetration-mm 15` skips heavily penetrating poses, but the
default includes all recorded poses so you can inspect the full library range.

The reports are saved as:

* `features.csv`: all 505 feature names, units, min/max, P01/P05/P50/P95/P99,
  mean/std, and diagnostic median/half-P01-to-P99 scale **candidates**.
* `blocks.csv`: concise 10-block ranges and number of constant features.
* `contact_forces.csv`: all four raw touch-sensor values in **newtons**.
* `metadata.json`: number of scanned samples and limitations.

**Important:** contact forces from kinematic replay are NOT actual GRFs and
can explode at penetrating poses. Do **not** select the final contact scaler
from this replay alone. Likewise, real phase drift/correction, dynamic muscle
activation, tracking errors and perturbed/falling states are *not* sampled.
They will need data from genuine dynamic training rollouts. Values from
reference imitation should be treated as an initial normalization diagnostic,
not an imposed clipping range or a final normalizer.
