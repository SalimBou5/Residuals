# MyoLeg MEP-informed residual gait RL — Steps 1, 2 and 3

This package **extends** the tested `test_rsi_mep_from_library.py` through a cached reference loader (Step 1), randomized state initialization (Step 2), and a physics-rate action/phase controller with a **minimal Gymnasium skeleton** (Step 3).

**Important:** This is not ready for PPO training: the `Gymnasium` observation is explicitly a *temporary diagnostic vector* and `reward=0.0` until Steps 4–5. The state initialization, action mapping, phase oscillator, contact-event synchronizer, and physics loop are implemented. No unvalidated reward or imitation observation is silently introduced.

## Agreed timing

| Component | When |
|---|---|
| MyoLeg MuJoCo physics | **0.001 s / 1000 Hz** — leave the original model timestep unchanged |
| Phase-clock progression | Every 0.001 s |
| MEP sampling (piecewise-linear periodic interpolation) | Every 0.001 s, at the *current* adapted phase |
| `data.ctrl` = interpolated MEP + held policy correction | Every 0.001 s |
| Sensor readings, HS/TO detection and queued event correction | Every 0.001 s |
| 81-dimensional RL action | Every **0.005 s / 200 Hz**, `frame_skip=5` |
| `data.act` internal muscle state | MuJoCo evolves this every physics step; initialized from lookup **once** at RSI |

The MEP has approximately 200 samples per original ~1-second gait cycle. Sampling the interpolator at 1000 Hz retains its original information and tracks phase warping; it does **not** create extra measured physiological information. Muscle and phase-rate policy actions are held over the five physics steps. We don't freeze the MEP between control steps.

### Per-step ordering

At reset: choose one reference / phase; set `qpos`, `qvel`, and `data.act` from the duration-matched periodic activation lookup; set initial `data.ctrl` to the MEP baseline for `mj_forward`; seed the contact detector (existing ground contacts are not new heel strikes); set `theta = theta_nominal = phi0`.

At each **policy** step: obtain 81 actions. Repeat **five physics substeps**:

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
    control.py        # Step 3: 80 MEP residual/direct actions + 1 phase-rate action
    rollout.py        # Step 3: 5 physics steps per action, MEP updates each substep
    env.py            # Gymnasium skeleton (NOT YET TRAINABLE)
    __init__.py
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
python try_step3.py  --model  "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\Users\BoussofaraS\Documents\residuals\data_extraction\Transform_cycles_to_episodes\gait_with_landmarks --mep C:\Users\BoussofaraS\Documents\residuals\data_extraction\GV-2015\outputs\data\myoleg_bilateral_mep_80.csv --lookup C:\Users\BoussofaraS\Documents\residuals\data_extraction\GV-2015\outputs\data\activation_profiles\activation_lookup.npz  --steps 400 --print-every 20 --seed 7
```

`--muscle-action 0` means mapped muscles follow MEP exactly (no RL residual); unmapped muscles are zero. MEP alone is known not to sustain walking: falling during this test is **expected**, not evidence that the controller is broken. `--phase-action 0` means nominal phase speed; you can test `--phase-action 1` or `-1`. `--residual-limit 0.25` is an explicit **provisional calibration default**, not a validated physiological value.

**C. Run WITH contact-event synchronization**, after confirming the four sensor names:

```bash
python try_step3.py  --model  "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\Users\BoussofaraS\Documents\residuals\data_extraction\Transform_cycles_to_episodes\gait_with_landmarks --mep C:\Users\BoussofaraS\Documents\residuals\data_extraction\GV-2015\outputs\data\myoleg_bilateral_mep_80.csv --lookup C:\Users\BoussofaraS\Documents\residuals\data_extraction\GV-2015\outputs\data\activation_profiles\activation_lookup.npz 
  --contact-sensors RIGHT_HEEL_SENSOR RIGHT_TOE_SENSOR LEFT_HEEL_SENSOR LEFT_TOE_SENSOR --steps 400
```

Replace `RIGHT_*` and `LEFT_*` **placeholders with real sensor names**, not arbitrary text. Check the force thresholds (`--contact-on-n`, `--contact-off-n`) against the model's normal-force magnitudes; defaults 5 N/2 N are **initial engineering settings**. If no sensors are specified, physics, MEP and the RL phase clock work, but **event sync is deliberately disabled**, with a warning.

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

## Gymnasium skeleton

```python
from gait_rl import ReferenceAssets, MyoLegResidualEnv
assets = ReferenceAssets('/path/to/gait_with_landmarks',
                         '/path/to/myoleg_bilateral_mep_80.csv',
                         '/path/to/activation_lookup.npz')
env = MyoLegResidualEnv('/path/to/myoleg.xml', assets)
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

It uses **direct MuJoCo `data.ctrl` writes**, bypassing MyoSuite action normalization, because excitation must be updated at *every physics step*. With no verified sensor names, the phase controller still works but event corrections are disabled. Supply `sensor_names=(RH, RT, LH, LT)` to enable real contact events.

`observation_space` currently holds a **temporary diagnostic vector** (qpos/qvel/act, raw contacts, MEP, phase context). It **is not** the agreed final ten-block observation, and reward is **exactly zero**, with fixed-time truncation and nonfinite-state exceptions. The final observations (without hip root-relative positions or linear velocities), imitation rewards, physical-failure/early termination and PPO training follow in Steps 4–7. **Do not train on this skeleton.**

Policy initialization for unmapped-muscle action means near `-0.9` is **deferred to PPO**, as agreed; the default disabled mode sends exactly zero regardless of action.

## Validation boundary

The test suite uses the **genuine uploaded 226-frame reference** plus synthetic MEP/activation lookup and a lightweight MuJoCo stand-in. The sample reference and its manifest are bundled under `tests/fixtures/`, so these tests work outside this conversation's workspace. They verify phase wrap/forward root displacement, RSI, 1-ms interpolation with held 5-ms residual, event hysteresis, deadband/completion, effective correction after clipping, assumed TO filtering, and model-dt validation.

**We have not run this against your actual MyoLeg XML or the real MEP/activation lookup**, which were not available in the attachments for this turn. In particular, real sensor naming, contact thresholds and gym runtime need verification on your machine. An actual training policy is not yet implemented.
