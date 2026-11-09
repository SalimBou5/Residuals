# MyoLeg MEP-informed residual controller — implementation starter

**Completed here: Step 1 (reference loader) and Step 2 (RSI).** These are reusable components, not a complete Gymnasium environment. This starts from `test_rsi_mep_from_library.py`, preserving its interpolation, actuator-name consistency checks and one-time `data.act` initialization. The original test file is *not* modified.

## Files

- `gait_rl/reference.py`: one shared bilateral MEP CSV, one duration-indexed activation NPZ, **full `*_loopable.npz` preload by default** (optional bounded LRU), per-cycle manifest events, phase sampling and continuous root displacement. All reference arrays are decompressed once at startup; no per-episode NPZ reads.
- `gait_rl/rsi.py`: caches the model-dependent actuator/activation mapping, hierarchical reference sampling, random phase, lookup activation initialization, 15 mm floor-penetration rejection, baseline excitation and optional unmapped direct actions.
- `try_rsi.py`: run against your actual MyoLeg XML/MJB, MEP and activation files. Optionally reproduces a **MEP-only unconstrained physics rollout** (no RL, no phase synchronization).
- `tests/test_reference_rsi.py`: tests on the uploaded genuine 226-frame kinematic reference plus synthetic MEP/lookup and a MuJoCo stand-in.

## Directory layout expected for the 969 strides

```text
gait_with_landmarks/
    S011/
        Trial01/
            step01_loopable.npz
            step01_loopable_manifest.json
            step02_loopable.npz
            step02_loopable_manifest.json
    ...
myoleg_bilateral_mep_80.csv
activation_lookup.npz
```

The loader checks actual keys, not names proposed in earlier design docs. The NPZ contains `time_since_start_s`, `phase_right_01`, `qpos`, `qvel`, `site_pos_rel`, `site_quat_rel`, **`site_linvel_rel`**, `site_angvel_rel`, etc. The uploaded example has **226**, not 201, samples. The CSV and activation NPZ are only loaded once per `ReferenceAssets` instance.

### RAM cache strategy (updated)

By default, `ReferenceAssets(...)` constructs `ReferenceLibrary(...)` and **preloads all references once**, checking file shapes, timings, landmark data and manifests before training. When `RSIInitializer.reset()` selects a new stride, `get()` returns the already cached object without disk access. The cache also tracks `n_cached` and `cached_array_bytes` (numeric arrays only).

For the uploaded 226-sample reference, the arrays retained by our loader occupy **0.340 MB**. For 969 references of similar length this is **about 329 MB (314 MiB)**, excluding Python object/metadata overhead, the shared activation lookup and any model/simulation state. Actual library usage depends on each reference's length and should be read from the printed cache statistics.

Multiple independent worker processes **may duplicate** this RAM; on fork-based systems a parent preload before forking can share read-only array pages through copy-on-write, depending on the worker framework. Do not assume that sharing is automatic. If needed, select `--cache-mode lru --cache-size 8` to restore bounded lazy loading, or configure `ReferenceAssets(..., max_cached_references=8)`. 

## Run the real-model smoke test

Install your project's MuJoCo, NumPy and other dependencies in your existing environment, then from this starter directory:

```bash
python try_rsi.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\Users\BoussofaraS\Documents\residuals\data_extraction\Transform_cycles_to_episodes\gait_with_landmarks --mep C:\Users\BoussofaraS\Documents\residuals\data_extraction\GV-2015\outputs\data\myoleg_bilateral_mep_80.csv --lookup C:\Users\BoussofaraS\Documents\residuals\data_extraction\GV-2015\outputs\data\activation_profiles\activation_lookup.npz --resets 20 --seed 42
```

To test one phase in one reference, use `--references /path/to/S011/Trial01/step01_loopable.npz --phase 0.30 --resets 1`. You can add `--baseline-seconds 2` to reproduce a two-second **MEP-only physics** rollout after each RSI. This deliberately updates MEP every physics substep, as your validated script does, *not* the forthcoming 10 ms RL control loop.

```bash
python -m pytest -q
```

## Key semantics already fixed

1. **Unwrapped** `theta` is the future phase-clock state; `phase=theta % 1`. Sampling the reference at `theta=1.3` adds one `root_displacement_xyz_m` to the pelvis position. Joint and pelvis-relative site references loop, but global root position does **not** jump backward.
2. Reference velocities are returned unchanged for RSI; `reference.sample(theta, phase_rate=cycles_per_second)` optionally scales all velocities by `T_ref * phase_rate` for future time-warped rewards/observations. This is a physical phase rate, not a fractional RL action.
3. At reset, `qpos`, `qvel`, and **internal activation** are written once, followed by `mj_forward`. Lookups select the closest precomputed duration with a strict 5 ms tolerance (same as the original test). Normal steps must NEVER overwrite `data.act`.
4. The initial `data.ctrl` defaults to **zero** to preserve the frozen state of the tested script. `--initialize-ctrl mep` is available for a baseline-initialized forward calculation. Either way, on the first actual step, apply `binding.baseline(theta)` or, later, `binding.excitation(theta, policy_action, limits)`.
5. The unmapped muscles are `piri`, `grac`, `edl`, `ehl`, `fhl` on **both legs**. Their baseline excitation, control, and initial `data.act` are **zero by default**. Use `--enable-unmapped` to allow direct control in the subsequent RL action builder; then `(a+1)/2` gives excitation 0.05 for a bounded action `a=-0.9`. The action builder is present but no PPO initialization is assumed. The future PPO policy's distribution mean/std need separate attention.
6. Random reference choice is **subject -> trial -> stride**, then uniform random phase within the chosen stride; phases with >15 mm of initial floor penetration are rejected and re-sampled without modifying the retargeted pose.
7. The manifest's event phases **and source strings** are retained, including the distinction between *observed LHS* and *assumed toe-offs*. No false measured-TO claim is made.
8. No kinematic tether is applied. This remains unconstrained MyoLeg physics when steps begin.

## Deliberately deferred — phase and contact handling

The phase controller is **not skipped**. It should be the next standalone, tested component before building the Gymnasium control loop:

- State `theta`, `theta_nom`, remaining event correction, correction-active flag, and contact history, initialized at RSI as `(phi0, phi0, 0, False)`.
- After each **0.01 s control step** (five 0.002 s physics steps), update phase using `dt/T_ref * (1+kappa_max*a_phi)` and the buffered event correction; clip progression to a positive, bounded increment.
- Detect events from **verified MyoLeg contact sensors** (hysteresis, refractory interval, plausible sequence). Seed the detector at RSI using current contact state, so an existing contact cannot trigger a false HS.
- For a new event, compare the observed phase with the corresponding manifest phase (RHS=0, observed LHS when present; TO values explicitly flagged as assumptions), and dissipate the correction gradually using **actual applied correction** after clipping, not the unmodified requested correction. Keep distinct activation vs completion tolerances.
- All baseline commands and reference targets for control step `k` use the same current phase `phi_k`. Following physics integration, rewards/observations use the new `phi_(k+1)` and the **actual phase rate**.

`RSIInitializer` already accepts a `contact_reader(model, data)` callback and stores its result in `ResetResult.contact_seed`; to avoid inventing heel/toe sensor names, that callback is not implemented until we inspect the actual model and verify the four contact signals.

**Validation boundary:** The unit tests pass on the provided real NPZ and synthetic MEP/activation fixtures. The real MuJoCo model, shared MEP CSV and activation lookup were not provided in this turn, so full-model resets, penetration rejection and rollout still require the commands above in your existing environment.
