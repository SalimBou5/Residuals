# Single-source refactor (16 October 2026)

The previous version removed `gpu_components.py` but *retained duplicated equations* in `GaitPhaseEstimator` and `BatchedGaitPhaseEstimator`, and duplicated NumPy/Torch interpolation in `GaitReference`. This version fixes the duplication, rather than merely moving it.

- **`gait_phase.py`**: one `GaitPhaseEstimator` class with **one** tensor-based `update()` algorithm for a scalar walker or [N] parallel walkers. `num_envs` omitted preserves the user-facing scalar API, while `num_envs=N` yields batched GPU tensors. One `_frequency_tensor()` calculation, one confidence schedule, and one circular phase-error function. The user's PhaseConfig gains, alpha, cadence confidence rule, and stride bounds are retained. `FootContactEvents` is also one GRF-threshold crossing algorithm, including the exact 20 N rule and reset baseline. Optional CPU contact-force extraction is separate from threshold crossing by design.
- **`gait_reference.py`**: retains the user's `GaitReference` dataclass, `CHANNELS`, NPZ validation and name-based ordering. Removes separate NumPy `_periodic_interp` and Torch `at_phases`; `at_phase()` performs *one* cached PyTorch periodic interpolation for both float and [N] torch.Tensor inputs, with only the scalar output formatting differing.
- **`muscle_mapping.py`**: one mapping matrix multiplication for scalar dictionaries and batched tensors. Provisional 15→80 assignments unchanged.
- **`rl_env.py`**: uses `GaitPhaseEstimator(..., num_envs=N)` rather than a second estimator class. Reference muscle profiles still provide feedforward **commands**, policy outputs residuals, imitation rewards still concern joint kinematics only.
- **`test_skeleton.py`**: updated and extended to validate scalar/batched wrapper parity, periodic boundary interpolation, 20 N HS/TO, mapping, feedforward/residual, reward, and RSI interface.

**Scope:** no new RL algorithm; RSL-RL registration and live GPU physics remain untested/not implemented. The first detected HS anchors the beginning of a stride rather than being counted as a completed one (same as the previous package).
