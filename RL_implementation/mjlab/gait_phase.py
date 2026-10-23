"""Event-corrected gait phase for one walker or parallel MyoSuite 3 walkers.

ONE vectorized implementation drives both APIs. With ``num_envs`` omitted,
``GaitPhaseEstimator`` accepts bool events and returns Python scalar properties.
With ``num_envs=N``, it accepts PyTorch tensors [N] and stays on the GPU.

Phase is in [0, 1), and frequency is in cycles/second, NOT radians/second.
The confidence schedule, gains and stride bounds follow the user's edited code.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class PhaseConfig:
    reference_stride_s: float
    toe_off_phase: float  # From the reference gait, not a universal constant.
    alpha: float = 0.3  # Smoothed stride-duration update.
    k_hs: float = 1  # Heel-strike phase correction.
    k_to: float = 0.5  # Toe-off phase correction.
    min_stride_s: float = 0.5  # Allowed ratio of measured/reference stride.
    max_stride_s: float = 2

    def __post_init__(self):
        if self.reference_stride_s <= 0:
            raise ValueError('reference_stride_s must be positive')
        if not 0.0 <= self.toe_off_phase < 1.0:
            raise ValueError('toe_off_phase must be in [0, 1)')
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError('alpha must be in (0, 1]')
        if not (0 <= self.k_hs <= 1 and 0 <= self.k_to <= 1):
            raise ValueError('Event-correction gains must be in [0, 1]')
        if not (0 < self.min_stride_s < self.max_stride_s):
            raise ValueError('Invalid stride-time sanity bounds')


def circular_error(target, current):
    """Shortest signed difference in gait *cycles*; supports tensors or floats."""
    return (target - current + 0.5) % 1.0 - 0.5


class GaitPhaseEstimator:
    """One estimator algorithm, operating on a tensor of one or more walkers.

    Scalar mode: ``GaitPhaseEstimator(config)`` gives Python scalar public state.
    Parallel mode: ``GaitPhaseEstimator(config, num_envs=N, device='cuda:0')``
    gives [N] tensor public state. Both invoke the SAME ``update``.
    """

    def __init__(self, config: PhaseConfig, num_envs: int | None = None,
                 device: str | torch.device = 'cpu'):
        self.cfg = config
        self._scalar = num_envs is None
        self.num_envs = 1 if num_envs is None else int(num_envs)
        if self.num_envs <= 0:
            raise ValueError('num_envs must be positive')
        self.device = torch.device(device)
        self.initial_phase = 0.0  # Used when a mjlab walker auto-resets.

        '''
        - self.phase : current phase
        - self.elapsed_s : time elapsed in the current stride
        - self.smoothed_stride_s : smoothed stride duration 
        - self.valid_strides : how many acceptable consecutive strides
        - self.confidence : how much the estimate is trusted (will evolve in time)
        - self.last_stride_s : duration of the last stride
        '''
        self._phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._elapsed_s = torch.zeros_like(self._phase)
        self._smoothed_stride_s = torch.zeros_like(self._phase)
        self._valid_strides = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._confidence = torch.zeros_like(self._phase)
        self._last_stride_s = torch.full_like(self._phase, float('nan'))
        self._has_heel_strike = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    def _show(self, value):
        if not self._scalar:
            return value
        v = value[0].item()
        return None if isinstance(v, float) and np.isnan(v) else v

    @property
    def phase(self):
        return self._show(self._phase)

    @property
    def elapsed_s(self):
        return self._show(self._elapsed_s)

    @property
    def smoothed_stride_s(self):
        return self._show(self._smoothed_stride_s)

    @property
    def valid_strides(self):
        return self._show(self._valid_strides)

    @property
    def confidence(self):
        return self._show(self._confidence)

    @property
    def last_stride_s(self):
        return self._show(self._last_stride_s)

    @property
    def has_heel_strike(self):
        return self._show(self._has_heel_strike)

    def _bool_vector(self, value, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.dtype != torch.bool or value.device != self.device or value.shape != (self.num_envs,):
                raise ValueError(f'{label} must be a bool tensor [{self.num_envs}] on {self.device}')
            return value
        if self._scalar and isinstance(value, (bool, np.bool_)):
            return torch.tensor([bool(value)], device=self.device, dtype=torch.bool)
        raise ValueError(f'{label} must be a bool tensor [{self.num_envs}] on {self.device}')

    def _reset_mask(self, mask):
        if mask is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self._bool_vector(mask, 'reset mask')

    def reset(self, mask: torch.Tensor | None = None, initial_phase: float | torch.Tensor = 0.0):
        mask = self._reset_mask(mask)
        initial = torch.as_tensor(initial_phase, device=self.device, dtype=self._phase.dtype)
        if initial.ndim == 0:
            initial = initial.expand(self.num_envs)
        if initial.shape != (self.num_envs,):
            raise ValueError(f'initial_phase must be scalar or [{self.num_envs}]')
        # torch.where avoids host synchronization when resetting a subset of GPU envs.
        self._phase = torch.where(mask, torch.remainder(initial, 1.0), self._phase)
        self._elapsed_s = torch.where(mask, 0.0, self._elapsed_s)
        self._smoothed_stride_s = torch.where(mask, self.cfg.reference_stride_s, self._smoothed_stride_s)
        self._last_stride_s = torch.where(mask, float('nan'), self._last_stride_s)
        self._valid_strides = torch.where(mask, 0, self._valid_strides)
        self._confidence = torch.where(mask, 0.0, self._confidence)
        self._has_heel_strike = torch.where(mask, False, self._has_heel_strike)

    def _frequency_tensor(self):
        # The user's confidence blend: reference to estimated cadence.
        '''
            This function defines the phase rate dot{phi_t} that will be the base of the evolution of the gait phase 
            (phi_{t+1} = phi_t + dot{phi_t} * control_timestep)
            phi_t should ideally represent the gait phase in which our model is which is not necessarily the reference one
            --> Therefore dot{phi_t} should ideally represent the phase rate of the model.
            However, the model will start from a completely unstable gait and should learn until finding a stable gait.
            When it is unstable, we cannot rely on the model phase 
                --> solution : consider the reference one (which is the same as giving non-normalized reference kinematics and muscle activations)
                --> But then it's too rigid and we lose the real gait phase we want to get
                    --> Solution: as soon as we can trust the model we take it
            --> FINAL SOLUTION: a confidence blending of both rates (reference and estimated from the model) 
            At first confidence will be 0, then it will improve depending in the how many strides we have and how much time does it take. 
        '''
        f_ref = 1.0 / self.cfg.reference_stride_s  # reference frequency
        f_est = 1.0 / self._smoothed_stride_s      # estimated frequency from the smoothed stride duration
        return (1.0 - self._confidence) * f_ref + self._confidence * f_est

    @property
    def frequency_hz(self):
        return self._show(self._frequency_tensor())

    def _update_confidence(self, completed: torch.Tensor):
        '''
        Update the confidence based on the number of valid strides and the timing error.
        '''
        # EXACT confidence schedule from the user's modified gait_phase.py:
        # c_strides = 0.1*n if n < 4 else 0.5
        # timing: >.4->0; >.3->.2; >.15->.3; otherwise->.5
        n = self._valid_strides
        c_strides = torch.where(n < 4, n.float()*0.1, 0.5)
        e_t = torch.abs(self._smoothed_stride_s - self.cfg.reference_stride_s)
        c_timing = torch.where(e_t > 0.4, 0.0,
                   torch.where(e_t > 0.3, 0.2,
                   torch.where(e_t > 0.15, 0.3, 0.5)))
        self._confidence = torch.where(completed, 
                                       torch.clamp(c_strides + c_timing, max=1.0), 
                                       self._confidence)

    def update(self, dt: float, heel_strike=False, toe_off=False,
               done: torch.Tensor | None = None):
        """Advance and correct phase; HS and TO are *edges* from the same foot.
        One call handles scalar and batched inputs
        """
        if dt <= 0:
            raise ValueError('dt must be positive')
        hs = self._bool_vector(heel_strike, 'heel_strike')
        to = self._bool_vector(toe_off, 'toe_off')
        if done is None:
            done = torch.zeros_like(hs)
        else:
            done = self._bool_vector(done, 'done')
        if self._scalar and bool((hs & to).any()):
            raise ValueError('Heel strike and toe off cannot be simultaneous')
        hs = hs & ~done
        to = to & ~done

        self._phase = torch.remainder(self._phase + self._frequency_tensor() * dt, 1.0) # phi_{t+1} = phi_t + dot{phi_t} * control_timestep
        self._elapsed_s += dt

        # A stride is considered valid if the stride time is reasonable (min * ref_time < stride_time < max * ref_time, with min = 0.5 and max = 2)
        lo = self.cfg.min_stride_s * self.cfg.reference_stride_s
        hi = self.cfg.max_stride_s * self.cfg.reference_stride_s
        completed = hs & self._has_heel_strike
        valid = completed & (self._elapsed_s >= lo) & (self._elapsed_s <= hi)
        invalid = completed & ~valid

        # We update the last stride duration only if the stride is valid, otherwise we set it to NaN.
        # We need to go through the invalid case because something that is not valid can simply be an incomplete stride
        # In the case of incomplete stride we don't want to update the last stride duration, we want to keep it as it was before.
        self._last_stride_s = torch.where(valid, self._elapsed_s, self._last_stride_s)
        self._last_stride_s = torch.where(invalid, float('nan'), self._last_stride_s)

        # Update the stride time in a smoothed way, *a* determines how fast the stride time estimate adapts to the new value
        a = self.cfg.alpha
        self._smoothed_stride_s = torch.where(valid, 
                                              (1.0 - a)*self._smoothed_stride_s + a*self._elapsed_s,
                                              self._smoothed_stride_s)
        self._valid_strides = torch.where(valid, self._valid_strides + 1, self._valid_strides)
        self._valid_strides = torch.where(invalid, 0, self._valid_strides)

        self._update_confidence(completed)

        self._has_heel_strike = self._has_heel_strike | hs
        self._elapsed_s = torch.where(hs, 0.0, self._elapsed_s)

        # Same circular-error formula for HS and TO, and both backends.
        hs_corrected = self._phase + self.cfg.k_hs * circular_error(0.0, self._phase)
        to_corrected = self._phase + self.cfg.k_to * circular_error(self.cfg.toe_off_phase, self._phase)
        self._phase = torch.where(hs, hs_corrected, torch.where(to, to_corrected, self._phase))
        self._phase = torch.remainder(self._phase, 1.0)

        # mjlab auto-resets done walkers. Clear their estimator histories, with
        # the configurable reset phase. Other walkers retain their histories.
        self.reset(mask=done, initial_phase=self.initial_phase)
        return self.phase

# #######################################################################################
# !!!!!!!!!!!!!!!!   CHECK, I AM NOT CONVINCED OF HOW THE FORCE IS MEASURED !!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class FootContactEvents:
    """One 20 N threshold-crossing algorithm for CPU or batched GPU GRF data.

    For mjlab, the adapter supplies sensor-based foot GRF in Newtons.
    For CPU MuJoCo, an optional model lets ``update(data)`` extract world-Z GRF
    directly from collision contacts. First sample establishes the baseline.
    """
    # The threshold 20.0 N follows the HS and TO extraction of the Gait120 dataset
    def __init__(self, model=None, side='right', threshold_N=20.0,
                 num_envs: int | None = None, device='cpu'):
        if side not in ('right', 'left'):
            raise ValueError("side must be 'right' or 'left'")
        if threshold_N <= 0:
            raise ValueError('threshold_N must be positive')
        self.model = model
        self.side = side
        self.threshold_N = float(threshold_N)
        self._scalar = num_envs is None
        self.num_envs = 1 if num_envs is None else int(num_envs)
        if self.num_envs <= 0:
            raise ValueError('num_envs must be positive')
        self.device = torch.device(device)
        self._foot_geom_ids = set()
        self._ground_geom_ids = set()
        if model is not None:
            import mujoco
            short = 'r' if side == 'right' else 'l'

            # !!! TO BE CHECKED: the names of the foot collision geometries in the model must match these names !!!
            # !!!!!!!!! CHECK : MAYBE we need to consider other sensors - same for floor , terrain !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
            foot_names = (f'{short}_foot_col1', f'{short}_foot_col3',
                          f'{short}_foot_col4', f'{short}_bofoot_col1',
                          f'{short}_bofoot_col2')
            for name in foot_names:
                idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if idx >= 0:
                    self._foot_geom_ids.add(int(idx))
            for name in ('floor', 'terrain'):
                idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if idx >= 0:
                    self._ground_geom_ids.add(int(idx))
            if not self._foot_geom_ids or not self._ground_geom_ids:
                raise ValueError('Could not resolve foot collision geometries or ground geometry')
        self.previous_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._initialized = torch.zeros_like(self.previous_contact)
        self.normal_grf_N = torch.full((self.num_envs,), float('nan'), device=self.device)

    def _as_forces(self, x):
        if self.model is not None and not isinstance(x, (int, float, np.number, torch.Tensor)):
            x = self.measure_vertical_grf(x)
        force = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if force.ndim == 0 and self._scalar:
            force = force.reshape(1)
        if force.shape != (self.num_envs,):
            raise ValueError(f'GRF must have shape [{self.num_envs}]')
        if self._scalar and (not bool(torch.isfinite(force).all()) or bool((force < 0).any())):
            raise ValueError('GRF must be nonnegative and finite')
        return force

    def reset(self, mask=None, initial_grf_N=None):
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        elif not isinstance(mask, torch.Tensor) or mask.shape != (self.num_envs,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError(f'reset mask must be bool [{self.num_envs}] on {self.device}')
        if initial_grf_N is None:
            initial_contact = torch.zeros_like(mask)
            initialized = torch.zeros_like(mask)
            force = torch.full_like(self.normal_grf_N, float('nan'))
        else:
            force = self._as_forces(initial_grf_N)
            initial_contact = force >= self.threshold_N
            initialized = torch.ones_like(mask)
        self.previous_contact = torch.where(mask, initial_contact, self.previous_contact)
        self._initialized = torch.where(mask, initialized, self._initialized)
        self.normal_grf_N = torch.where(mask, force, self.normal_grf_N)

    def update(self, normal_grf_N):
        """HS if <20 -> >=20 N; TO if >=20 -> <20 N (one algorithm)."""
        force = self._as_forces(normal_grf_N)
        contact = force >= self.threshold_N
        hs = self._initialized & contact & ~self.previous_contact
        to = self._initialized & ~contact & self.previous_contact
        self.previous_contact = contact
        self._initialized = torch.ones_like(contact)
        self.normal_grf_N = force
        if self._scalar:
            return bool(hs[0].item()), bool(to[0].item())
        return hs, to

    def measure_vertical_grf(self, data) -> float:
        """Optional CPU MuJoCo contact-force extraction; *not* GPU hot path."""
        if self.model is None:
            raise ValueError('CPU MuJoCo model is required')
        import mujoco
        total = 0.0
        wrench = np.zeros(6, dtype=np.float64)
        for idx in range(int(data.ncon)):
            contact = data.contact[idx]
            if not ((contact.geom1 in self._foot_geom_ids and contact.geom2 in self._ground_geom_ids)
                    or (contact.geom2 in self._foot_geom_ids and contact.geom1 in self._ground_geom_ids)):
                continue
            mujoco.mj_contactForce(self.model, data, idx, wrench)
            frame = np.asarray(contact.frame).reshape(3, 3)
            world_force = frame.T @ wrench[:3]
            total += abs(float(world_force[2]))
        return total
