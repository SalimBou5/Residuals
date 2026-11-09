"""Step 5: physical-unit gait imitation and MEP-residual reward.

Implements Section IV of 'RL Implemetation details.pdf'. Temperatures and
weights are explicit *initial calibration guesses*, not fitted performance
claims; use inspect_reference_rewards.py before tuning them.

No normalized observation enters this module. Site packing is the published
Step-4 layout: positions 6x3, velocities 6x3, quaternions 8x4,
angular velocities 8x3. All 8 orientations include hip orientation, but
hip POSITIONS and LINEAR VELOCITIES are NOT rewarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np

from observations import POSITION_SITES, ORIENTATION_SITES


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest orientation error in radians, invariant to q -> -q."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError('Orientation arrays must have matching last dimension 4')
    an = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return 2 * np.arccos(np.clip(np.abs(np.sum(an * bn, axis=-1)), 0, 1))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector from body to world, scalar-first quaternion."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    xyz = q[..., 1:]
    t = 2 * np.cross(xyz, v)
    return v + q[..., :1] * t + np.cross(xyz, t)


@dataclass(frozen=True)
class RewardConfig:
    # Scores are exp(-beta * mean_squared_error). Units stated below.
    # Half-score errors were revised after a 2,400-case static FK perturbation
    # scan. They remain provisional until evaluated on dynamic PPO rollouts.
    beta_joint_position: float = float(np.log(2) / .15**2)        # 0.15 rad RMS
    beta_joint_velocity: float = float(np.log(2) / 2.0**2)        # 2 rad/s RMS
    beta_site_position: float = float(np.log(2) / .12**2)         # 12 cm site Euclidean RMS
    beta_site_orientation: float = float(np.log(2) / .30**2)      # 0.30 rad site RMS
    beta_site_linear_velocity: float = float(np.log(2) / 1.5**2)  # 1.5 m/s site Euclidean RMS
    beta_site_angular_velocity: float = float(np.log(2) / 3.0**2)# 3 rad/s site Euclidean RMS
    beta_speed: float = float(np.log(2) / .35**2)                 # 0.35 m/s error
    beta_upright: float = float(np.log(2) / (1-np.cos(np.deg2rad(18)))) # 18 deg tilt

    # Imitation terms sum to 1. Speed/upright are additive, as in design PDF.
    w_joint_position: float = .20
    w_joint_velocity: float = .10
    w_site_position: float = .28
    w_site_orientation: float = .14
    w_site_linear_velocity: float = .18
    w_site_angular_velocity: float = .10
    w_speed: float = .20
    w_upright: float = .45

    lambda_residual: float = .50
    lambda_phase: float = .05
    # Fivefold stronger than the provisional V1 activation costs. These act
    # on ABSOLUTE activation, not on a deviation from the MEP prior.
    lambda_activation_l1: float = .15
    lambda_activation_l2: float = .10
    # Optionally weight experimentally mapped channels differently: 80 actuator
    # order, strictly nonnegative. Default all equal; do not invent reliability.
    residual_muscle_weights: tuple[float, ...] | None = None
    upright_body_name: str = 'pelvis'   # same body as root observation; override if needed
    upright_local_axis: tuple[float, float, float] = (0., 1., 0.)
    # In this MyoLeg model, the pelvis BODY's local +Y points upward in
    # healthy gait. Using local +Z would reward an unnatural orientation.

    def __post_init__(self) -> None:
        for k,v in vars(self).items():
            if k.startswith('beta_'):
                _positive(k, v)
            elif k.startswith(('w_', 'lambda_')):
                if not np.isfinite(v) or v < 0:
                    raise ValueError(f'{k} must be finite and nonnegative')
        imsum = sum(getattr(self, 'w_' + k) for k in (
            'joint_position', 'joint_velocity', 'site_position',
            'site_orientation', 'site_linear_velocity', 'site_angular_velocity'))
        if abs(imsum - 1.0) > 1e-8:
            raise ValueError(f'Imitation weights must sum to 1 (got {imsum})')
        axis=np.asarray(self.upright_local_axis,float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1.,atol=1e-8):
            raise ValueError('upright_local_axis must be a unit 3-vector')
        if self.residual_muscle_weights is not None:
            w=np.asarray(self.residual_muscle_weights, float)
            if w.shape != (80,) or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError('residual_muscle_weights must be a nonnegative 80-tuple')


@dataclass(frozen=True)
class TerminationConfig:
    """Conservative trial safeguards. Tracking thresholds are NOT calibrated by replay."""
    min_pelvis_height_m: float = .55
    min_upright_dot: float = 0.0
    max_mean_site_error_m: float = .50
    max_world_root_error_m: float = 1.0
    max_root_orientation_error_rad: float = 1.5
    tracking_grace_control_steps: int = 5
    enable_tracking_termination: bool = True
    enable_physical_termination: bool = True

    def __post_init__(self):
        for k in ('min_pelvis_height_m', 'max_mean_site_error_m',
                  'max_world_root_error_m', 'max_root_orientation_error_rad'):
            _positive(k, getattr(self, k))
        if not -1 <= self.min_upright_dot <= 1:
            raise ValueError('min_upright_dot must be in [-1,1]')
        if self.tracking_grace_control_steps < 0:
            raise ValueError('tracking_grace_control_steps must be >= 0')


def component_errors(blocks: Mapping[str, np.ndarray], reference_qvel: np.ndarray
                     ) -> dict[str, float]:
    """Tracking errors from *raw* Step-4 blocks, in SI units.

    Reference velocity must already include the actual phase-rate time warp.
    """
    q = np.asarray(blocks['joint_proprioception'], float)
    qr = np.asarray(blocks['reference_joints'], float)
    vref = np.asarray(reference_qvel, float)
    if q.shape != (56,) or qr.shape != (28,) or vref.shape != (34,):
        raise ValueError('Expected 28 MyoLeg nonroot joint DoFs and nv=34')
    a = np.asarray(blocks['actual_sites'], float)
    b = np.asarray(blocks['reference_sites'], float)
    if a.shape != (92,) or b.shape != (92,):
        raise ValueError('Site block must be 92-D, without hip position/velocity')
    na, nv = len(POSITION_SITES), len(ORIENTATION_SITES)
    if na != 6 or nv != 8:
        raise ValueError('Site packing changed; update reward definitions')
    site_p = (a[:18] - b[:18]).reshape(na, 3)
    site_v = (a[18:36] - b[18:36]).reshape(na, 3)
    site_q = quat_angle(a[36:68].reshape(nv,4), b[36:68].reshape(nv,4))
    site_w = (a[68:92] - b[68:92]).reshape(nv,3)
    # Average squared Euclidean norm *per site*, not per Cartesian scalar;
    # joint errors are mean squared per DoF.
    result = {
        'joint_position': float(np.mean((q[:28]-qr)**2)),
        'joint_velocity': float(np.mean((q[28:]-vref[6:])**2)),
        'site_position': float(np.mean(np.sum(site_p**2, axis=1))),
        'site_orientation': float(np.mean(site_q**2)),
        'site_linear_velocity': float(np.mean(np.sum(site_v**2, axis=1))),
        'site_angular_velocity': float(np.mean(np.sum(site_w**2, axis=1))),
    }
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError('Nonfinite imitation error')
    return result


def score_from_errors(errors: Mapping[str, float], cfg: RewardConfig) -> dict[str, float]:
    return {k: float(np.exp(-getattr(cfg, 'beta_'+k) * x)) for k,x in errors.items()}


def regularization_costs(data_act: np.ndarray, action_controller: Any,
                         kappa_max: float, cfg: RewardConfig) -> dict[str,float]:
    """Policy *requested* residual, policy phase-rate, and actual muscle states.

    Event corrections never enter phase penalty. Command clipping does NOT
    change requested residual cost; actual applied residual is logged separately.
    """
    if not np.isfinite(kappa_max) or kappa_max < 0:
        raise ValueError('Invalid phase rate bound')
    a = np.asarray(data_act, float)
    if a.shape != (80,) or not np.isfinite(a).all():
        raise ValueError('Pass actual 80 muscle activations in actuator order')
    mask = np.asarray(action_controller.binding.mapped_mask, dtype=bool)
    residual = np.asarray(action_controller.residual_limits, float) * np.asarray(action_controller.actuator_action, float)
    if residual.shape != (80,) or mask.shape != (80,):
        raise ValueError('Unexpected actuator vector/mask')
    weights = (np.ones(80) if cfg.residual_muscle_weights is None
               else np.asarray(cfg.residual_muscle_weights, float))
    denom = float(np.sum(weights[mask]))
    if denom <= 0:
        raise ValueError('Residual muscle weights must be nonzero on mapped muscles')
    return {
        'residual': float(np.sum(weights[mask]*residual[mask]**2)/denom),
        'phase': float((kappa_max*action_controller.phase_action)**2),
        'activation_l1': float(np.mean(np.abs(a))),
        'activation_l2': float(np.mean(a*a)),
    }


def combine_reward(errors: Mapping[str,float], speed_m_s: float,
                   speed_target_m_s: float, upright_dot: float,
                   regularizers: Mapping[str,float], cfg: RewardConfig
                   ) -> tuple[float,dict[str,float]]:
    scores = score_from_errors(errors, cfg)
    speed_error = float(speed_m_s - speed_target_m_s)
    scores['speed'] = float(np.exp(-cfg.beta_speed * speed_error**2))
    scores['upright'] = float(np.exp(-cfg.beta_upright * (1-np.clip(upright_dot,-1.,1.))))
    terms: dict[str,float] = {}
    imitation = 0.0
    for k in errors:
        terms['error_'+k] = errors[k]
        terms['score_'+k] = scores[k]
        value = getattr(cfg, 'w_'+k) * scores[k]
        terms['contribution_'+k] = value
        imitation += value
    terms['imitation'] = float(imitation)
    terms['error_speed_m_s'] = speed_error
    terms['score_speed'] = scores['speed']
    terms['score_upright'] = scores['upright']
    terms['upright_dot'] = float(upright_dot)
    terms['contribution_speed'] = cfg.w_speed * scores['speed']
    terms['contribution_upright'] = cfg.w_upright * scores['upright']
    total_penalty = 0.0
    for k, w in [('residual', cfg.lambda_residual), ('phase', cfg.lambda_phase),
                 ('activation_l1', cfg.lambda_activation_l1),
                 ('activation_l2', cfg.lambda_activation_l2)]:
        terms['cost_'+k] = float(regularizers[k])
        terms['penalty_'+k] = float(w*regularizers[k])
        total_penalty += terms['penalty_'+k]
    terms['total_penalty'] = float(total_penalty)
    reward = float(imitation + terms['contribution_speed'] + terms['contribution_upright'] - total_penalty)
    terms['total'] = reward
    if not np.isfinite(reward):
        raise FloatingPointError('Nonfinite reward')
    return reward,terms


class GaitRewardEvaluator:
    """Pure reward definitions + minimal MyoLeg state-dependent extraction."""
    def __init__(self, model: Any, mujoco: Any, obs_builder: Any,
                 config: RewardConfig = RewardConfig(),
                 termination: TerminationConfig = TerminationConfig()):
        self.model,self.mj,self.obs,self.cfg,self.term = model,mujoco,obs_builder,config,termination
        self.up_id = int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,config.upright_body_name))
        if self.up_id < 0:
            raise ValueError(f'Missing upright body {config.upright_body_name!r}')

    def evaluate(self, data: Any, clock: Any, controller: Any,
                 control_steps: int) -> tuple[float,dict[str,float],bool,str|None]:
        """Call after observation.build() at the same post-physics clock phase."""
        ref = self.obs.reference
        sample = ref.sample(clock.theta, phase_rate=clock.phase_rate_hz)
        blocks = self.obs.last_blocks     # RAW values: invariant to observation normalizer
        errors = component_errors(blocks,sample.qvel)
        root = np.asarray(blocks['root_state'],float)
        root_quat = root[1:5]
        local_vel = root[5:8]
        world_velocity = quat_rotate(root_quat,local_vel)
        forward = np.asarray(ref.root_displacement[:2], dtype=float)
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm <= 1e-6:
            raise ValueError('Cannot infer walking direction from zero root stride displacement')
        # A loaded reference may expose read-only arrays. Never mutate them.
        forward_unit = forward / norm
        speed = float(world_velocity[:2] @ forward_unit)
        up_axis=np.asarray(self.cfg.upright_local_axis,float)
        upright = float((np.asarray(data.xmat[self.up_id],float).reshape(3,3) @ up_axis)[2])
        # Gather all 80 actual activation states in ACTUATOR order, not a
        # compact observed subset, and never confuse activation with excitation.
        addrs = self.obs.activation_addresses
        act = np.asarray(data.act[addrs],float)
        costs = regularization_costs(act,controller,clock.config.kappa_max,self.cfg)
        value, terms = combine_reward(errors,speed,float(ref.desired_speed),upright,costs,self.cfg)
        terms['actual_forward_speed_m_s'] = speed
        terms['desired_forward_speed_m_s'] = float(ref.desired_speed)  # exactly gait_phase[0]
        terms['actual_applied_residual_rms'] = float(np.sqrt(np.mean(controller.applied_residual[controller.binding.mapped_mask]**2)))

        # Loose physical/trajectory failure tests, logged independently from reward.
        root_ref = np.asarray(sample.qpos[:3],float)
        root_real = np.asarray(data.qpos[:3],float)
        root_error = float(np.linalg.norm(root_real-root_ref))
        qroot_error = float(quat_angle(np.asarray(data.qpos[3:7],float),sample.qpos[3:7]))
        site_delta = np.asarray(blocks['actual_sites'][:18],float) - np.asarray(blocks['reference_sites'][:18],float)
        mean_site_error = float(np.mean(np.linalg.norm(site_delta.reshape(6,3),axis=1)))
        terms['world_root_position_error_m'] = root_error
        terms['world_root_orientation_error_rad'] = qroot_error
        terms['mean_site_position_error_m'] = mean_site_error
        reason = None
        if self.term.enable_physical_termination:
            if root[0] < self.term.min_pelvis_height_m:
                reason='pelvis_too_low'
            elif upright < self.term.min_upright_dot:
                reason='upside_down'
        if reason is None and self.term.enable_tracking_termination and control_steps > self.term.tracking_grace_control_steps:
            if mean_site_error > self.term.max_mean_site_error_m:
                reason='site_tracking_error'
            elif root_error > self.term.max_world_root_error_m:
                reason='world_root_tracking_error'
            elif qroot_error > self.term.max_root_orientation_error_rad:
                reason='root_orientation_tracking_error'
        return value,terms,bool(reason),reason
