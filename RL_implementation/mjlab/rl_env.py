"""Phase-indexed imitation-learning environment on MyoSuite 3/mjlab.

Stage 1: MyoSuite standard reset, no RSI; CPU-independent GPU inference and
tracking rewards. This is an algorithm-agnostic experiment wrapper, NOT YET
registered as an RSL-RL training task. See README for integration caveat.
"""
from __future__ import annotations

import torch

from gait_phase import PhaseConfig, GaitPhaseEstimator, FootContactEvents
from gait_reference import GaitReference
from muscle_mapping import MuscleMapper
from imitation_reward import ImitationReward, ImitationRewardConfig
from myosuite3_adapter import MyoSuite3LegAdapter
from rsi import RSIConfig, RSISampler


class GaitRLEnv:
    def __init__(
        self, reference: GaitReference, phase_config: PhaseConfig, *,
        reward_config: ImitationRewardConfig | None = None,
        rsi_config: RSIConfig | None = None,
        residual_scale: float = 0.05,
        num_envs: int = 1024, device: str = 'cuda:0', sim=None,
    ):
        self.sim = sim if sim is not None else MyoSuite3LegAdapter(num_envs=num_envs, device=device)
        self.device = torch.device(self.sim.device)
        self.num_envs = self.sim.num_envs
        self.dt = self.sim.dt
        self.reference = reference.to(self.device)
        self.mapper = MuscleMapper(self.sim.actuator_names, self.device)
        self.phase_estimator = GaitPhaseEstimator(phase_config, num_envs=self.num_envs, device=self.device)
        self.contact_events = FootContactEvents(threshold_N=20.0, num_envs=self.num_envs, device=self.device)
        self.rsi_config = rsi_config or RSIConfig()
        self.rsi_sampler = RSISampler(self.rsi_config, self.reference, self.device)
        self.phase_estimator.initial_phase = self.rsi_config.initial_phase
        self.imitation = ImitationReward(reward_config or ImitationRewardConfig(),
                                         self.reference, self.mapper, self.sim)
        if residual_scale < 0:
            raise ValueError('residual_scale must be >= 0')
        self.residual_scale = residual_scale
        self.action_dim = len(self.sim.actuator_names)
        self.policy = None  # Trainer-independent

    def reset(self):
        if self.rsi_config.mode != 'standard':
            raise NotImplementedError(
                f'{self.rsi_config.mode} RSI reset is deliberately disabled until '
                'the mjlab physics-state restoration is implemented and tested')
        raw_obs, extras = self.sim.reset()
        self.phase_estimator.reset(initial_phase=self.rsi_config.initial_phase)
        self.contact_events.reset(initial_grf_N=self.sim.get_vertical_grf('r'))
        targets = self.reference_targets()
        return self.policy_observation(raw_obs, targets), {
            'phase': self.phase_estimator.phase.clone(),
            'reference': targets,
            'raw_myosuite_info': extras,
        }

    def reference_targets(self) -> dict:
        '''Return the reference targets at the current phase, including left-leg and feedforward expansions.
            The final dictionary has the following keys and shapes:
            {
                'muscles_15':              [B, 15],
                'muscles_15_left':         [B, 15],
                'feedforward_muscles_80':  [B, 80],
                'joint_angles':            [B, D] or None,
                'joint_velocities':        [B, D] or None,
                'provisional_actuators':   list of actuator names,
                'phase':                   [B],
                'confidence':              [B],
            }
        '''
        phase = self.phase_estimator.phase
        ref = self.reference.at_phase(phase)   # right leg reference; It is queried at the current phase.
        # for batched phases, ref will be a dictionary with the following keys:
        # {
        #     'muscles_15':       [B, 15],
        #     'joint_angles':     [B, D] or None,
        #     'joint_velocities': [B, D] or None,
        # }
        left = self.reference.at_phase(torch.remainder(phase + 0.5, 1.0))['muscles_15'] # The left leg is phase-shifted by 0.5 
        base = self.mapper.expand(ref['muscles_15'], left)
        return {**ref, 'muscles_15_left': left, 'feedforward_muscles_80': base,
                'provisional_actuators': self.mapper.provisional,
                'phase': phase.clone(),
                'confidence': self.phase_estimator.confidence.clone()}

    def action_to_excitation(self, action: torch.Tensor, base: dict):
        """Excitation = clip(mapped muscle drive + scaled policy residual, 0, 1).

        The processed 15 activation profiles define the feedforward command.
        The policy DOES NOT emit direct muscle excitations, and those profiles
        are NOT targets in the imitation reward. Every muscle has a provisional
        mapped base command; residuals are bounded by residual_scale.
        """
        if action.shape != (self.num_envs, self.action_dim) or action.device != self.device:
            raise ValueError(f'Need actions [{self.num_envs},{self.action_dim}] on {self.device}')

        ##### WHY TANH !!!!!!!!!!!!!!!
        return torch.clamp(base + self.residual_scale * torch.tanh(action), 0.0, 1.0)

    # REDO !!!!!!!!!!!!!!!!!!!!!!!!
    def policy_observation(self, sim_obs: dict, targets: dict):
        """Provisional algorithm-neutral policy input; can be redesigned later."""
        actor = sim_obs['actor']
        phase = targets['phase']
        features = [actor, torch.sin(2 * torch.pi * phase)[:, None],
                    torch.cos(2 * torch.pi * phase)[:, None],
                    targets['confidence'][:, None], targets['feedforward_muscles_80']]
        if targets['joint_angles'] is not None:
            features.append(targets['joint_angles'])
        if targets['joint_velocities'] is not None:
            features.append(targets['joint_velocities'])
        return {'actor': torch.cat(features, dim=1)}

    def _physics_step(self, excitations: torch.Tensor):
        raw_obs, base_reward, terminated, truncated, extras = self.sim.step(excitations)
        done = terminated | truncated
        grf_N = self.sim.get_vertical_grf('r')
        hs, to = self.contact_events.update(grf_N)
        self.phase_estimator.update(self.dt, hs, to, done=done)
        # mjlab automatically resets finished environments before returning.
        # Re-baseline contact sensor there; no false HS/TO on next step.
        self.contact_events.reset(mask=done, initial_grf_N=grf_N)
        hs, to = hs & ~done, to & ~done
        targets = self.reference_targets()
        return raw_obs, base_reward, terminated, truncated, targets, {
            'heel_strike': hs, 'toe_off': to, 'vertical_grf_N': grf_N, 'phase': targets['phase'],
            'phase_confidence': targets['confidence'],
            'raw_myosuite_reward': base_reward,
            'raw_myosuite_info': extras,
        }

    ### Unused, it is just a diagnostic helper for testing direct muscle commands and retrieving the original MyoSuite reward.
    def step_muscles(self, excitations: torch.Tensor):
        """Direct muscle-command integration check; returns ORIGINAL MyoSuite reward."""
        raw_obs, base_reward, terminated, truncated, _, info = self._physics_step(excitations)
        return raw_obs, base_reward, terminated, truncated, info

    def step(self, action: torch.Tensor):
        """Actual phase-aligned imitation step. Applies actions BEFORE advancing phase.

        Compare post-step simulated joint state with reference at
        post-step phase. On auto-reset envs, reports/sets zero reward on done
        indices until terminal observation semantics have been verified.
        """
        base_muscles = self.reference_targets()['feedforward_muscles_80']
        command = self.action_to_excitation(action, base_muscles)
        raw_obs, base_reward, terminated, truncated, targets, info = self._physics_step(command)
        im_reward, components = self.imitation.compute(targets['phase'], targets)
        done = terminated | truncated
        # mjlab auto-resets internally; post-step state is a reset state on done
        # rows, not the terminal state. Do not score reset poses as terminal gait.
        im_reward = torch.where(done, torch.zeros_like(im_reward), im_reward)
        info['imitation_terms'] = components
        info['feedforward_excitation'] = base_muscles

        ### WHY the TANH !!!!!!!!!!!!!!
        info['policy_residual'] = self.residual_scale * torch.tanh(action)
        info['applied_excitation'] = command
        return self.policy_observation(raw_obs, targets), im_reward, terminated, truncated, info

    def train(self):
        raise NotImplementedError(
            'Next stage: register an mjlab manager-based task and connect RSL-RL PPO; '
            'do not mistake this experiment wrapper for an RSL-RL Env.')

    def close(self):
        self.sim.close()
