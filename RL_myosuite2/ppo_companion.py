"""Step 6 environment factory and numerical diagnostics.

No Stable-Baselines3 or W&B imports here: this module remains importable by
unit tests and by Windows spawn workers before training begins.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math

import numpy as np


@dataclass(frozen=True)
class EnvironmentSettings:
    model: str
    references: str
    mep: str
    lookup: str
    normalization: str
    reward_config: str
    seed: int = 0
    max_episode_seconds: float = 4.0
    reference_cache_size: int = 16
    event_sync: bool = True
    enable_unmapped: bool = False
    enable_tracking_termination: bool = True

    def __post_init__(self) -> None:
        if self.reference_cache_size < 1:
            raise ValueError('reference_cache_size must be at least one')
        if not (math.isfinite(self.max_episode_seconds) and self.max_episode_seconds > 0):
            raise ValueError('max_episode_seconds must be finite and positive')
        if not isinstance(self.seed, int):
            raise TypeError('seed must be an integer')

    def verify_paths(self) -> None:
        for field in ('model', 'mep', 'lookup', 'normalization', 'reward_config'):
            p = Path(getattr(self, field))
            if not p.is_file():
                raise FileNotFoundError(f'{field}: {p}')
        if not Path(self.references).exists():
            raise FileNotFoundError(f'references: {self.references}')


def create_environment(settings: EnvironmentSettings, *, worker_index: int = 0):
    """Instantiate assets *inside* a worker, not in the parent process.

    Crucial for Windows spawn: each worker owns its MjModel/MjData and an LRU
    cache of reference cycles. Do not preload all 944 cycles per worker.
    """
    from RSI.reference import ReferenceAssets
    from env import MyoLegResidualEnv
    from control import ActionConfig
    from observations import ObservationConfig
    from rewards import RewardConfig, TerminationConfig

    with Path(settings.reward_config).open('r', encoding='utf-8') as stream:
        reward_cfg = RewardConfig(**json.load(stream))
    assets = ReferenceAssets(
        settings.references, settings.mep, settings.lookup,
        max_cached_references=settings.reference_cache_size,
        preload_references=False,
    )
    env = MyoLegResidualEnv(
        settings.model, assets, seed=settings.seed + worker_index,
        action_config=ActionConfig(enable_unmapped=settings.enable_unmapped),
        observation_config=ObservationConfig(normalization_file=settings.normalization),
        reward_config=reward_cfg,
        termination_config=TerminationConfig(
            enable_tracking_termination=settings.enable_tracking_termination),
        event_sync=settings.event_sync,
        max_episode_seconds=settings.max_episode_seconds,
    )
    return env


def neutral_mep_action(env: Any) -> np.ndarray:
    """Zero MEP residual and phase-rate correction.

    In 81-action direct-unmapped mode, -1 means zero excitation, whereas zero
    means 0.5 excitation. Therefore set unmapped action coordinates to -1.
    """
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    control = env.core.actions
    if control.config.enable_unmapped:
        mask = control.binding.unmapped_mask[control.policy_to_actuator_ids]
        action[:-1][mask] = -1.0
    return action


def validate_dynamic_step(env: Any, obs: np.ndarray, reward: float,
                          info: Mapping[str, Any], *, baseline_only: bool = True) -> None:
    """A hard safety check; terminated/fallen trials remain valid observations."""
    if obs.shape != env.observation_space.shape or not np.isfinite(obs).all():
        raise FloatingPointError('Dynamic observation shape or values are invalid')
    if not math.isfinite(float(reward)):
        raise FloatingPointError('Dynamic reward is not finite')
    if not all(math.isfinite(float(v)) for v in info['reward_terms'].values()):
        raise FloatingPointError('Nonfinite reward term')
    data = env.rsi.data
    if not all(np.isfinite(getattr(data, name)).all() for name in ('qpos','qvel','act','ctrl')):
        raise FloatingPointError('Nonfinite simulation state/control')
    if np.min(data.ctrl) < -1e-8 or np.max(data.ctrl) > 1.0 + 1e-8:
        raise ValueError('Muscle excitations must remain in [0,1]')
    if not math.isfinite(float(info['theta'])):
        raise FloatingPointError('Phase clock is not finite')
    if baseline_only:
        ctl = env.core.actions
        if not np.allclose(data.ctrl, ctl.last_baseline, atol=1e-10, rtol=0):
            raise AssertionError('Neutral MEP-only action did not reproduce the MEP command')


class TrainingDiagnostics:
    """Aggregate train rollout metrics and terminal reason counts per worker.

    W&B is called only from the parent PPO process, once per rollout. A small
    sum/number is kept for each term; no raw trajectory buffering is needed.
    """

    INFO_KEYS = (
        'phase_drift', 'event_remaining', 'phase_rate_hz',
        'normalization_clipped_features',
    )

    def __init__(self, n_envs: int, observation_dim: int):
        if n_envs < 1 or observation_dim < 1:
            raise ValueError('Invalid number of environments or observations')
        self.n_envs, self.observation_dim = n_envs, observation_dim
        self.episode_returns = [0.0] * n_envs
        self.episode_steps = [0] * n_envs
        self.total_episodes = 0
        self.references_seen_total: set[str] = set()
        self.clear_window()

    def clear_window(self) -> None:
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.terminal_reasons = Counter()
        self.completed_returns: list[float] = []
        self.completed_lengths: list[int] = []
        self.step_count = 0
        self.episode_count = 0
        self.event_observed = 0
        self.event_accepted = 0
        self.references_seen_window: set[str] = set()

    def _add(self, name: str, value: float) -> None:
        value = float(value)
        if math.isfinite(value):
            self.sums[name] += value
            self.counts[name] += 1

    def observe(self, worker: int, reward: float, info: Mapping[str, Any],
                done: bool, action: np.ndarray | None = None) -> None:
        if worker < 0 or worker >= self.n_envs:
            raise IndexError('Invalid worker index')
        self.step_count += 1
        self.episode_returns[worker] += float(reward)
        self.episode_steps[worker] += 1
        self._add('reward/step', reward)
        for k, v in info.get('reward_terms', {}).items():
            if isinstance(v, (float,int,np.floating,np.integer)) and not isinstance(v, (bool,np.bool_)):
                self._add('reward_terms/' + str(k), v)
        for k in self.INFO_KEYS:
            if k in info:
                self._add('state/' + k, info[k])
        if 'phase_drift' in info:
            self._add('state/abs_phase_drift', abs(float(info['phase_drift'])))
        if action is not None:
            a = np.asarray(action, float)
            if a.ndim == 1 and a.size:
                self._add('action/muscle_abs_mean', np.mean(np.abs(a[:-1])))
                self._add('action/phase_abs', abs(a[-1]))
        for event in info.get('events', ()):
            self.event_observed += 1
            if event.get('accepted', False):
                self.event_accepted += 1
        if done:
            if info.get('reference'):
                ref = str(info['reference'])
                self.references_seen_window.add(ref)
                self.references_seen_total.add(ref)
            self.episode_count += 1
            self.total_episodes += 1
            self.completed_returns.append(self.episode_returns[worker])
            self.completed_lengths.append(self.episode_steps[worker])
            reason = info.get('termination_reason') or info.get('truncation_reason') or 'unknown'
            self.terminal_reasons[str(reason)] += 1
            self.episode_returns[worker] = 0.0
            self.episode_steps[worker] = 0

    def snapshot(self, *, reset: bool = True) -> dict[str, float]:
        metrics = {name: self.sums[name]/count for name,count in self.counts.items() if count}
        metrics['rollout/steps'] = float(self.step_count)
        metrics['episodes/completed'] = float(self.episode_count)
        metrics['episodes/total'] = float(self.total_episodes)
        metrics['events/observed'] = float(self.event_observed)
        metrics['events/accepted'] = float(self.event_accepted)
        metrics['references/unique_window'] = float(len(self.references_seen_window))
        metrics['references/unique_total'] = float(len(self.references_seen_total))
        if self.event_observed:
            metrics['events/accept_fraction'] = self.event_accepted/self.event_observed
        if self.completed_returns:
            metrics['episodes/mean_return'] = float(np.mean(self.completed_returns))
            metrics['episodes/mean_length_steps'] = float(np.mean(self.completed_lengths))
        if self.step_count:
            metrics['normalization/clipped_fraction'] = (
                self.sums.get('state/normalization_clipped_features', 0.) /
                (self.step_count * self.observation_dim))
        for reason, count in self.terminal_reasons.items():
            metrics['episodes/ended_' + reason] = float(count)
            if self.episode_count:
                metrics['episodes/fraction_' + reason] = count/self.episode_count
        if reset:
            self.clear_window()
        return metrics
