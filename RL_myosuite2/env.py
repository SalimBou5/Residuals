"""Gymnasium/MyoLeg environment SKELETON (Steps 1-3 only).

Working reset and 200-Hz stepping with an 81-D action space. The observation
below is explicitly a TEMPORARY diagnostic vector (not the planned observation
from the technical specification). Reward is identically zero until Step 5.
Do not train PPO on this skeleton.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # unit-testing phase/rollout does not require Gymnasium
    gym = None
    spaces = None

from RSI.reference import ReferenceAssets
from RSI.rsi import RSIConfig, RSIInitializer
from phase.contacts import FourTouchSensorReader, ContactConfig
from control import ActionConfig
from phase.phase import PhaseConfig
from rollout import PhysicsControlLoop, RolloutConfig
from observations import ObservationConfig, ObservationBuilder
from rewards import RewardConfig, TerminationConfig, GaitRewardEvaluator

BaseEnv = gym.Env if gym is not None else object


class MyoLegResidualEnv(BaseEnv):
    metadata = {"render_modes": []}

    def __init__(self, model_path: str | Path, assets: ReferenceAssets, *,
                 sensor_names: tuple[str, str, str, str] = (
                     "r_foot", "r_toes", "l_foot", "l_toes"),
                 event_sync: bool = True,
                 rollout_config: RolloutConfig = RolloutConfig(),
                 action_config: ActionConfig = ActionConfig(),
                 phase_config: PhaseConfig = PhaseConfig(),
                 contact_config: ContactConfig = ContactConfig(),
                 observation_config: ObservationConfig = ObservationConfig(),
                 reward_config: RewardConfig = RewardConfig(),
                 termination_config: TerminationConfig = TerminationConfig(),
                 max_episode_seconds: float = 4.0, seed: int | None = None):
        if gym is None:
            raise ImportError("pip install gymnasium to use MyoLegResidualEnv")
        import mujoco
        self.mj = mujoco
        path = Path(model_path)
        self.model = (mujoco.MjModel.from_binary_path(str(path)) if path.suffix.lower() == '.mjb'
                      else mujoco.MjModel.from_xml_path(str(path)))
        self.assets = assets
        # Four actual force measurements are REQUIRED by the observation contract;
        # disabling event synchronization does not disable contact observations.
        self.reader = FourTouchSensorReader(self.model, mujoco, sensor_names)
        self.event_sync = bool(event_sync)
        self.rsi = RSIInitializer(
            self.model, mujoco, assets,
            config=RSIConfig(initialize_ctrl='mep', enable_unmapped=action_config.enable_unmapped),
            seed=seed, contact_reader=self.reader)
        self.core = PhysicsControlLoop(
            self.rsi, rollout_config=rollout_config, action_config=action_config,
            phase_config=phase_config, contact_config=contact_config,
            contact_reader=self.reader if self.event_sync else None)
        self.observations = ObservationBuilder(
            self.model, mujoco, self.rsi.binding,
            assets.library.get(assets.library.paths[0]), observation_config,
            enable_unmapped=action_config.enable_unmapped)
        self.rewards = GaitRewardEvaluator(
            self.model, mujoco, self.observations, reward_config, termination_config)
        self.max_episode_seconds = float(max_episode_seconds)
        if not np.isfinite(self.max_episode_seconds) or self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.core.actions.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observations.dimension,), dtype=np.float32)

    @property
    def observation_slices(self):
        """Public immutable mapping from each observation block to its flat slice."""
        return self.observations.slices

    def _get_obs(self) -> np.ndarray:
        if self.event_sync:
            forces = self.core.last_contact_forces
        else:
            forces = self.reader(self.model, self.rsi.data)
        return self.observations.build(self.rsi.data, self.core.clock, forces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        reset = self.core.reset(reference=options.get('reference'),
                                phase=options.get('phase'), seed=seed)
        self.observations.set_reference(reset.reference)
        info = {
            "action_dim": self.core.actions.action_dim,
            "enable_unmapped": self.core.actions.config.enable_unmapped,
            "reference": str(reset.reference.path),
            "reset_phase": reset.phase_state.phase,
            "max_reset_penetration_m": reset.max_floor_penetration_m,
            "activation_lookup_duration_s": reset.matched_activation_duration_s,
            "observation_dim": self.observations.dimension,
            "observation_status": "STEP4_COMPLETE",
            "observation_normalized": self.observations.normalizer is not None,
            "reward_status": "STEP5_PHYSICAL_UNITS",
            "event_sync_enabled": self.event_sync,
        }
        obs = self._get_obs()
        if self.observations.normalizer is not None:
            info["normalization_clipped_features"] = self.observations.normalizer.last_clipped_count
        return obs, info

    def step(self, action):
        info = self.core.step(action)
        truncated = float(self.rsi.data.time) >= self.max_episode_seconds - 1e-12
        info['reward_status'] = 'STEP5_PHYSICAL_UNITS'
        info['reference'] = str(self.observations.reference.path)
        info['observation_status'] = 'STEP4_COMPLETE'
        obs = self._get_obs()
        reward, terms, terminated, reason = self.rewards.evaluate(
            self.rsi.data, self.core.clock, self.core.actions, self.core.control_steps)
        info['reward_terms'] = terms
        info['termination_reason'] = reason
        info['truncation_reason'] = ('episode_time_limit' if truncated else None)
        if self.observations.normalizer is not None:
            info['normalization_clipped_features'] = self.observations.normalizer.last_clipped_count
            info['normalization_clip_fraction_total'] = self.observations.normalizer.saturation_fraction
        return obs, reward, terminated, truncated, info

    def close(self):
        pass
