"""MyoSuite-backed phase-indexed reference controller and future RL wrapper.

The real simulator is MyoSuite's myoLegWalk-v0, through MyoSuiteLegAdapter.
RL observation/action/reward/training are *intentionally* TODO, as requested.
"""
from __future__ import annotations

import numpy as np
from gait_phase import GaitPhaseEstimator, PhaseConfig, FootContactEvents
from gait_reference import GaitReference
from muscle_mapping import MuscleMapper
from myosuite_adapter import MyoSuiteLegAdapter


class GaitRLEnv:
    def __init__(self, reference: GaitReference, phase_config: PhaseConfig, sim=None):
        # Genuine MyoSuite environment (or an injected test double for unit tests).
        self.sim = sim if sim is not None else MyoSuiteLegAdapter("myoLegWalk-v0")
        self.dt = self.sim.dt                 # MyoSuite's actual *control* timestep
        self.mapper = MuscleMapper(self.sim.actuator_names)  # Actual order of 80 actuators
        self.reference = reference
        self.phase_estimator = GaitPhaseEstimator(phase_config)
        self.contact_events = FootContactEvents()
        self.last_myo_obs = None
        self.last_myo_info = {}
        self.policy = None                   # TODO: RL algorithm/policy
        self.action_space = None             # TODO: define RL residual action representation
        self.observation_space = None        # TODO: define agent's observation vector

    def reset(self, *, seed=None, options=None):
        myo_obs, myo_info = self.sim.reset(seed=seed)  # Actual MyoSuite physics reset
        self.phase_estimator.reset(initial_phase=0.0)  # TODO: select initial gait phase
        self.contact_events.reset()
        # Observe contact baseline: a foot already standing on the floor is NOT an HS event.
        self.contact_events.update(self.sim.right_foot_in_contact())
        self.last_myo_obs = myo_obs
        self.last_myo_info = myo_info
        return myo_obs, {"phase": self.phase_estimator.phase,
                         "raw_myosuite_info": myo_info}  # TODO: define RL observation

    def reference_targets(self) -> dict:
        """Reference values indexed by the *estimated* current gait phase."""
        ref = self.reference.at_phase(self.phase_estimator.phase)
        activations, unresolved = self.mapper.expand(ref["muscles_15"])
        return {**ref, "mapped_muscles": activations,
                "unresolved_actuators": unresolved,
                "phase": self.phase_estimator.phase,
                "confidence": self.phase_estimator.confidence}

    def _advance_myoleg(self, muscle_action):
        """The actual integration: 80 muscle commands -> MyoSuite -> contact -> phase."""
        myo_obs, myo_reward, terminated, truncated, myo_info = self.sim.step(muscle_action)
        self.last_myo_obs = myo_obs
        self.last_myo_info = myo_info
        hs, to = self.contact_events.update(self.sim.right_foot_in_contact())
        self.phase_estimator.update(self.dt, heel_strike=hs, toe_off=to)
        return myo_obs, myo_reward, terminated, truncated, {
            "heel_strike": hs, "toe_off": to,
            "phase": self.phase_estimator.phase,
            "phase_confidence": self.phase_estimator.confidence,
            "raw_myosuite_reward": myo_reward,
            "raw_myosuite_info": myo_info,
        }

    def step(self, action):
        targets = self.reference_targets()   # 15 normalized channels -> MyoLeg order
        if targets["unresolved_actuators"]:
            raise RuntimeError(
                "The PDF mapping is incomplete: resolve these muscles before running "
                "reference-plus-residual stimulation: "
                + ", ".join(targets["unresolved_actuators"])
            )
        # TODO: choose RL action definition, e.g. delta-a in muscle space.
        delta_a = self._action_to_delta_a(action)
        if delta_a.shape != targets["mapped_muscles"].shape:
            raise ValueError("RL residual dimensions must match MyoLeg 80 actuators")
        u = np.clip(targets["mapped_muscles"] + delta_a, 0.0, 1.0)
        myo_obs, myo_reward, terminated, truncated, extra_info = self._advance_myoleg(u)
        # TODO: agent observation and custom tracking reward.  Never use the
        # built-in walking reward as a substitute for the planned tracking reward.
        agent_obs = self._get_observation(myo_obs, self.reference_targets())
        custom_reward = self._compute_reward(agent_obs, action)
        return agent_obs, custom_reward, terminated, truncated, extra_info

    def _action_to_delta_a(self, action):
        raise NotImplementedError("TODO: decide RL correction/action representation")

    def _get_observation(self, myosuite_obs, reference_targets):
        raise NotImplementedError("TODO: decide agent observation (MyoSuite state + phase + target)")

    def _compute_reward(self, observation, action):
        raise NotImplementedError("TODO: decide imitation/tracking reward")

    def train(self):
        raise NotImplementedError("TODO: choose RL algorithm and training")

    def close(self):
        self.sim.close()
