"""Real CPU MyoSuite/MyoLeg simulator connection.

Works with the Gym style wrapper provided by MyoSuite.  This module imports
MyoSuite only when the adapter is instantiated, so the phase/mapping unit tests
remain usable without a physics installation.

No RL decisions or model-specific gait assumptions are made here.
"""
from __future__ import annotations

import numpy as np


class MyoSuiteLegAdapter:
    def __init__(self, env_id: str = "myoLegWalk-v0"):
        try:
            import myosuite  # noqa: F401; registers tasks
            from myosuite.utils import gym
        except ImportError as exc:
            raise ImportError(
                "Install MyoSuite in your research conda environment (pip install myosuite)"
            ) from exc

        # CRITICAL: Muscle reference values are in [0, 1].  The default
        # normalized action mode in some MyoSuite 2.x versions is [-1,1]
        # with a nonlinear transformation.  Explicitly request raw muscle
        # excitation commands for the reference + residual controller.
        self.env = gym.make(env_id, normalize_act=False)
        self.raw = self.env.unwrapped
        self.model = self._get_model()
        self.data = self._get_data()
        self.actuator_names = self._get_actuator_names()
        self.dt = self._get_dt()

        if len(self.actuator_names) != 80:
            raise RuntimeError(
                f"Expected 80 MyoLeg muscle actuators; found {len(self.actuator_names)}. "
                f"Check the model/environment ID: {env_id}"
            )
        bounds_low = np.asarray(self.env.action_space.low, dtype=float)
        bounds_high = np.asarray(self.env.action_space.high, dtype=float)
        if bounds_low.shape != (len(self.actuator_names),):
            raise RuntimeError(
                f"Action space {bounds_low.shape} does not match {len(self.actuator_names)} actuators"
            )
        if (np.any(bounds_low < -1e-8) or np.any(bounds_high > 1 + 1e-8)
                or np.any(bounds_high <= bounds_low)):
            raise RuntimeError(
                "Expected direct [0,1] muscle commands. This MyoSuite installation "
                "has a different action contract; inspect action_space and normalize_act. "
                f"Current bounds: {bounds_low.min()}..{bounds_high.max()}"
            )

        # Actual foot-terrain contact pairs in the MuJoCo contact array are
        # used for heel-strike and toe-off *edges* (not reference clock phase).
        self._foot_geoms, self._ground_geoms = self._identify_contact_geoms()

    def _get_model(self):
        if hasattr(self.raw, "mj_model"):
            return self.raw.mj_model
        if hasattr(self.raw, "sim") and hasattr(self.raw.sim, "model"):
            return self.raw.sim.model
        raise RuntimeError("Cannot locate MyoLeg MuJoCo model on this MyoSuite version")

    def _get_data(self):
        if hasattr(self.raw, "mj_data"):
            return self.raw.mj_data
        if hasattr(self.raw, "sim") and hasattr(self.raw.sim, "data"):
            return self.raw.sim.data
        raise RuntimeError("Cannot locate MyoLeg MuJoCo data on this MyoSuite version")

    @staticmethod
    def _name_at(model, obj: str, i: int) -> str:
        # Current official MuJoCo API; older MyoSuite sim wrappers expose *_id2name.
        try:
            name = getattr(model, obj)(i).name
        except (AttributeError, TypeError, ValueError):
            name = getattr(model, f"{obj}_id2name")(i)
        return "" if name is None else str(name)

    def _get_actuator_names(self) -> tuple[str, ...]:
        names = tuple(self._name_at(self.model, "actuator", i)
                      for i in range(int(self.model.nu)))
        if any(not name for name in names):
            raise RuntimeError("Unnamed MyoLeg actuators: cannot map reference profiles safely")
        return names

    def _get_dt(self) -> float:
        if hasattr(self.raw, "dt"):
            dt = float(self.raw.dt)
        else:
            dt = float(self.model.opt.timestep) * int(self.raw.frame_skip)
        if dt <= 0:
            raise RuntimeError("Simulation control timestep must be positive")
        return dt

    def _identify_contact_geoms(self) -> tuple[frozenset[int], frozenset[int]]:
        """Choose right-foot versus ground geoms by the actual MyoLeg names.

        Confirm these sets by printing adapter.contact_geometry_report() when
        integrating another MyoLeg version.  No contact/phase is guessed if
        these cannot be identified.
        """
        foot, ground = set(), set()
        for i in range(int(self.model.ngeom)):
            geom_name = self._name_at(self.model, "geom", i).lower()
            body_id = int(self.model.geom_bodyid[i])
            body_name = self._name_at(self.model, "body", body_id).lower()
            foot_token = ("talus", "calcn", "toes", "foot", "metatars", "mtp")
            if (body_name.endswith("_r") and any(t in body_name for t in foot_token)) or (
                geom_name.endswith("_r") and any(t in geom_name for t in foot_token)
            ):
                foot.add(i)
            if (body_id == 0 or any(t in geom_name for t in ("ground", "terrain", "floor", "plane"))):
                ground.add(i)
        return frozenset(foot), frozenset(ground)

    def contact_geometry_report(self) -> dict:
        """Inspectable names for checking the automatic contact classification."""
        def names(ids):
            return [self._name_at(self.model, "geom", i) for i in sorted(ids)]
        return {"right_foot_geoms": names(self._foot_geoms),
                "ground_geoms": names(self._ground_geoms)}

    def right_foot_in_contact(self) -> bool:
        if not self._foot_geoms or not self._ground_geoms:
            raise RuntimeError(
                "Could not identify right-foot or terrain geoms for contact events. "
                f"Inspect contact_geometry_report(): {self.contact_geometry_report()}"
            )
        # Refresh data reference in case MyoSuite has replaced it on reset.
        self.data = self._get_data()
        for i in range(int(self.data.ncon)):
            con = self.data.contact[i]
            a, b = int(con.geom1), int(con.geom2)
            if ((a in self._foot_geoms and b in self._ground_geoms)
                    or (b in self._foot_geoms and a in self._ground_geoms)):
                return True
        return False

    def reset(self, seed: int | None = None):
        result = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        self.data = self._get_data()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, muscle_excitations):
        cmd = np.asarray(muscle_excitations, dtype=np.float32)
        if cmd.shape != (len(self.actuator_names),):
            raise ValueError(f"Expected {len(self.actuator_names)} actuator commands; got {cmd.shape}")
        if not np.isfinite(cmd).all() or (cmd < 0).any() or (cmd > 1).any():
            raise ValueError("MyoLeg muscle excitations must be finite and within [0,1]")
        result = self.env.step(cmd)
        if len(result) == 5:
            return result
        if len(result) == 4:  # Older gym/MyoSuite
            obs, reward, done, info = result
            return obs, reward, bool(done), False, info
        raise RuntimeError(f"Unexpected MyoSuite step return length: {len(result)}")

    def close(self):
        self.env.close()
