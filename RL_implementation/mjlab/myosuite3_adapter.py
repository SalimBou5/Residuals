"""MyoSuite 3 mjlab (MuJoCo Warp) MyoLeg simulator adapter.

Real GPU physics; no NumPy copies in the per-step path. The model's foot touch
sensors provide a force-based stance signal used for HS/TO event edges.
"""
from __future__ import annotations

import torch


class MyoSuite3LegAdapter:
    def __init__(self, num_envs: int = 1024, device: str = "cuda:0"):
        if num_envs < 1:
            raise ValueError("num_envs must be >=1")
        try:
            from myosuite import make_env
        except ImportError as exc:
            raise ImportError("Install MyoSuite 3 from source with the [mjlab] extra") from exc
        self.env = make_env("myoLegWalk-v0", backend="mjlab",
                            num_envs=num_envs, device=device)
        self.device = torch.device(self.env.device)
        self.num_envs = int(self.env.num_envs)
        self.dt = float(self.env.step_dt)
        if self.dt <= 0:
            raise RuntimeError("Invalid mjlab control timestep")
        self.model = self.env.sim.mj_model
        self.entity = self.env.scene["robot"]

        # Important: the policy's action columns are in the entity's ctrl_ids
        # order, not necessarily the order of ALL model actuators.
        ctrl_ids = self.entity.indexing.ctrl_ids.cpu().tolist()
        self.actuator_names = tuple(self.model.actuator(i).name for i in ctrl_ids)
        if len(self.actuator_names) != 80:
            raise RuntimeError(f"Expected 80 MyoLeg actuators, found {len(self.actuator_names)}")
        if self.env.action_manager.total_action_dim != 80:
            raise RuntimeError("Unexpected mjlab action dimension/order; inspect your installation")

        # Joint state cache for repeated queries of the same joint names.
        self._joint_cache = {}

        # Muscle activation indices cached once, not inside rollout loops. The model's actuator_actadr/actuator_actnum
        # arrays are used to resolve the correct act[] columns for each actuator.
        ctrl_ids = self.entity.indexing.ctrl_ids.cpu().tolist()
        adr = [int(self.model.actuator_actadr[i]) for i in ctrl_ids]
        counts = [int(self.model.actuator_actnum[i]) for i in ctrl_ids]
        if any(a < 0 or n != 1 for a, n in zip(adr, counts)):
            raise RuntimeError('Some actuators do not have exactly one muscle activation state')
        self._activation_indices = adr
        
        # Model-side sensor indices resolved once, not inside rollout loops.
        self._sensor_names = tuple(self.model.sensor(i).name for i in range(self.model.nsensor))
        self._r_sensor_cols = self._resolve_sensor_cols(("r_foot", "r_toes"))
        self._l_sensor_cols = self._resolve_sensor_cols(("l_foot", "l_toes"))
        self._last_obs = None

    def _resolve_sensor_cols(self, sensor_names: tuple[str, str]) -> tuple[int, int]:
        columns = []
        for short_name in sensor_names:
            matches = [i for i, n in enumerate(self._sensor_names)
                       if n == short_name or n.endswith("/" + short_name)]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one touch sensor {short_name!r}; found {matches}. "
                                   f"Available: {self._sensor_names}")
            sid = matches[0]
            if int(self.model.sensor_dim[sid]) != 1:
                raise RuntimeError(f"Touch sensor {short_name!r} is not scalar")
            columns.append(int(self.model.sensor_adr[sid]))
        return tuple(columns)

    def get_normal_grf(self, side: str = 'r') -> torch.Tensor:
        """Foot + toe ground-reaction load in Newtons, per mjlab walker [B].

        The model defines real MuJoCo touch sensors at each heel/forefoot.
        Touch sensors aggregate normal contact forces (N). For myoLegWalk-v0
        on a horizontal plane, the ground normal is vertical, giving vertical
        GRF. This equivalence MUST be reconsidered on sloped terrain.
        """
        if side not in ('r', 'l'):
            raise ValueError("side must be 'r' or 'l'")
        cols = self._r_sensor_cols if side == 'r' else self._l_sensor_cols
        sensors = self.env.sim.data.sensordata
        if (not isinstance(sensors, torch.Tensor) or sensors.ndim != 2
                or sensors.shape[0] != self.num_envs):
            raise RuntimeError('Unexpected mjlab sensordata tensor layout')
        return sensors[:, cols[0]] + sensors[:, cols[1]]

    def _joint_indices(self, names: tuple[str, ...]):
        """This function finds the correct qpos and qvel columns by joint name."""
        import mujoco
        q_indices, dq_indices = [], []
        available = tuple(self.model.joint(i).name for i in range(self.model.njnt))
        for name in names:
            matches = [i for i, actual in enumerate(available)
                       if actual == name or actual.endswith('/' + name)]
            if len(matches) != 1:
                raise ValueError(f'Joint {name!r} resolves to {len(matches)} matches: {matches}')
            jid = matches[0]
            if int(self.model.jnt_type[jid]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                                     int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError(f'Joint {name!r} is not 1-DOF; map root/free joints separately')
            q_indices.append(int(self.model.jnt_qposadr[jid]))
            dq_indices.append(int(self.model.jnt_dofadr[jid]))
        return q_indices, dq_indices

    def get_joint_state(self, names: tuple[str, ...]):
        """Return [B,D] positions & true time velocities in named joint order."""
        key = tuple(names)
        if key not in self._joint_cache:
            self._joint_cache[key] = self._joint_indices(key)
        qi, vi = self._joint_cache[key]
        d = self.env.sim.data
        return d.qpos[:, qi], d.qvel[:, vi]

    def get_muscle_activations(self) -> torch.Tensor:
        """MuJoCo filtered muscle states (act), NOT muscle commands (ctrl)."""
        data = self.env.sim.data.act
        if data.ndim != 2 or data.shape[0] != self.num_envs:
            raise RuntimeError('Unexpected mjlab act state layout')
        return data[:, self._activation_indices]

    def get_full_state(self):
        """Copy of the current physical state for future approved-state RSI."""
        d = self.env.sim.data
        return {'qpos': d.qpos.clone(), 'qvel': d.qvel.clone(),
                'act': d.act.clone()}

    def sensor_report(self) -> dict:
        return {
            "r_touch_cols": self._r_sensor_cols,
            "l_touch_cols": self._l_sensor_cols,
            "n_sensors": len(self._sensor_names),
        }

    def reset(self):
        # mjlab's reset signature differs from the single-env Gymnasium API.
        obs, extras = self.env.reset()
        self._last_obs = obs
        return obs, extras

    def step(self, excitations: torch.Tensor):
        if excitations.shape != (self.num_envs, 80):
            raise ValueError(f"Expected [num_envs, 80], got {tuple(excitations.shape)}")
        if excitations.device != self.device:
            raise ValueError(f"Muscle actions must live on {self.device}")
        if not torch.is_floating_point(excitations):
            raise ValueError("Muscle actions must be floating point")
        
        # No .item() or CPU transfers here: keep the hot path on GPU.
        cmd = excitations.clamp(0.0, 1.0)  # !!!!!! IMPORTANT: WHY THE CLAMP, CHECK IT !!!!!!!!!!!
        obs, reward, terminated, truncated, extras = self.env.step(cmd)
        self._last_obs = obs
        return obs, reward, terminated, truncated, extras

    def close(self):
        self.env.close()
