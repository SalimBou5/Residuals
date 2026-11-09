"""Step 2: model-index-safe MyoLeg random-state initialization (RSI).

Follows the validated test_rsi_mep_from_library.py: reset qpos/qvel, set the
periodic internal activation once, mj_forward, then use direct MEP commands.
No forces, kinematic tether, rewards, physics stepping or RL are hidden here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .reference import LoopableReference, ReferenceAssets, ReferenceSample


# Exactly five unmapped MyoLeg muscles per leg in the implementation PDF.
UNMAPPED_MUSCLES = frozenset(
    f"{stem}_{side}"
    for stem in ("piri", "grac", "edl", "fdl", "fhl")
    for side in ("r", "l")
)


class MyoLegMuscleBinding:
    """Check actuator and activation indices ONCE, then reuse at every reset/step.

    Public excitation/action arrays are in actual MuJoCo actuator-ID order.
    The MEP CSV and lookup are reordered by names, never assumed to match.
    """

    def __init__(self, model: Any, mujoco: Any, assets: ReferenceAssets):
        self.model = model
        self.mj = mujoco
        self.assets = assets
        if model.nu != 80 or model.na < 80:
            raise ValueError(f"Expected 80 muscle actuators/states; got nu={model.nu}, na={model.na}")

        def actuator_id(name: str) -> int:
            aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
            if aid < 0:
                raise ValueError(f"Actuator {name!r} missing from MyoLeg model")
            return aid

        self.mep_ids = np.asarray([actuator_id(name) for name in assets.mep.names], dtype=int)
        self.lookup_ids = np.asarray([actuator_id(name) for name in assets.lookup.names], dtype=int)
        if (set(self.mep_ids.tolist()) != set(range(80)) or
                set(self.lookup_ids.tolist()) != set(range(80))):
            raise ValueError("MEP/lookup must each cover all 80 MyoLeg actuators once")

        self.lookup_activation_addresses = np.empty(80, dtype=int)
        for col, aid in enumerate(self.lookup_ids):
            adr = int(model.actuator_actadr[aid])
            nstate = int(model.actuator_actnum[aid])
            if adr < 0 or nstate != 1:
                raise ValueError(f"Muscle {assets.lookup.names[col]!r} lacks a single activation state")
            if assets.lookup.saved_addresses is not None:
                saved_adr = int(assets.lookup.saved_addresses[col])
                if saved_adr != adr:
                    raise ValueError(f"Activation lookup address mismatch for "
                                     f"{assets.lookup.names[col]}: lookup={saved_adr}, model={adr}")
            self.lookup_activation_addresses[col] = adr
        if len(set(self.lookup_activation_addresses)) != 80:
            raise ValueError("Duplicate muscle activation state addresses")

        self.names_by_id = [None] * 80
        for col, aid in enumerate(self.mep_ids):
            self.names_by_id[int(aid)] = assets.mep.names[col]
        self.names_by_id = tuple(self.names_by_id)
        if not UNMAPPED_MUSCLES.issubset(self.names_by_id):
            missing = sorted(UNMAPPED_MUSCLES - set(self.names_by_id))
            raise ValueError(f"Expected 5 unmapped muscles per leg; missing names: {missing}")
        self.unmapped_mask = np.asarray([name in UNMAPPED_MUSCLES for name in self.names_by_id])
        self.mapped_mask = ~self.unmapped_mask
        self.unmapped_ids = np.flatnonzero(self.unmapped_mask)
        self.unmapped_activation_addresses = np.asarray(
            [int(model.actuator_actadr[aid]) for aid in self.unmapped_ids], dtype=int)

    def baseline(self, theta: float, *, enable_unmapped: bool = False) -> np.ndarray:
        """Get 80 MEP excitations, in actuator-ID order; unmapped off by default."""
        values = self.assets.mep.sample(theta)
        if (not np.all(np.isfinite(values)) or np.min(values) < -1e-5
                or np.max(values) > 1 + 1e-5):
            raise ValueError("Invalid MEP excitation value")
        ctrl = np.zeros(80, dtype=np.float64)
        ctrl[self.mep_ids] = np.clip(values, 0, 1)
        # Even when direct unmapped actions are enabled, those muscles have NO
        # experimental MEP prior. Only the policy may excite them.
        ctrl[self.unmapped_mask] = 0.0
        return ctrl

    def excitation(self, theta: float, action: np.ndarray, residual_limits: float | np.ndarray,
                   *, enable_unmapped: bool = False) -> np.ndarray:
        """Future 80-D action mapping, available now for isolated verification.

        Mapped: clip(MEP_i + limit_i * action_i, 0, 1).
        Unmapped disabled (DEFAULT): exactly zero, regardless of action.
        Unmapped enabled: (action_i + 1)/2 (direct, independent of MEP).
        """
        a = np.asarray(action, dtype=np.float64)
        lam = np.broadcast_to(np.asarray(residual_limits, dtype=np.float64), (80,))
        if a.shape != (80,) or not np.all(np.isfinite(a)) or np.any(np.abs(a) > 1 + 1e-8):
            raise ValueError("Muscle action must be a finite 80-D vector in [-1,1]")
        if not np.all(np.isfinite(lam)) or np.any(lam < 0):
            raise ValueError("Residual limits must be finite and nonnegative")
        ctrl = self.baseline(theta)
        ctrl[self.mapped_mask] = np.clip(
            ctrl[self.mapped_mask] + lam[self.mapped_mask] * a[self.mapped_mask], 0, 1)
        ctrl[self.unmapped_mask] = ((a[self.unmapped_mask] + 1) / 2
                                    if enable_unmapped else 0.0)
        return ctrl

    def init_activations(self, data: Any, profile_values: np.ndarray,
                         *, enable_unmapped: bool = False) -> None:
        """Only called during reset; NEVER impose activation profiles during physics."""
        a = np.asarray(profile_values, dtype=np.float64)
        if a.shape != (80,) or not np.all(np.isfinite(a)) or np.min(a) < -1e-5 or np.max(a) > 1+1e-5:
            raise ValueError("RSI lookup must provide 80 finite activations in [0,1]")
        data.act[self.lookup_activation_addresses] = np.clip(a, 0, 1)
        if not enable_unmapped:
            data.act[self.unmapped_activation_addresses] = 0.0


@dataclass(frozen=True)
class RSIConfig:
    max_floor_penetration_m: float = 0.015
    max_phase_attempts: int = 64
    initialize_ctrl: Literal["zero", "mep"] = "mep"  # We initialize the control with the MEP
    enable_unmapped: bool = False                     # User-selected DEFAULT: off.
    floor_geom_name: str = "floor"

    def __post_init__(self) -> None:
        if self.max_floor_penetration_m < 0 or self.max_phase_attempts < 1:
            raise ValueError("RSI penetration threshold and attempts must be valid")
        if self.initialize_ctrl not in ("zero", "mep"):
            raise ValueError("initialize_ctrl must be 'zero' or 'mep'")


@dataclass(frozen=True)
class PhaseReset:
    """Initial *unwrapped* phase clocks, ready for the separate phase controller.

    The next module must update these after physics steps and apply buffered
    event corrections. No hidden time/sim.time coupling in this reset module.
    """
    theta: float
    theta_nominal: float
    remaining_event_correction: float = 0.0
    event_correction_active: bool = False

    @property
    def phase(self) -> float:
        return self.theta % 1.0


@dataclass(frozen=True)
class ResetResult:
    reference: LoopableReference
    sample: ReferenceSample
    phase_state: PhaseReset
    matched_activation_duration_s: float
    max_floor_penetration_m: float
    phase_attempts: int
    contact_seed: Any     # Returned by optional verified contact reader; otherwise None.


class RSIInitializer:
    """Reusable environment setup; keeps one MjData and references for every reset.

    A contact_reader(model, data) can be injected once real heel/toe sensors
    have been verified. Its output is captured AFTER mj_forward at RSI and
    should be used to seed the future contact-event detector.
    """

    def __init__(self, model: Any, mujoco: Any, assets: ReferenceAssets, *,
                 data: Any | None = None, config: RSIConfig = RSIConfig(),
                 seed: int | None = None, contact_reader: Callable | None = None):
        self.model = model
        self.mj = mujoco
        self.assets = assets
        self.config = config
        self.data = data if data is not None else mujoco.MjData(model)
        self.rng = np.random.default_rng(seed)
        self.binding = MyoLegMuscleBinding(model, mujoco, assets)
        self.contact_reader = contact_reader
        self.floor_geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                  config.floor_geom_name))
        if self.floor_geom_id < 0:
            raise ValueError(f"Floor geom {config.floor_geom_name!r} missing; "
                             "cannot enforce the 15 mm RSI penetration check")
        self.active: ResetResult | None = None

    def _max_floor_penetration(self) -> float:
        max_depth = 0.0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            if contact.geom1 == self.floor_geom_id or contact.geom2 == self.floor_geom_id:
                max_depth = max(max_depth, -float(contact.dist))
        return max_depth

    def reset(self, *, reference: LoopableReference | str | Path | None = None,
              phase: float | None = None, seed: int | None = None) -> ResetResult:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if phase is not None and (not np.isfinite(phase) or not 0 <= phase < 1):
            raise ValueError("Explicit RSI phase must be in [0,1)")
        if reference is None:
            ref = self.assets.select(self.rng)  # hierarchical selection ONCE per episode
        elif isinstance(reference, LoopableReference):
            ref = reference
        else:
            ref = self.assets.library.get(Path(reference))
        if ref.qpos.shape[1] != self.model.nq or ref.qvel.shape[1] != self.model.nv:
            raise ValueError(f"Reference nq/nv={ref.qpos.shape[1]}/{ref.qvel.shape[1]} "
                             f"does not match model {self.model.nq}/{self.model.nv}")
        profile, matched_duration = self.assets.lookup.for_duration(ref.duration)
        n_attempts = 1 if phase is not None else self.config.max_phase_attempts
        previous_max = 0.0

        for attempt in range(1, n_attempts + 1):
            phi0 = float(phase) if phase is not None else float(self.rng.random())
            sample = ref.sample(phi0)
            self.mj.mj_resetData(self.model, self.data)
            self.data.qpos[:] = sample.qpos
            self.data.qvel[:] = sample.qvel
            self.binding.init_activations(
                self.data, profile.sample(phi0), enable_unmapped=self.config.enable_unmapped)
            self.data.ctrl[:] = 0.0
            if self.config.initialize_ctrl == "mep":
                self.data.ctrl[:] = self.binding.baseline(
                    phi0, enable_unmapped=self.config.enable_unmapped)
            self.data.time = 0.0

            # mj_forward here updates the model's derived state. Although it doesn't advance simulation time, 
            # control values can influence acceleration-related quantities.
            self.mj.mj_forward(self.model, self.data)  
            previous_max = self._max_floor_penetration()
            if previous_max > self.config.max_floor_penetration_m:
                continue  # Reject state; do NOT silently alter its measured pose.
            contact_seed = (self.contact_reader(self.model, self.data)
                            if self.contact_reader is not None else None)
            result = ResetResult(
                reference=ref, sample=sample,
                phase_state=PhaseReset(theta=phi0, theta_nominal=phi0),
                matched_activation_duration_s=matched_duration,
                max_floor_penetration_m=previous_max,
                phase_attempts=attempt, contact_seed=contact_seed)
            self.active = result
            return result
        self.active = None
        raise RuntimeError(f"RSI rejected {n_attempts} phase(s) for {ref.path}; "
                           f"last floor penetration={previous_max * 1000:.1f} mm > "
                           f"limit={self.config.max_floor_penetration_m * 1000:.1f} mm. "
                           "No root-height changes or joint corrections were applied.")

    def apply_baseline(self, theta: float) -> np.ndarray:
        """Apply the MEP at the phase chosen by the future clock, not sim.time."""
        ctrl = self.binding.baseline(theta, enable_unmapped=self.config.enable_unmapped)
        self.data.ctrl[:] = ctrl
        return ctrl
