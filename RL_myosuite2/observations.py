"""Step 4: complete, model-index-safe MyoLeg residual-gait observations.

Actual site quantities use the SAME moving-pelvis frame definition as the
prepared Gait120 data (see MEP-informed technical specification, Section 2.2):
    p_rel = R_p.T @ (p_site - p_pelvis)
    v_rel = R_p.T @ (v_site - v_pelvis - omega_pelvis x (p_site - p_pelvis))
    R_rel = R_p.T @ R_site
    omega_rel = R_p.T @ (omega_site - omega_pelvis)

The 0.1/0.2 s future joint lookahead follows the reference's nominal stride
period, NOT a guessed future event correction. No absolute root XY is exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from RSI.reference import LoopableReference, ReferenceSample
from RSI.rsi import MyoLegMuscleBinding
from phase.phase import PhaseController, wrap_half_cycle
from normalization import FixedObservationNormalizer

# No hip relative positions/linear velocities (they are essentially constants).
POSITION_SITES = ("knee_r", "knee_l", "ankle_r", "ankle_l", "toe_r", "toe_l")
ORIENTATION_SITES = ("hip_r", "hip_l", *POSITION_SITES)

def canonical_quaternion(q: np.ndarray) -> np.ndarray:
    """Choose q_w >= 0; -q represents the same physical orientation."""
    out = np.array(q, dtype=np.float64, copy=True)
    if out.shape[-1] != 4:
        raise ValueError("Expected scalar-first quaternion(s) with final dimension 4")
    out = np.where(out[..., :1] < 0, -out, out)
    return out


@dataclass(frozen=True)
class ObservationConfig:
    lookahead_seconds: tuple[float, ...] = (0.1, 0.2)
    contact_force_scale_n: float = 1000.0  # Retained for optional legacy linear mode
    contact_half_saturation_n: float = 100.0
    contact_normalization: str = "raw"
    normalization_file: str | None = None
    pelvis_body_name: str = "pelvis"

    def __post_init__(self):
        if (not self.lookahead_seconds or
                not all(np.isfinite(t) and t > 0 for t in self.lookahead_seconds) or
                any(b <= a for a, b in zip(self.lookahead_seconds, self.lookahead_seconds[1:]))):
            raise ValueError("Lookahead times must be positive and strictly increasing")
        if not np.isfinite(self.contact_force_scale_n) or self.contact_force_scale_n <= 0:
            raise ValueError("contact_force_scale_n must be positive and finite")
        if not np.isfinite(self.contact_half_saturation_n) or self.contact_half_saturation_n <= 0:
            raise ValueError("contact_half_saturation_n must be positive and finite")
        if self.contact_normalization not in ("raw", "saturating", "linear"):
            raise ValueError("contact_normalization must be raw, saturating or linear")
        if self.normalization_file is not None and self.contact_normalization != "raw":
            raise ValueError("Fixed normalization needs RAW contact forces in Newtons (no double scaling)")


class ObservationBuilder:
    """Stable, named 10-block layout shared by reset() and step().

    The MEP and activation blocks follow ascending *actuator ID* order.
    With unmapped disabled, they contain only 70 mapped muscles each;
    otherwise they include all 80. MuJoCo data.act and data.ctrl remain 80-D.
    Four force sensors are RH, RT, LH, LT in raw Newtons before the optional
    fixed normalizer (which applies a continuous asinh force transform).
    Contact events still use raw Newtons at 20/10 N thresholds.

    The constructor verifies all eight site names, free-root conventions,
    and the reference's required site labels. No site order is guessed.
    """

    BLOCK_NAMES = (
        "joint_proprioception", "root_state", "foot_contacts", "muscle_activations",
        "actual_sites", "reference_sites", "reference_joints", "future_joints",
        "mep_baseline", "gait_phase",
    )

    def __init__(self, model: Any, mujoco: Any, binding: MyoLegMuscleBinding,
                 reference: LoopableReference, config: ObservationConfig = ObservationConfig(),
                 *, enable_unmapped: bool = False):
        self.model, self.mj, self.binding, self.config = model, mujoco, binding, config
        if model.nq < 7 or model.nv < 6:
            raise ValueError("Expected free-root MyoLeg with nq>=7, nv>=6")
        # Our reference uses qpos[:7] and qvel[:6] for the free pelvis root.
        first_free = (int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
                      and int(model.jnt_qposadr[0]) == 0
                      and int(model.jnt_dofadr[0]) == 0)
        if not first_free:
            raise ValueError("Expected a free-root joint starting at qpos[0], qvel[0]")
        self.pelvis_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                               config.pelvis_body_name))
        if self.pelvis_id < 0:
            raise ValueError(f"Missing pelvis body {config.pelvis_body_name!r}")

        self.site_ids: dict[str, int] = {}
        for name in ORIENTATION_SITES:
            sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
            if sid < 0:
                raise ValueError(f"Missing required MyoLeg anatomical site {name!r}")
            self.site_ids[name] = sid

        self.activation_addresses = np.asarray(
            [int(model.actuator_actadr[i]) for i in range(model.nu)], dtype=int)
        if len(self.activation_addresses) != 80 or len(set(self.activation_addresses)) != 80:
            raise ValueError("Each of the 80 actuators must have one unique activation address")
        if np.any(self.activation_addresses < 0) or np.any(self.activation_addresses >= model.na):
            raise ValueError("Invalid muscle activation addresses")

        self.enable_unmapped = bool(enable_unmapped)
        mapped = np.asarray(binding.mapped_mask, dtype=bool)
        if mapped.shape != (80,) or int(mapped.sum()) != 70:
            raise ValueError("Expected exactly 70 mapped and 10 unmapped muscles")
        self.observed_actuator_ids = (np.arange(80, dtype=int) if self.enable_unmapped
                                      else np.flatnonzero(mapped))
        self.observed_activation_addresses = self.activation_addresses[self.observed_actuator_ids]
        self.observed_muscle_names = tuple(binding.names_by_id[i]
                                           for i in self.observed_actuator_ids)
        self.n_muscle_features = len(self.observed_actuator_ids)

        self.n_joint_q = model.nq - 7
        self.n_joint_v = model.nv - 6
        sizes = (
            self.n_joint_q + self.n_joint_v,
            1 + 4 + 3 + 3,
            4,
            self.n_muscle_features,
            self._n_site_features,
            self._n_site_features,
            self.n_joint_q,
            len(config.lookahead_seconds) * self.n_joint_q,
            self.n_muscle_features,
            6,  # speed, sin, cos, wrapped drift, pending correction, correction active
        )
        offset = 0
        slices = {}
        for name, size in zip(self.BLOCK_NAMES, sizes):
            slices[name] = slice(offset, offset + size)
            offset += size
        self.slices = MappingProxyType(slices)
        self.dimension = offset
        self.last_blocks: dict[str, np.ndarray] = {}  # UNNORMALIZED, for debugging
        self.last_normalized_blocks: dict[str, np.ndarray] = {}
        self.normalizer = None
        if config.normalization_file is not None:
            self.normalizer = FixedObservationNormalizer.load(
                config.normalization_file, expected_dimension=self.dimension,
                observed_muscles=self.observed_muscle_names,
                lookahead_seconds=config.lookahead_seconds,
                expected_slices=self.slices)
        self.reference = None
        self.ref_indices = None
        self.set_reference(reference)

    @property
    def _n_site_features(self) -> int:
        return 6*3 + 6*3 + 8*4 + 8*3

    def set_reference(self, ref: LoopableReference) -> None:
        missing = set(ORIENTATION_SITES) - set(ref.site_names)
        if missing:
            raise ValueError(f"Reference {ref.path} missing anatomical sites: {sorted(missing)}")
        if ref.qpos.shape[1] != self.model.nq or ref.qvel.shape[1] != self.model.nv:
            raise ValueError("Reference joint dimensions mismatch MyoLeg")
        if ref.desired_speed is None or not np.isfinite(ref.desired_speed):
            raise ValueError(f"Reference {ref.path} has no valid desired forward speed")
        self.ref_indices = {name: ref.site_names.index(name) for name in ORIENTATION_SITES}
        self.reference = ref

    @staticmethod
    def _relative_quaternion(mujoco: Any, root_R: np.ndarray, site_R: np.ndarray) -> np.ndarray:
        relative_R = root_R.T @ site_R
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(relative_R.reshape(9)))
        return q / np.linalg.norm(q)

    def _actual_site_blocks(self, data: Any, root_R: np.ndarray,
                            root_pos: np.ndarray, root_v: np.ndarray,
                            root_omega: np.ndarray, reference_sample: ReferenceSample
                            ) -> np.ndarray:
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        pos, linvel, quat, angvel = {}, {}, {}, {}
        for name, sid in self.site_ids.items():
            self.mj.mj_jacSite(self.model, data, jp, jr, sid)
            site_world_p = np.asarray(data.site_xpos[sid], dtype=np.float64)
            site_world_R = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape((3,3))
            site_world_v = jp @ data.qvel
            site_world_w = jr @ data.qvel
            rel_p_world = site_world_p - root_pos
            if name in POSITION_SITES:
                pos[name] = root_R.T @ rel_p_world
                linvel[name] = root_R.T @ (site_world_v - root_v -
                                            np.cross(root_omega, rel_p_world))
            q = self._relative_quaternion(self.mj, root_R, site_world_R)
            # q and -q are the SAME orientation. Match reference sign to avoid
            # presenting physically equivalent orientations as different states.
            q_ref = canonical_quaternion(reference_sample.site_quat_rel[self.ref_indices[name]])
            if np.dot(q, q_ref) < 0:
                q = -q
            quat[name] = canonical_quaternion(q)
            angvel[name] = root_R.T @ (site_world_w - root_omega)
        return np.concatenate([
            *[pos[n] for n in POSITION_SITES],
            *[linvel[n] for n in POSITION_SITES],
            *[quat[n] for n in ORIENTATION_SITES],
            *[angvel[n] for n in ORIENTATION_SITES],
        ])

    def _reference_site_blocks(self, sample: ReferenceSample) -> np.ndarray:
        return np.concatenate([
            *[sample.site_pos_rel[self.ref_indices[n]] for n in POSITION_SITES],
            *[sample.site_linvel_rel[self.ref_indices[n]] for n in POSITION_SITES],
            *[canonical_quaternion(sample.site_quat_rel[self.ref_indices[n]]) for n in ORIENTATION_SITES],
            *[sample.site_angvel_rel[self.ref_indices[n]] for n in ORIENTATION_SITES],
        ])

    def build(self, data: Any, clock: PhaseController,
              contact_forces: np.ndarray) -> np.ndarray:
        if self.reference is None or clock.duration is None:
            raise RuntimeError("Set reference and reset phase clock before building observation")
        f = np.asarray(contact_forces, dtype=np.float64)
        if f.shape != (4,) or not np.all(np.isfinite(f)) or np.any(f < -1e-6):
            raise ValueError("Four verified, nonnegative contact-force values are required")
        f = np.maximum(f, 0.)

        root_R = np.asarray(data.xmat[self.pelvis_id], dtype=np.float64).reshape(3,3)
        root_pos = np.asarray(data.xpos[self.pelvis_id], dtype=np.float64)
        root_q = np.asarray(data.xquat[self.pelvis_id], dtype=np.float64).copy()
        root_q /= np.linalg.norm(root_q)
        root_q = canonical_quaternion(root_q)
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        self.mj.mj_jacBody(self.model, data, jp, jr, self.pelvis_id)
        root_v = jp @ data.qvel
        root_omega = jr @ data.qvel

        # Use current adapted phase and *actual* last physics-step phase rate
        # for recorded target velocities, as specified in the technical PDF.
        current_target = self.reference.sample(clock.theta, phase_rate=clock.phase_rate_hz)
        lookaheads = [self.reference.sample(clock.theta + H / self.reference.duration).qpos[7:]
                      for H in self.config.lookahead_seconds]
        mep = self.binding.baseline(clock.theta)
        parts = {
            "joint_proprioception": np.concatenate((data.qpos[7:], data.qvel[6:])),
            "root_state": np.concatenate(([root_pos[2]], root_q,
                                           root_R.T @ root_v, root_R.T @ root_omega)),
            "foot_contacts": (f if self.config.contact_normalization == "raw"
                              else (f / (f + self.config.contact_half_saturation_n)
                                    if self.config.contact_normalization == "saturating"
                                    else f / self.config.contact_force_scale_n)),
            "muscle_activations": np.asarray(data.act[self.observed_activation_addresses], dtype=np.float64),
            "actual_sites": self._actual_site_blocks(data, root_R, root_pos,
                                                       root_v, root_omega, current_target),
            "reference_sites": self._reference_site_blocks(current_target),
            "reference_joints": current_target.qpos[7:],
            "future_joints": np.concatenate(lookaheads),
            "mep_baseline": mep[self.observed_actuator_ids],
            "gait_phase": np.array([
                self.reference.desired_speed,
                np.sin(2*np.pi*clock.phase), np.cos(2*np.pi*clock.phase),
                wrap_half_cycle(clock.phase_drift), clock.remaining_error, float(clock.active),
            ], dtype=np.float64),
        }
        for key, value in parts.items():
            arr = np.asarray(value)
            if arr.shape != (self.slices[key].stop - self.slices[key].start,):
                raise RuntimeError(f"Observation {key} has unexpected shape {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise FloatingPointError(f"Nonfinite observation component: {key}")
        self.last_blocks = parts  # named views for debugging, not part of PPO input
        result = np.concatenate([parts[k] for k in self.BLOCK_NAMES]).astype(np.float32)
        if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
            raise RuntimeError("Final observation shape/value mismatch")
        if self.normalizer is not None:
            result = self.normalizer.transform(result)
        self.last_normalized_blocks = {name: result[sl].copy() for name, sl in self.slices.items()}
        return result
