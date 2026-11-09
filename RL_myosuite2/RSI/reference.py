"""Step 1: cached Gait120 loopable kinematics, shared MEP and activation lookup.

File formats and interpolation follow test_rsi_mep_from_library.py. No model
kinematics are recomputed here; the retargeted reference remains authoritative.
"""
from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tqdm import tqdm

import numpy as np


def _unit_quaternion_nlerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc normalized linear quaternion interpolation, as in RSI test."""
    # Adjust the sign of the quaternion to ensure we are taking the shortest way (Physical way)
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
    # Now we can apply standard linear interpolation
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Cannot interpolate near-zero quaternion")
    return q / norm


class PeriodicTable:
    """One shared periodic table, sampled without resampling its original phase grid."""

    def __init__(self, phase: np.ndarray, values: np.ndarray, name: str):
        p = np.asarray(phase, dtype=np.float64).reshape(-1)
        y = np.asarray(values, dtype=np.float64)
        if y.ndim != 2 or len(y) != len(p) or len(p) < 3:
            raise ValueError(f"{name}: expected (samples, channels) table")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)):
            raise ValueError(f"{name}: nonfinite phase or data")
        if abs(p[0]) > 1e-6 or np.any(np.diff(p) <= 0) or p[-1] > 1.0 + 1e-6:
            raise ValueError(f"{name}: phase must start at 0, rise strictly and end <= 1")
        if abs(p[-1] - 1.0) < 1e-6:
            p, y = p[:-1], y[:-1]  # Duplicate 100% row; explicit wrap below.
        if len(p) < 2 or p[-1] >= 1:
            raise ValueError(f"{name}: invalid periodic phase array")
        self.name = name
        self.phase = p
        self.values = y
        self.phase.setflags(write=False)
        self.values.setflags(write=False)

    def sample(self, phase: float) -> np.ndarray:
        # linearly interpolates the table's values for a given wrapped phase + handling the cycling wrapping between the end and the beginning of the table
        phi = float(phase) % 1.0
        idx = max(0, int(np.searchsorted(self.phase, phi, side="right") - 1))
        j = (idx + 1) % len(self.phase)
        left = float(self.phase[idx])
        right = float(self.phase[j]) + (1.0 if j == 0 else 0.0)
        weight = (phi - left) / (right - left)
        return (1.0 - weight) * self.values[idx] + weight * self.values[j]


@dataclass(frozen=True)
class ReferenceEvents:
    """Event phases and their provenance; assumed TO remains explicitly labelled."""

    phases: Mapping[str, float]
    sources: Mapping[str, str]

    @classmethod
    def from_manifest(cls, metadata: dict) -> "ReferenceEvents":
        ev = metadata.get("events", {})
        phases = {"right_hs": 0.0}
        sources = {"right_hs": "dataset_stride_boundary"}
        aliases = (
            ("left_hs", "left_hs_phase_in_right_stride", "left_hs_source"),
            ("left_toe_off", "left_toe_off_phase", "left_toe_off_source"),
            ("right_toe_off", "right_toe_off_phase", "right_toe_off_source"),
        )
        for label, phase_key, source_key in aliases:
            if ev.get(phase_key) is not None:
                phase = float(ev[phase_key])
                if not np.isfinite(phase) or not 0 <= phase <= 1:
                    raise ValueError(f"Invalid {phase_key}: {phase}")
                phases[label] = phase % 1.0
                sources[label] = str(ev.get(source_key, "unspecified"))
        return cls(phases=phases, sources=sources)


@dataclass(frozen=True)
class ReferenceSample:
    theta: float                 # UNWRAPPED phase (may exceed 1)
    phase: float                 # Wrapped phase in [0, 1)
    cycle_index: int
    qpos: np.ndarray             # Root world XY progresses each cycle
    qvel: np.ndarray             # Optionally time-warped
    site_pos_rel: np.ndarray
    site_quat_rel: np.ndarray
    site_linvel_rel: np.ndarray
    site_angvel_rel: np.ndarray


class LoopableReference:
    """One right-HS to right-HS MyoLeg gait cycle with continuous forward travel."""

    def __init__(self, path: Path, *, require_manifest: bool = True):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            needed = (
                "qpos", "qvel", "time_since_start_s", "stride_duration_s",
                "site_names", "site_pos_rel", "site_quat_rel", "site_angvel_rel",
            )
            missing = [key for key in needed if key not in data]
            if missing:
                raise ValueError(f"{self.path}: missing reference arrays {missing}")
            self.qpos = np.asarray(data["qpos"], dtype=np.float64)
            self.qvel = np.asarray(data["qvel"], dtype=np.float64)
            self.time = np.asarray(data["time_since_start_s"], dtype=np.float64)
            self.duration = float(data["stride_duration_s"])
            self.phase_grid = (np.asarray(data["phase_right_01"], dtype=np.float64)
                               if "phase_right_01" in data else self.time / self.duration)
            self.site_names = tuple(str(name) for name in data["site_names"].tolist())
            self.site_pos = np.asarray(data["site_pos_rel"], dtype=np.float64)
            self.site_quat = np.asarray(data["site_quat_rel"], dtype=np.float64)
            site_v_key = "site_linvel_rel" if "site_linvel_rel" in data else "site_vel_rel"
            if site_v_key not in data:
                raise ValueError(f"{self.path}: missing site-relative linear velocities")
            self.site_linvel = np.asarray(data[site_v_key], dtype=np.float64)
            self.site_angvel = np.asarray(data["site_angvel_rel"], dtype=np.float64)
            self.root_displacement = (np.asarray(data["root_displacement_xyz_m"], dtype=np.float64)
                                      if "root_displacement_xyz_m" in data
                                      else self.qpos[-1, :3] - self.qpos[0, :3])
            self.desired_speed = (float(data["forward_speed_m_s"])
                                  if "forward_speed_m_s" in data else None)

        n = len(self.time)
        if self.qpos.ndim != 2 or self.qvel.ndim != 2 or self.qpos.shape[1] < 7:
            raise ValueError(f"{self.path}: invalid qpos/qvel shape")
        if n < 3 or self.qpos.shape[0] != n or self.qvel.shape[0] != n:
            raise ValueError(f"{self.path}: inconsistent kinematic sample counts")
        if (self.duration <= 0 or np.any(np.diff(self.time) <= 0)
                or not np.isclose(self.time[0], 0, atol=1e-6)
                or not np.isclose(self.time[-1], self.duration, atol=1e-4)):
            raise ValueError(f"{self.path}: invalid reference timestamps")
        if (self.phase_grid.shape != (n,) or not np.allclose(self.phase_grid, self.time / self.duration, atol=1e-5)):
            raise ValueError(f"{self.path}: reference phase and timestamps disagree")
        ns = len(self.site_names)
        for arr, last_dim, name in (
            (self.site_pos, 3, "site_pos_rel"),
            (self.site_quat, 4, "site_quat_rel"),
            (self.site_linvel, 3, "site_linvel_rel"),
            (self.site_angvel, 3, "site_angvel_rel"),
        ):
            if arr.shape != (n, ns, last_dim):
                raise ValueError(f"{self.path}: {name} shape {arr.shape} != {(n, ns, last_dim)}")
        if len(set(self.site_names)) != ns or self.root_displacement.shape != (3,):
            raise ValueError(f"{self.path}: invalid site names / root displacement")
        for name in ("qpos", "qvel", "time", "phase_grid", "site_pos", "site_quat",
                     "site_linvel", "site_angvel", "root_displacement"):
            arr = getattr(self, name)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{self.path}: {name} contains NaN/Inf")
            arr.setflags(write=False)
        if not np.allclose(self.qpos[-1, :3] - self.qpos[0, :3], self.root_displacement, atol=1e-4):
            raise ValueError(f"{self.path}: recorded displacement and root positions disagree")

        # The per-step data files use 'step01_loopable_manifest.json' as sidecar.
        manifest_path = self.path.with_name(self.path.stem + "_manifest.json")
        if require_manifest and not manifest_path.is_file():
            raise FileNotFoundError(f"Missing event manifest: {manifest_path}")
        self.manifest_path = manifest_path if manifest_path.is_file() else None
        self.metadata = (json.loads(manifest_path.read_text(encoding="utf-8"))
                         if self.manifest_path else {})
        self.events = ReferenceEvents.from_manifest(self.metadata)

    @property
    def n_samples(self) -> int:
        return len(self.time)

    @property
    def array_nbytes(self) -> int:
        """Bytes of numeric reference arrays retained in memory (excluding overhead)."""
        return sum(getattr(self, name).nbytes for name in (
            "qpos", "qvel", "time", "phase_grid", "site_pos", "site_quat",
            "site_linvel", "site_angvel", "root_displacement",
        ))

    def sample(self, theta: float, *, phase_rate: float | None = None) -> ReferenceSample:
        """Sample an UNWRAPPED cycle count, preserving forward root displacement."""

        theta = float(theta)
        if not np.isfinite(theta):
            raise ValueError("Unwrapped phase must be finite")

        # theta is continous (e.g.: 2.3 means the 3rd stride, 30% of the way through)
        # cycles isolates the whole number of strides completed (e.g. floor(2.3) = 2)
        # 'phase' isolates the fractional part within the current stride,
        cycles = int(np.floor(theta)) 
        phase = theta - cycles

        # Convert the fractional phase (0 to 1) into an actual target time in seconds.
        t = phase * self.duration
        # Find the array index representing the closest recorded time step *before* our target 't'.
        idx = min(max(0, int(np.searchsorted(self.time, t, side="right") - 1)), len(self.time) - 2)
        # alpha is the percentage of progress between time[idx] and time[idx+1].
        alpha = float((t - self.time[idx]) / (self.time[idx + 1] - self.time[idx]))

        def interp(array: np.ndarray) -> np.ndarray:
            # standard linear interpolation : Value = (1 - alpha) * Left_Value + (alpha) * Right_Value
            return ((1.0 - alpha) * array[idx] + alpha * array[idx + 1]).copy()

        # interpolate joint positions
        q = interp(self.qpos)

        # The root rotation (quaternion at indices 3:7) requires special spherical/shortest-arc 
        # interpolation, standard linear interpolation would deform the rotation.
        # Because q and -q represent the same 3D physical rotation --> to animate the transition 
        # between 2 orientations, the object needs to take the shortest path (physical rotational path)
        q[3:7] = _unit_quaternion_nlerp(self.qpos[idx, 3:7], self.qpos[idx + 1, 3:7], alpha)

        # This is what makes it loopable --> we take the base XYZ translation (indices 0:3) and 
        # push it forward by the total displacement of a full stride, multiplied by the 
        # number of full cycles completed. This allows infinite forward walking.
        q[:3] += cycles * self.root_displacement

        # Interpolate joint velocities
        v = interp(self.qvel)
        
        # interpolate root-relative site positions 
        pos = interp(self.site_pos)

        # Site rotations also need quaternion interpolation
        quat = _unit_quaternion_nlerp(self.site_quat[idx], self.site_quat[idx + 1], alpha)
        linear_vel = interp(self.site_linvel)
        angular_vel = interp(self.site_angvel)

        if phase_rate is not None:
            if not np.isfinite(phase_rate) or phase_rate <= 0:
                raise ValueError("Phase rate must be positive (cycles/second)")
            # If the reference was recorded at 1 stride per second, but we want it to run at 
            # 1.5 strides per second, we need to artificially scale all velocities by 1.5.
            # factor = (recorded duration) * (desired cycles/second).
            factor = self.duration * float(phase_rate)
            v *= factor
            linear_vel *= factor
            angular_vel *= factor
        return ReferenceSample(theta, phase, cycles, q, v, pos, quat, linear_vel, angular_vel)


class ReferenceLibrary:
    """Hierarchical subject->trial->stride library, with full RAM preload by default.

    Default: load all reference arrays once during initialization, for disk-free
    randomized RSI. Optional bounded LRU mode remains available when memory is
    limited or when creating many independent PPO worker processes.
    """

    def __init__(self, root: str | Path, *, max_cached_references: int | None = None,
                 require_manifest: bool = True, preload: bool | None = None):
        root = Path(root).resolve()
        if max_cached_references is not None and max_cached_references < 1:
            raise ValueError("max_cached_references must be >= 1 or None")
        if preload is None:
            preload = max_cached_references is None
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = sorted(root.rglob("*_loopable.npz"))
        else:
            raise FileNotFoundError(root)
        if not paths:
            raise ValueError(f"No *_loopable.npz references found under {root}")
        self.paths = tuple(paths)
        self.require_manifest = require_manifest
        self.max_cached_references = max_cached_references
        self._cache: OrderedDict[Path, LoopableReference] = OrderedDict()
        self._known_paths = set(self.paths)
        if preload and max_cached_references is not None and max_cached_references < len(self.paths):
            raise ValueError("Cannot preload entire library with a smaller cache limit; "
                             "use max_cached_references=None or preload=False")
        groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
        for path in self.paths:
            if path.parent.name.startswith("Trial") and path.parent.parent.name.startswith("S"):
                subject, trial = path.parent.parent.name, path.parent.name
            elif root.is_file():
                # Single-file use is handy for reproducing the validated RSI test.
                sidecar = path.with_name(path.stem + "_manifest.json")
                metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
                subject, trial = str(metadata.get("subject", "single")), str(metadata.get("trial", "single"))
            else:
                raise ValueError(f"Expected .../S###/Trial##/*_loopable.npz, got {path}")
            groups[subject][trial].append(path)
        self.groups = {s: {t: tuple(sorted(paths)) for t, paths in sorted(trials.items())}
                       for s, trials in sorted(groups.items())}
        if preload:
            self.preload_all()

    @property
    def cached_array_bytes(self) -> int:
        """RAM used by retained numeric reference arrays, excluding Python overhead."""
        return sum(ref.array_nbytes for ref in self._cache.values())

    @property
    def n_cached(self) -> int:
        return len(self._cache)

    def preload_all(self) -> None:
        """Load and validate every reference immediately; fail before training."""
        if (self.max_cached_references is not None
                and self.max_cached_references < len(self.paths)):
            raise ValueError("Full preload requires max_cached_references=None "
                             "or a capacity at least as large as the library")
        # Wrap the iteration with tqdm
        for path in tqdm(self.paths, desc="Preloading reference library"):
            self.get(path)

    def get(self, path: str | Path) -> LoopableReference:
        path = Path(path).resolve()
        if path not in self._known_paths:
            raise KeyError(f"Reference not indexed in library: {path}")
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        ref = LoopableReference(path, require_manifest=self.require_manifest)
        self._cache[path] = ref
        if self.max_cached_references is not None and len(self._cache) > self.max_cached_references:
            self._cache.popitem(last=False)
        return ref

    def choose(self, rng: np.random.Generator) -> LoopableReference:
        subjects = tuple(self.groups)
        subject = subjects[int(rng.integers(len(subjects)))]
        trials = tuple(self.groups[subject])
        trial = trials[int(rng.integers(len(trials)))]
        paths = self.groups[subject][trial]
        return self.get(paths[int(rng.integers(len(paths)))])


class BilateralMEP:
    def __init__(self, csv_path: str | Path):
        csv_path = Path(csv_path)
        with csv_path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames or []
            if "gait_cycle_pct" not in fields:
                raise ValueError("MEP CSV must contain 'gait_cycle_pct'")
            names = [name for name in fields if name != "gait_cycle_pct"]
            if len(names) != 80 or len(set(names)) != 80:
                raise ValueError("MEP CSV must contain exactly 80 unique muscle actuator columns")
            rows = list(reader)
        self.names = tuple(names)
        phase = np.asarray([float(row["gait_cycle_pct"]) / 100.0 for row in rows])
        values = np.asarray([[float(row[name]) for name in names] for row in rows])
        self.table = PeriodicTable(phase, values, "bilateral MEP")
        if np.any(values < -1e-5) or np.any(values > 1 + 1e-5):
            raise ValueError("MEP excitations must be in [0,1]")

    def sample(self, theta: float) -> np.ndarray:
        return self.table.sample(theta)


class ActivationLookup:
    """Load all duration profiles once; cache individual interpolator wrappers."""

    def __init__(self, path: str | Path, *, duration_tolerance_s: float = 0.005):
        if duration_tolerance_s < 0:
            raise ValueError("duration_tolerance_s must be >=0")
        self.duration_tolerance_s = float(duration_tolerance_s)
        with np.load(path, allow_pickle=False) as data:
            needed = ("durations_s", "phase_grid", "activation_profiles", "muscle_names")
            missing = [key for key in needed if key not in data]
            if missing:
                raise ValueError(f"Activation lookup missing keys: {missing}")
            self.durations = np.asarray(data["durations_s"], dtype=np.float64).reshape(-1)
            self.phase = np.asarray(data["phase_grid"], dtype=np.float64).reshape(-1)
            self.profiles = np.asarray(data["activation_profiles"], dtype=np.float64)
            self.names = tuple(str(x) for x in data["muscle_names"].tolist())
            self.saved_addresses = (np.asarray(data["act_addresses"], dtype=int).reshape(-1)
                                    if "act_addresses" in data else None)
        if (self.profiles.ndim != 3 or self.profiles.shape !=
                (len(self.durations), len(self.phase), len(self.names))):
            raise ValueError("Activation lookup inconsistent shape")
        if len(set(self.names)) != len(self.names) or len(self.names) != 80:
            raise ValueError("Activation lookup expected 80 unique actuator names")
        if self.saved_addresses is not None and len(self.saved_addresses) != len(self.names):
            raise ValueError("Activation lookup saved act_addresses has incorrect length")
        if (np.any(~np.isfinite(self.profiles)) or np.any(~np.isfinite(self.durations))
                or np.any(self.durations <= 0) or np.min(self.profiles) < -1e-5
                or np.max(self.profiles) > 1 + 1e-5):
            raise ValueError("Invalid activation profiles or stride durations")
        self._table_cache: dict[int, PeriodicTable] = {}

    def for_duration(self, duration: float) -> tuple[PeriodicTable, float]:
        idx = int(np.argmin(np.abs(self.durations - duration)))
        matched = float(self.durations[idx])
        if abs(matched - duration) > self.duration_tolerance_s:
            raise ValueError(f"No lookup for {duration:.6f}s; closest {matched:.6f}s "
                             f"(tolerance {self.duration_tolerance_s:.6f}s)")
        if idx not in self._table_cache:
            self._table_cache[idx] = PeriodicTable(self.phase, self.profiles[idx],
                                                   f"activation profile T={matched:.6f}s")
        return self._table_cache[idx], matched


class ReferenceAssets:
    """Reusable assets: preloaded kinematics, shared MEP and activation lookup."""

    def __init__(self, reference_root: str | Path, mep_csv: str | Path, lookup_npz: str | Path,
                 *, max_cached_references: int | None = None,
                 preload_references: bool | None = None,
                 duration_tolerance_s: float = 0.005):
        self.library = ReferenceLibrary(
            reference_root, max_cached_references=max_cached_references,
            preload=preload_references,
        )
        self.mep = BilateralMEP(mep_csv)
        self.lookup = ActivationLookup(lookup_npz, duration_tolerance_s=duration_tolerance_s)
        if set(self.mep.names) != set(self.lookup.names):
            raise ValueError("MEP and activation lookup have different actuator-name sets")

    def select(self, rng: np.random.Generator) -> LoopableReference:
        ref = self.library.choose(rng)
        self.lookup.for_duration(ref.duration)  # Fail early if the reference cannot be reset.
        return ref
