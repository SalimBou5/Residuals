from pathlib import Path
import hashlib
import json
import csv

import numpy as np
import pandas as pd
import mujoco


# ============================================================
# 0. Configuration
# ============================================================

HERE = Path(__file__).resolve().parent

MODEL_PATH = "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml"

MEP_PATH = HERE / "outputs/data/myoleg_bilateral_mep_80.csv"

REFERENCE_DIR = (
    HERE / "../references_gait_120_standardized"
)

OUTPUT_DIR = HERE / "outputs/data/activation_profiles"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKUP_PATH = OUTPUT_DIR / "activation_lookup.npz"
MANIFEST_INDEX_PATH = OUTPUT_DIR / "manifest_lookup.csv"

TOL = 1e-6
MAX_CYCLES = 30

# Numerical integration uses the smaller of:
# - MyoLeg physics timestep
# - one tenth of the shortest activation/deactivation constant
DT_FACTOR = 10.0


# ============================================================
# 1. Load MyoLeg model and MEP reference
# ============================================================

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
mep = pd.read_csv(MEP_PATH)

phase_grid = mep["gait_cycle_pct"].to_numpy(dtype=float) / 100.0

muscle_names = [
    c for c in mep.columns if c != "gait_cycle_pct"
]

u_base = mep[muscle_names].to_numpy(dtype=float)

if len(muscle_names) != 80:
    raise ValueError(f"Expected 80 muscles, found {len(muscle_names)}")

if (
    not np.isclose(phase_grid[0], 0)
    or not np.isclose(phase_grid[-1], 1)
    or not np.all(np.diff(phase_grid) > 0)
):
    raise ValueError("Invalid gait phase grid")

if not np.all(np.isfinite(u_base)):
    raise ValueError("MEP contains non-finite values")

if u_base.min() < 0 or u_base.max() > 1:
    raise ValueError("MEP values must lie in [0, 1]")


# ============================================================
# 2. Extract actual MyoLeg muscle activation parameters
# ============================================================

actuator_ids = []
act_addresses = []
dyn_params = []
act_limits = []

for name in muscle_names:

    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name
    )

    if idx < 0:
        raise ValueError(f"Muscle not found in MyoLeg: {name}")

    if model.actuator_dyntype[idx] != mujoco.mjtDyn.mjDYN_MUSCLE:
        raise ValueError(
            f"{name} does not use MuJoCo muscle activation dynamics"
        )

    if model.actuator_actnum[idx] != 1:
        raise ValueError(f"Unexpected activation dimension: {name}")

    actuator_ids.append(idx)
    act_addresses.append(model.actuator_actadr[idx])

    # tau_act, tau_deact, tausmooth
    dyn_params.append(model.actuator_dynprm[idx, :3].copy())

    if model.actuator_actlimited[idx]:
        act_limits.append(model.actuator_actrange[idx].copy())
    else:
        act_limits.append([0.0, 1.0])

actuator_ids = np.asarray(actuator_ids, dtype=int)
act_addresses = np.asarray(act_addresses, dtype=int)
dyn_params = np.asarray(dyn_params, dtype=float)
act_limits = np.asarray(act_limits, dtype=float)

if len(np.unique(act_addresses)) != 80:
    raise ValueError("Duplicate muscle activation-state addresses")

if np.any(dyn_params[:, :2] <= 0):
    raise ValueError("Invalid muscle activation time constants")

dt = min(
    float(model.opt.timestep),
    float(np.min(dyn_params[:, :2])) / DT_FACTOR
)

print(f"Loaded {len(muscle_names)} muscles")
print(f"Integration timestep: {dt:.6f} s")


# ============================================================
# 3. Periodic excitation interpolation
# ============================================================

def excitation_at_phases(phases):
    """
    Evaluate all 80 excitation profiles at arbitrary gait phases.

    phases: (N,)
    returns: (N, 80)

    The duplicate endpoint at phase 1 is removed, and periodic
    interpolation connects the final samples back to phase 0.
    """

    phases = np.asarray(phases, dtype=float) % 1.0

    return np.column_stack([
        np.interp(
            phases,
            phase_grid[:-1],
            u_base[:-1, j],
            period=1.0
        )
        for j in range(len(muscle_names))
    ])


# ============================================================
# 4. Integrate muscle activation dynamics over one cycle
# ============================================================

def integrate_cycle(a0, excitations, h):
    """
    Integrate one complete cycle using MuJoCo's muscle
    activation dynamics:

        da/dt = mju_muscleDynamics(u, a, dyn_params)

    Numerical integration: explicit Euler.

    No skeletal dynamics are simulated.
    """

    n_steps = len(excitations)
    n_muscles = len(muscle_names)

    activations = np.empty(
        (n_steps + 1, n_muscles),
        dtype=float
    )

    activations[0] = a0

    for k in range(n_steps):

        a = activations[k]
        u = excitations[k]

        da = np.array([
            mujoco.mju_muscleDynamics(
                float(u[j]),
                float(a[j]),
                dyn_params[j]
            )
            for j in range(n_muscles)
        ])

        a_next = a + h * da

        activations[k + 1] = np.clip(
            a_next,
            act_limits[:, 0],
            act_limits[:, 1]
        )

    return activations


# ============================================================
# 5. Converge to periodic activation
# ============================================================

def converge_activation(excitations, h, initial_activation):
    """
    Repeat cycles until:
      1. Two consecutive activation trajectories agree.
      2. The beginning and end of the cycle agree.
    """

    a0 = initial_activation.copy()

    previous_cycle = None

    for cycle in range(1, MAX_CYCLES + 1):

        current_cycle = integrate_cycle(a0, excitations, h)

        boundary_error = np.max(
            np.abs(current_cycle[-1] - current_cycle[0])
        )

        if previous_cycle is not None:

            cycle_error = np.max(
                np.abs(current_cycle - previous_cycle)
            )

            if cycle_error < TOL and boundary_error < TOL:

                return current_cycle, cycle

        previous_cycle = current_cycle

        # Continue from the previous cycle's final activation
        a0 = current_cycle[-1].copy()

    raise RuntimeError(
        f"Activation did not converge after {MAX_CYCLES} cycles"
    )


# ============================================================
# 6. Compute one periodic activation lookup for a duration
# ============================================================

def compute_activation_lookup(T):
    """
    Calculate the periodic activation profile for duration T.

    Verify that two different initial activation conditions
    converge to the same periodic profile.

    Return shape: (n_phase_samples, 80)
    """

    n_steps = int(np.ceil(T / dt))
    h = T / n_steps

    # Precompute excitation once, reused for every cycle
    integration_phases = np.linspace(
        0.0, 1.0, n_steps + 1
    )

    excitations = excitation_at_phases(
        integration_phases[:-1]
    )

    # Initial condition 1: minimum activation
    act_low, cycles_low = converge_activation(
        excitations,
        h,
        act_limits[:, 0]
    )

    # Initial condition 2: maximum activation
    act_high, cycles_high = converge_activation(
        excitations,
        h,
        act_limits[:, 1]
    )

    # Verify independence from initial conditions
    initial_condition_error = np.max(
        np.abs(act_low - act_high)
    )

    if initial_condition_error > TOL:
        raise RuntimeError(
            f"Initial-condition convergence failed for T={T:.6f}s: "
            f"error={initial_condition_error:.2e}"
        )

    # Interpolate the converged activation to the MEP phase grid
    activation_lookup = np.column_stack([
        np.interp(
            phase_grid,
            integration_phases,
            act_low[:, j]
        )
        for j in range(len(muscle_names))
    ])

    periodicity_error = np.max(
        np.abs(activation_lookup[-1] - activation_lookup[0])
    )

    if periodicity_error > TOL:
        raise RuntimeError(
            f"Periodicity check failed for T={T:.6f}s"
        )

    print(
        f"T={T:.6f}s | "
        f"cycles={cycles_low}/{cycles_high} | "
        f"IC error={initial_condition_error:.2e} | "
        f"periodicity error={periodicity_error:.2e}"
    )

    return activation_lookup


# ============================================================
# 7. Scan all subject/trial/step manifests
# ============================================================

def collect_manifest_durations():

    manifest_paths = sorted(
        REFERENCE_DIR.rglob("step0[12]_manifest.json")
    )

    if not manifest_paths:
        raise FileNotFoundError(
            f"No step01/step02 manifests found in {REFERENCE_DIR}"
        )

    records = []

    for path in manifest_paths:

        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if "stride_duration_s" not in manifest:
            raise KeyError(
                f"Missing stride_duration_s in {path}"
            )

        T = float(manifest["stride_duration_s"])

        if not np.isfinite(T) or T <= 0:
            raise ValueError(f"Invalid stride duration in {path}")

        records.append({
            "manifest": path.relative_to(REFERENCE_DIR).as_posix(),
            "duration": T
        })

    return records


# ============================================================
# 8. Cache validation
# ============================================================

def compute_cache_fingerprint():
    """
    Fingerprint all information affecting the activation
    calculation, so previously saved profiles can be reused
    safely when their inputs have not changed.
    """

    sha = hashlib.sha256()

    sha.update(b"activation_lookup_euler_v1")

    for array in (
        phase_grid,
        u_base,
        dyn_params,
        act_limits,
        act_addresses
    ):
        sha.update(np.ascontiguousarray(array).tobytes())

    sha.update(json.dumps(muscle_names).encode("utf-8"))

    sha.update(
        np.array(
            [dt, TOL, MAX_CYCLES],
            dtype=np.float64
        ).tobytes()
    )

    return sha.hexdigest()


def load_existing_cache(fingerprint):
    """
    Return {duration: activation_profile} for compatible
    previously computed profiles.
    """

    cache = {}

    if not LOOKUP_PATH.exists():
        return cache

    with np.load(LOOKUP_PATH, allow_pickle=False) as saved:

        if "fingerprint" not in saved.files:
            return cache

        if str(saved["fingerprint"].item()) != fingerprint:
            print("Existing cache is incompatible. Recomputing.")
            return cache

        durations = saved["durations_s"]
        profiles = saved["activation_profiles"]

        if profiles.shape != (
            len(durations), len(phase_grid), len(muscle_names)
        ):
            return cache

        for T, profile in zip(durations, profiles):
            cache[float(T)] = profile.copy()

    return cache


# ============================================================
# 9. Build unique-duration activation lookup table
# ============================================================

def main():

    records = collect_manifest_durations()

    unique_durations = sorted({
        record["duration"] for record in records
    })

    print(f"\nTotal manifests: {len(records)}")
    print(f"Unique stride durations: {len(unique_durations)}")

    fingerprint = compute_cache_fingerprint()

    lookup = load_existing_cache(fingerprint)

    reused = 0
    computed = 0

    for i, T in enumerate(unique_durations, start=1):

        if T in lookup:
            reused += 1
            print(
                f"[{i}/{len(unique_durations)}] "
                f"T={T:.6f}s — cached"
            )
            continue

        print(
            f"[{i}/{len(unique_durations)}] "
            f"Computing T={T:.6f}s"
        )

        lookup[T] = compute_activation_lookup(T)
        computed += 1

    # --------------------------------------------------------
    # Save one combined activation lookup table
    # --------------------------------------------------------

    activation_profiles = np.stack([
        lookup[T] for T in unique_durations
    ])

    # Shape:
    # (n_unique_durations, n_phase_samples, 80)

    np.savez_compressed(
        LOOKUP_PATH,
        durations_s=np.asarray(unique_durations),
        phase_grid=phase_grid,
        activation_profiles=activation_profiles,
        muscle_names=np.asarray(muscle_names),
        actuator_ids=actuator_ids,
        act_addresses=act_addresses,
        fingerprint=np.asarray(fingerprint),
        integration_dt=dt,
        convergence_tolerance=TOL
    )

    # --------------------------------------------------------
    # Map every manifest to its lookup index
    # --------------------------------------------------------

    duration_to_index = {
        T: i for i, T in enumerate(unique_durations)
    }

    with open(
        MANIFEST_INDEX_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "manifest",
            "stride_duration_s",
            "lookup_index"
        ])

        for record in records:

            writer.writerow([
                record["manifest"],
                record["duration"],
                duration_to_index[record["duration"]]
            ])

    print("\n===================================")
    print("Activation lookup completed")
    print("===================================")
    print(f"Step manifests: {len(records)}")
    print(f"Unique durations: {len(unique_durations)}")
    print(f"Profiles computed: {computed}")
    print(f"Profiles reused: {reused}")
    print(f"Lookup shape: {activation_profiles.shape}")
    print(f"Saved: {LOOKUP_PATH}")
    print(f"Saved: {MANIFEST_INDEX_PATH}")


if __name__ == "__main__":
    main()