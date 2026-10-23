"""Event-corrected, normalized gait phase estimator (one reference leg).

Phase is in [0, 1), and frequency is in cycles/second, NOT radians/second.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhaseConfig:
    reference_stride_s: float
    toe_off_phase: float   # From the reference gait, not an assumed universal constant.
    alpha: float = 0.3    # Smoothed stride-duration update; initial working value.
    k_hs: float = 1     # Partial heel-strike correction; tune later.
    k_to: float = 0.5     # Partial toe-off correction; tune later.
    min_stride_s: float = 0.5  # Sanity bounds: CONFIGURATION PLACEHOLDERS.
    max_stride_s: float = 2

    def __post_init__(self):
        if self.reference_stride_s <= 0:
            raise ValueError("reference_stride_s must be positive")
        if not 0.0 <= self.toe_off_phase < 1.0:
            raise ValueError("toe_off_phase must be in [0, 1)")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if not (0 <= self.k_hs <= 1 and 0 <= self.k_to <= 1):
            raise ValueError("Event-correction gains must be in [0, 1]")
        if not (0 < self.min_stride_s < self.max_stride_s):
            raise ValueError("Invalid stride-time sanity bounds")


def circular_error(target: float, current: float) -> float:
    """Shortest signed phase difference measured in gait cycles."""
    return (target - current + 0.5) % 1.0 - 0.5


class GaitPhaseEstimator:
    def __init__(self, config: PhaseConfig):
        self.cfg = config
        self.reset()

    '''
    - self.phase : current phase
    - self.elapsed_s : time elapsed in the current stride
    - self.smoothed_stride_s : smoothed stride duration 
    - self.valid_strides : how many acceptable consecutive strides
    - self.confidence : how much the estimate is trusted (will evolve in time)
    - self.last_stride_s : duration of the last stride
    '''
    def reset(self, initial_phase: float = 0.0):
        self.phase = initial_phase if initial_phase < 1 else initial_phase - 1
        self.elapsed_s = 0.0  
        self.smoothed_stride_s = self.cfg.reference_stride_s
        self.valid_strides = 0
        self.confidence = 0.0
        self.last_stride_s = None

    @property
    def frequency_hz(self) -> float:
        '''
            This function defines the phase rate dot{phi_t} that will be the base of the evolution of the gait phase 
            (phi_{t+1} = phi_t + dot{phi_t} * control_timestep)
            phi_t should ideally represent the gait phase in which our model is which is not necessarily the reference one
            --> Therefore dot{phi_t} should ideally represent the phase rate of the model.
            However, the model will start from a completely unstable gait and should learn until finding a stable gait.
            When it is unstable, we cannot rely on the model phase 
                --> solution : consider the reference one (which is the same as giving non-normalized reference kinematics and muscle activations)
                --> But then it's too rigid and we lose the real gait phase we want to get
                    --> Solution: as soon as we can trust the model we take it
            --> FINAL SOLUTION: a confidence blending of both rates (reference and estimated from the model) 
            At first confidence will be 0, then it will improve depending in the how many strides we have and how much time does it take. 
        '''
        f_ref = 1.0 / self.cfg.reference_stride_s  # reference frequency
        f_est = 1.0 / self.smoothed_stride_s       # estimated frequency from the smoothed stride duration
        return (1.0 - self.confidence) * f_ref + self.confidence * f_est  

    def _update_confidence(self):
        n = self.valid_strides
        c_strides = n * 0.1 if n < 4 else 0.5 # confidence increases with the number of strides

        e_t = abs(self.smoothed_stride_s - self.cfg.reference_stride_s)  # duration error between the reference stride and the actual strides
        if e_t > 0.4:
            c_timing = 0.0
        elif e_t > 0.3:
            c_timing = 0.2
        elif e_t > 0.15:
            c_timing = 0.3
        else:
            c_timing = 0.5
        self.confidence = min(1.0, c_strides + c_timing)

    def update(self, dt: float, heel_strike: bool = False, toe_off: bool = False) -> float:
        """Advance the phase then correct it with newly observed contact events.

        heel_strike and toe_off should be *event edges*, not continuous contact.
        Both events refer to the same (reference) leg.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if heel_strike and toe_off:
            raise ValueError("Heel strike and toe off cannot be simultaneous")

        self.phase = (self.phase + self.frequency_hz * dt) % 1.0  # phi_{t+1} = phi_t + dot{phi_t} * control_timestep
        self.elapsed_s += dt

        if heel_strike:
            stride_s = self.elapsed_s 
            # A stride is considered valid if the stride time is reasonable (min * ref_time < stride_time < max * ref_time, with min = 0.5 and max = 2)
            if self.cfg.min_stride_s * self.cfg.reference_stride_s <= stride_s <= self.cfg.max_stride_s * self.cfg.reference_stride_s:
                self.last_stride_s = stride_s
                a = self.cfg.alpha
                # Update the stride time in a smoothed way, *a* determines how fast the stride time estimate adapts to the new value
                self.smoothed_stride_s = ((1.0 - a) * self.smoothed_stride_s + a * stride_s)
                self.valid_strides += 1
            else:
                # Implausible event timing: do not train the cadence estimate on it.
                self.valid_strides = 0
                self.last_stride_s = None

            self._update_confidence()
            self.elapsed_s = 0
            
            # If we have HS, we want the phase to be 0 but we might not want to have an abrupt change. That's why this calculation exists
            # That said, k_hs is 1 now, which is exactly as just putting the phase directly to 0 (because k_hs should not be noisy)
            self.phase = (self.phase + self.cfg.k_hs * circular_error(0.0, self.phase)) % 1.0

        elif toe_off:
            # If we have TO, we hypothesize that the phase should be around the reference toe off phase
            # but we don't want to have an abrupt change --> we go through this calculation
            self.phase = (self.phase + self.cfg.k_to * circular_error(self.cfg.toe_off_phase, self.phase)) % 1.0
            # If toe-off corrections are too weak and stance/swing alignment is still drifting, increase toward k_to = 0.7-1
            # else if toe-off causes abrupt jumps in the reference and destabilizes learning, reduce toward k_t0 = 0.2-0.4
        return self.phase


class FootContactEvents:
    """Simple *placeholder* contact-edge detection for a chosen leg.

    The model-specific extraction of the boolean contact state is left to the user.
    """

    def __init__(self):
        self.previous_contact = None

    def reset(self):
        self.previous_contact = None

    def update(self, in_contact: bool) -> tuple[bool, bool]:
        contact = bool(in_contact)
        if self.previous_contact is None:
            self.previous_contact = contact
            return False, False  # Initial state is not a detected event.
        hs = contact and not self.previous_contact
        to = not contact and self.previous_contact
        self.previous_contact = contact
        return hs, to

class FootContactEvents:
    """Detect heel strike and toe-off from MyoLeg ground reaction forces."""

    # The threshold 20.0 N follows the HS and TO extraction of the Gait120 dataset
    def __init__(self, model, side="right", threshold_N=20.0):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        if threshold_N <= 0:
            raise ValueError("threshold_N must be positive")

        self.model = model
        self.threshold_N = float(threshold_N)

        prefix = "r" if side == "right" else "l"

        # MyoLeg foot and toe collision geometries
        foot_names = (
            f"{prefix}_foot_col1",
            f"{prefix}_foot_col3",
            f"{prefix}_foot_col4",
            f"{prefix}_bofoot_col1",
            f"{prefix}_bofoot_col2",
        )

        self.foot_geom_ids = set()
        for name in foot_names:
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, name
            )
            if geom_id < 0:
                raise ValueError(f"Missing collision geom: {name}")
            self.foot_geom_ids.add(geom_id)

        self.ground_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
        )
        if self.ground_geom_id < 0:
            raise ValueError("Missing ground geom: terrain")

        self.reset()

    def reset(self):
        self.previous_contact = None
        self.vertical_grf_N = 0.0

    def measure_vertical_grf(self, data):
        """Sum vertical normal-force contributions at foot-ground contacts."""
        total = 0.0
        wrench = np.zeros(6)

        for i in range(data.ncon):
            contact = data.contact[i]

            is_foot_ground = (
                (contact.geom1 == self.ground_geom_id
                 and contact.geom2 in self.foot_geom_ids)
                or
                (contact.geom2 == self.ground_geom_id
                 and contact.geom1 in self.foot_geom_ids)
            )

            if not is_foot_ground:
                continue

            mujoco.mj_contactForce(self.model, data, i, wrench)

            # Normal force projected onto the vertical (world Z).
            # For flat ground, this equals the vertical GRF.
            normal_z = abs(
                float(np.asarray(contact.frame).reshape(3, 3)[0, 2])
            )
            total += max(0.0, float(wrench[0])) * normal_z

        return total

    def update(self, data) -> tuple[bool, bool]:
        self.vertical_grf_N = self.measure_vertical_grf(data)
        in_contact = self.vertical_grf_N >= self.threshold_N

        if self.previous_contact is None:
            self.previous_contact = in_contact
            return False, False

        heel_strike = in_contact and not self.previous_contact
        toe_off = not in_contact and self.previous_contact

        self.previous_contact = in_contact

        return heel_strike, toe_off
