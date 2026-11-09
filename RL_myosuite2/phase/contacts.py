"""Step 3: four verified force sensors and debounced foot-contact events.

No sensor names are guessed: the MyoLeg XML must be inspected and the sensor
names supplied explicitly. Force order: right heel, right toe, left heel, left toe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

CONTACT_ORDER = ("right_heel", "right_toe", "left_heel", "left_toe")


class FourTouchSensorReader:
    def __init__(self, model: Any, mujoco: Any, names: Sequence[str]):
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError("Provide 4 distinct sensor names: RH, RT, LH, LT")
        addresses = []
        for name in names:
            idx = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name))
            if idx < 0:
                raise ValueError(f"No sensor named {name!r} in MyoLeg model")
            dim = int(model.sensor_dim[idx])
            if dim != 1:
                raise ValueError(f"Sensor {name!r} has dimension {dim}; expected scalar force")
            # Only touch sensors are guaranteed to encode normal contact force;
            # reject other kinds rather than silently interpreting a binary flag.
            touch = getattr(getattr(mujoco, "mjtSensor", None), "mjSENS_TOUCH", None)
            if touch is not None and int(model.sensor_type[idx]) != int(touch):
                raise ValueError(f"Sensor {name!r} is not a MuJoCo touch-force sensor")
            addresses.append(int(model.sensor_adr[idx]))
        self.names = tuple(names)
        self.addresses = np.asarray(addresses, dtype=int)

    def __call__(self, model: Any, data: Any) -> np.ndarray:
        forces = np.asarray(data.sensordata[self.addresses], dtype=float).copy()
        if forces.shape != (4,) or not np.all(np.isfinite(forces)):
            raise ValueError("Invalid four-contact force readings")
        return forces


@dataclass(frozen=True)
class ContactConfig:
    on_threshold: float = 20.0  # Newtons; tune once the model sensors are verified
    off_threshold: float = 10.0
    debounce_s: float = 0.005
    refractory_s: float = 0.08

    def __post_init__(self) -> None:
        if not (0 <= self.off_threshold < self.on_threshold and
                self.debounce_s >= 0 and self.refractory_s >= 0):
            raise ValueError("Invalid contact thresholds or timing")


class ContactEventDetector:
    """HS: foot goes airborne->contact; TO: last foot part loses contact.

    Contact sensing has hysteresis; foot stance transitions are debounced and
    refractory-filtered. RSI seeds contact state without emitting an event.
    """
    def __init__(self, config: ContactConfig = ContactConfig()):
        self.config = config
        self.active_sensors = np.zeros(4, dtype=bool)
        self.stance = np.zeros(2, dtype=bool)  # [right, left]
        self.candidates = self.stance.copy()
        self.candidate_since = np.zeros(2)
        self.last_change = np.full(2, -np.inf)
        self.is_seeded = False

    def _read(self, forces: np.ndarray) -> np.ndarray:
        f = np.asarray(forces, dtype=float)
        if f.shape != (4,) or np.any(~np.isfinite(f)) or np.any(f < -1e-6):
            raise ValueError("Expected nonnegative finite [RH, RT, LH, LT] forces")
        now = self.active_sensors.copy()
        now[(~now) & (f >= self.config.on_threshold)] = True
        now[now & (f <= self.config.off_threshold)] = False
        return now

    def seed(self, forces: np.ndarray, time_s: float = 0.0) -> None:
        f = np.asarray(forces, dtype=float)
        if f.shape != (4,) or not np.all(np.isfinite(f)):
            raise ValueError("RSI contact force seed must contain four values")
        self.active_sensors = f >= self.config.on_threshold
        self.stance = np.array([np.any(self.active_sensors[:2]),
                                np.any(self.active_sensors[2:])])
        self.candidates = self.stance.copy()
        self.candidate_since[:] = time_s
        self.last_change[:] = -np.inf
        self.is_seeded = True

    def update(self, forces: np.ndarray, time_s: float) -> tuple[str, ...]:
        if not self.is_seeded:
            raise RuntimeError("Contact detector must be seeded at RSI")
        if not np.isfinite(time_s):
            raise ValueError("Simulation time must be finite")
        self.active_sensors = self._read(forces)
        raw_stance = (bool(np.any(self.active_sensors[:2])),
                      bool(np.any(self.active_sensors[2:])))
        events = []
        for side, new_stance in enumerate(raw_stance):
            if new_stance != self.candidates[side]:
                self.candidates[side] = new_stance
                self.candidate_since[side] = time_s
            if (self.candidates[side] != self.stance[side] and
                    time_s - self.candidate_since[side] >= self.config.debounce_s - 1e-12 and
                    time_s - self.last_change[side] >= self.config.refractory_s - 1e-12):
                self.stance[side] = new_stance
                self.last_change[side] = time_s
                side_name = "right" if side == 0 else "left"
                events.append(f"{side_name}_{'hs' if new_stance else 'toe_off'}")
        return tuple(events)
