"""Standalone smoke checks. Run: python test_skeleton.py"""
import tempfile
from pathlib import Path
import numpy as np
from gait_phase import PhaseConfig, GaitPhaseEstimator, FootContactEvents
from gait_reference import CHANNELS, GaitReference
from muscle_mapping import MuscleMapper, RIGHT_MAPPING
from rl_env import GaitRLEnv


def main():
    assert len(CHANNELS) == 15
    assert len(RIGHT_MAPPING) == 40
    phase = np.linspace(0, 1, 101, endpoint=False)
    a15 = np.tile(np.linspace(0.1, 0.9, 15), (len(phase), 1))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "synthetic_reference.npz"
        np.savez(path, phase=phase, muscle_activations=a15,
                 muscle_names=np.array(CHANNELS),
                 joint_angles=np.zeros((len(phase), 3)),
                 joint_velocities=np.zeros((len(phase), 3)))
        ref = GaitReference.from_npz(path)
        profiles = ref.at_phase(0.995)
        assert len(profiles["muscles_15"]) == 15
        assert profiles["joint_angles"].shape == (3,)

        names = ["tibant_r", "ehl_r", "glmin1_r", "vasint_r", "addbrev_r", "tibant_l"]
        mapper = MuscleMapper(names)
        expanded, unknown = mapper.expand(profiles["muscles_15"])
        assert np.isclose(expanded[0], expanded[1])
        assert np.isclose(expanded[2], (profiles["muscles_15"]["GlutMed"] +
                                             profiles["muscles_15"]["GlutMax"]) / 2)
        assert np.isclose(expanded[3], (profiles["muscles_15"]["VastLat"] +
                                             profiles["muscles_15"]["VastMed"]) / 2)
        assert unknown == ["addbrev_r", "tibant_l"]
        assert mapper.unused_reference_channels() == ["Add"]

        config = PhaseConfig(reference_stride_s=1.0, toe_off_phase=0.62)
        # 0.62 is SYNTHETIC TEST INPUT, not a physiological reference value.
        estimator = GaitPhaseEstimator(config)
        assert np.isclose(estimator.frequency_hz, 1.0)
        for i in range(400):
            estimator.update(0.01, heel_strike=(i % 100 == 99))
        assert estimator.valid_strides == 3
        assert np.isclose(estimator.confidence, 0.8)  # c_strides=.3 + c_timing=.5
        assert np.isclose(estimator.smoothed_stride_s, 1.0)
        detector = FootContactEvents()
        assert detector.update(False) == (False, False)
        assert detector.update(True) == (True, False)
        assert detector.update(True) == (False, False)
        assert detector.update(False) == (False, True)

        # Test-double only: actual MyoSuite connection is in inspect_myosuite.py.
        class FakeSim:
            actuator_names = names
            dt = 0.01

            def reset(self, seed=None):
                return np.array([1.0]), {}

            def right_foot_in_contact(self):
                return True

            def close(self):
                pass

        env = GaitRLEnv(ref, config, sim=FakeSim())
        env.reset()
        assert env.reference_targets()["unresolved_actuators"] == unknown

    print("OK: reference loading, periodic interpolation, mapping, phase estimation, events.")
    print("Explicit unresolved right/left assignments:", unknown)
    print("Reference channel unused until adductor mapping is corrected: Add")


if __name__ == "__main__":
    main()
