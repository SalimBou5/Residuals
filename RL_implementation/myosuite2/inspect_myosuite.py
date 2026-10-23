"""Real MyoSuite connection smoke-check. Usage: python inspect_myosuite.py

Optional: python inspect_myosuite.py --step   (one random-control PHYSICS TEST,
not a biologically meaningful gait policy or part of RL training).
"""
import argparse
from myosuite_adapter import MyoSuiteLegAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', action='store_true')
    args = parser.parse_args()
    sim = MyoSuiteLegAdapter('myoLegWalk-v0')
    try:
        obs, info = sim.reset(seed=0)
        print('MyoSuite environment: myoLegWalk-v0')
        print('Control timestep (s):', sim.dt)
        print('Number of muscle actuators:', len(sim.actuator_names))
        print('Actuator names, in MyoLeg control order:', sim.actuator_names)
        print('Contact geometry candidates:', sim.contact_geometry_report())
        print('Right foot in contact:', sim.right_foot_in_contact())
        if args.step:
            # For API smoke testing ONLY: not the reference stimulation controller.
            a = sim.env.action_space.sample()
            obs, reward, terminated, truncated, info = sim.step(a)
            print('One MyoSuite physics step OK; terminated:', terminated,
                  'truncated:', truncated)
    finally:
        sim.close()


if __name__ == '__main__':
    main()
