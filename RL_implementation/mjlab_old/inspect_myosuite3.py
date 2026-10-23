"""Real GPU/MyoSuite3 smoke test (requires NVIDIA GPU + mjlab install).

Example: python inspect_myosuite3.py --num-envs 16 --steps 5
"""
import argparse
import torch
from myosuite3_adapter import MyoSuite3LegAdapter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    sim = MyoSuite3LegAdapter(num_envs=args.num_envs, device=args.device)
    try:
        obs, _ = sim.reset()
        print('MyoSuite 3 mjlab:', type(sim.env).__name__)
        print('num_envs:', sim.num_envs, 'device:', sim.device, 'control_dt:', sim.dt)
        print('actuators:', len(sim.actuator_names), sim.actuator_names)
        print('sensors:', sim.sensor_report())
        print('obs groups:', {k: tuple(v.shape) for k, v in obs.items()})
        # Zero muscle excitation for an API test ONLY. Not meant to walk.
        actions = torch.zeros(sim.num_envs, 80, device=sim.device)
        for t in range(args.steps):
            obs, reward, terminated, timeout, info = sim.step(actions)
            contact = sim.get_foot_contact('r')
            print('step', t + 1, 'right contacts', int(contact.sum().item()),
                  'reward mean', float(reward.mean().item()),
                  'done', int((terminated | timeout).sum().item()))
    finally:
        sim.close()


if __name__ == '__main__':
    main()
