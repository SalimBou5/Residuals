"""Live mjlab smoke test for the phase-indexed imitation-reward pipeline.

Requires a REAL gait .npz containing `phase`, `muscle_activations`, `muscle_names`,
`joint_angles`, `joint_velocities`, and `joint_names` mapped to MyoLeg units.
Only runs a few simulation steps; this is NOT PPO training.
"""
from __future__ import annotations
import argparse
import torch

from gait_reference import GaitReference
from gait_phase import PhaseConfig
from rsi import RSIConfig
from rl_env import GaitRLEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--stride', type=float, required=True,
                        help='Measured reference stride duration in seconds')
    parser.add_argument('--toe-off', type=float, required=True,
                        help='Measured normalized RIGHT-foot toe-off in [0,1)')
    parser.add_argument('--reset-phase', type=float, required=True,
                        help='Reference phase matching MyoSuite default reset keyframe')
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    reference = GaitReference.from_npz(args.reference)
    cfg = PhaseConfig(reference_stride_s=args.stride, toe_off_phase=args.toe_off)
    env = GaitRLEnv(reference, cfg, num_envs=args.num_envs, device=args.device,
                    rsi_config=RSIConfig(mode='standard', initial_phase=args.reset_phase),
                    )
    try:
        obs, _ = env.reset()
        print('actor obs', tuple(obs['actor'].shape),
              'mapped muscle count', len(env.mapper.actuator_names),
              'provisional mappings', len(env.mapper.provisional))
        # No trained agent yet: ZERO residual means base feedforward only.
        raw_policy_action = torch.zeros((env.num_envs, env.action_dim),
                                        device=env.device)
        for i in range(args.steps):
            obs, reward, terminated, truncated, extras = env.step(raw_policy_action)
            diagnostics = extras['imitation_terms']
            print(f'{i+1}: imitation reward={reward.mean().item():.5f}, '
                  f'q_MSE={diagnostics.get("q_mse", torch.tensor(float("nan"))).mean().item():.5f}, '
                  f'phase={extras["phase"].mean().item():.3f}, '
                  f'done={(terminated | truncated).sum().item()}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
