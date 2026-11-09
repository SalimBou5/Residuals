#!/usr/bin/env python3
"""Smoke test for Step 5: reward terms from a short, untrained MEP-only rollout.

This is NOT a realistic walking trial or an alternative to reference replay.
No PPO agent is involved; falling is expected.
"""
import argparse,json
from pathlib import Path
import numpy as np
import sys

# Add the root 'RL_myosuite2' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now use absolute imports from the root directory
from RSI.reference import ReferenceAssets
from control import ActionConfig
from observations import ObservationConfig
from env import MyoLegResidualEnv
from rewards import RewardConfig, TerminationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','references','mep','lookup'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--normalization',type=Path,default=None)
    p.add_argument('--reward-config',type=Path,default=None)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--enable-unmapped',action='store_true')
    p.add_argument('--no-event-sync',action='store_true')
    p.add_argument('--disable-tracking-termination',action='store_true')
    args=p.parse_args()
    assets=ReferenceAssets(args.references,args.mep,args.lookup)
    cfg=(RewardConfig(**json.loads(args.reward_config.read_text(encoding='utf-8')))
         if args.reward_config else RewardConfig())
    env=MyoLegResidualEnv(args.model,assets,
        action_config=ActionConfig(enable_unmapped=args.enable_unmapped),
        observation_config=ObservationConfig(normalization_file=str(args.normalization)
                                            if args.normalization else None),
        reward_config=cfg,
        termination_config=TerminationConfig(enable_tracking_termination=not args.disable_tracking_termination),
        event_sync=not args.no_event_sync,seed=args.seed)
    try:
        _,info=env.reset(seed=args.seed)
        print(f'Reference={info["reference"]} phase={info["reset_phase"]:.4f}')
        print(f'Actions={env.action_space.shape[0]} observations={env.observation_space.shape[0]}')
        a=np.zeros(env.action_space.shape,dtype=np.float32)
        if args.enable_unmapped:
            ids=env.core.actions.policy_to_actuator_ids
            a[:-1][env.rsi.binding.unmapped_mask[ids]]=-1.0
        for i in range(args.steps):
            obs,r,terminated,truncated,info=env.step(a)
            t=info['reward_terms']
            if i==0 or (i+1)%10==0 or terminated or truncated:
                print(f'k={i+1:4d} phi={info["phase"]:.3f} reward={r:+.4f} '
                      f'imit={t["imitation"]:.4f} speed={t["score_speed"]:.3f} '
                      f'upright={t["score_upright"]:.3f} '
                      f'penalty={t["total_penalty"]:.4f} '
                      f'nonfinite={not bool(np.all(np.isfinite(obs)))} '
                      f'stop={info["termination_reason"]}')
            if terminated or truncated:break
        print('No trained policy: these are diagnostic reward values, NOT walking-performance results.')
    finally:env.close()
if __name__=='__main__':main()
