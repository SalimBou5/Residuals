#!/usr/bin/env python3
"""Evaluate a saved Step-6 PPO checkpoint with actual muscle-driven physics.

Reports step-by-step site/speed/upright errors, episode failure reasons and
contact-force histories; never overwrites the reference trajectories.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from smoke_test_mep import build_parser, settings_from_args, run_dynamic_episodes

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main() -> None:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--stochastic', action='store_true',help='Sample stochastic PPO actions instead of using policy mean')
    parser.set_defaults(output=Path('runs/ppo_evaluation'),episodes=16,seconds=4.0)
    args = parser.parse_args()
    settings = settings_from_args(args)
    settings.verify_paths()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit('Please install Step-6 dependencies: pip install -r requirements_step6.txt') from exc
    model = PPO.load(str(args.checkpoint), device='cpu')
    expected_obs = (505 if args.enable_unmapped else 485,)
    expected_act = (81 if args.enable_unmapped else 71,)
    if model.observation_space.shape != expected_obs or model.action_space.shape != expected_act:
        raise ValueError('Checkpoint dimensions do not match current observation/action configuration')
    def policy(obs):
        return model.predict(obs, deterministic=not args.stochastic)[0]
    report = run_dynamic_episodes(settings, episodes=args.episodes, output=args.output, policy=policy)
    print(json.dumps(report['summary'], indent=2))
    print('Detailed dynamic evaluation:', args.output.resolve())


if __name__ == '__main__':
    main()
