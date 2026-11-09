#!/usr/bin/env python3
"""Step 6: dynamic MEP-only MuJoCo check (not imposed reference replay).

Random references / phases -> set RSI qpos/qvel/act -> advance real mj_step.
Zero policy residual/phase actions throughout. Falling is a *result*, not a
reason to declare the script successful or unsuccessful by itself.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Callable

import numpy as np

from ppo_companion import EnvironmentSettings, create_environment, neutral_mep_action, validate_dynamic_step


def run_dynamic_episodes(settings: EnvironmentSettings, *, episodes: int,
                         output: str | Path, policy: Callable | None = None,
                         deterministic: bool = True) -> dict:
    """Simulate full physics episodes and save detailed CSV and JSON reports.

    policy=None means exact MEP-only control. A policy callable takes an
    observation and returns a bounded action (e.g. model.predict(obs)[0]).
    Creates exactly one model / reference cache for all diagnostic episodes.
    """
    if episodes < 1:
        raise ValueError('episodes must be >= 1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = create_environment(settings)
    steps, summaries = [], []
    try:
        for ep in range(episodes):
            obs, reset_info = env.reset(seed=settings.seed + ep)
            neutral = neutral_mep_action(env)
            start_phi = float(reset_info['reset_phase'])
            total_reward = 0.
            terminated = truncated = False
            last_info = None
            last_terms = {}
            number_events = 0
            number_accepted_events = 0
            max_clip_count = 0
            # Bound loop even if the simulator does not advance time correctly.
            limit = int(np.ceil(settings.max_episode_seconds / env.core.config.control_dt_s)) + 2
            for k in range(limit):
                if policy is None:
                    action = neutral
                else:
                    action = np.asarray(policy(obs), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                validate_dynamic_step(env, obs, reward, info, baseline_only=policy is None)
                if (not terminated and not truncated and
                        float(info['time_s']) >= settings.max_episode_seconds - 1e-9):
                    raise AssertionError('Episode time exceeded configured time limit')
                terms = info['reward_terms']
                force = env.reader(env.model, env.rsi.data)
                total_reward += float(reward)
                number_events += len(info['events'])
                number_accepted_events += sum(bool(e.get('accepted')) for e in info['events'])
                max_clip_count = max(max_clip_count, int(info.get('normalization_clipped_features', 0)))
                steps.append({
                    'episode': ep, 'step': k+1, 'time_s': info['time_s'],
                    'reference': reset_info['reference'], 'start_phase': start_phi,
                    'theta': info['theta'], 'phase': info['phase'],
                    'phase_drift': info['phase_drift'],
                    'phase_rate_hz': info['phase_rate_hz'],
                    'reward': reward,
                    'imitation': terms['imitation'],
                    'speed_score': terms['score_speed'],
                    'upright_score': terms['score_upright'],
                    'actual_speed_m_s': terms['actual_forward_speed_m_s'],
                    'desired_speed_m_s': terms['desired_forward_speed_m_s'],
                    'site_error_m': terms['mean_site_position_error_m'],
                    'root_error_m': terms['world_root_position_error_m'],
                    'residual_rms': terms['actual_applied_residual_rms'],
                    'activation_l1': terms['cost_activation_l1'],
                    'activation_l2': terms['cost_activation_l2'],
                    'penalty': terms['total_penalty'],
                    'force_r_heel_N': float(force[0]),
                    'force_r_toe_N': float(force[1]),
                    'force_l_heel_N': float(force[2]),
                    'force_l_toe_N': float(force[3]),
                    'events': '|'.join(str(e['event']) for e in info['events']),
                    'events_accepted': sum(bool(e.get('accepted')) for e in info['events']),
                    'normalization_clipped_features': info.get('normalization_clipped_features', 0),
                    'terminated': int(terminated), 'truncated': int(truncated),
                    'end_reason': info.get('termination_reason') or info.get('truncation_reason') or '',
                })
                last_info, last_terms = info, terms
                if terminated or truncated:
                    break
            else:
                raise RuntimeError('Simulation did not terminate/truncate before the hard step limit')
            summaries.append({
                'episode': ep,
                'reference': reset_info['reference'],
                'reset_phase': start_phi,
                'reset_penetration_mm': 1000*float(reset_info['max_reset_penetration_m']),
                'steps': k+1,
                'simulated_seconds': float(last_info['time_s']),
                'return': total_reward,
                'mean_reward': total_reward/(k+1),
                'last_site_error_m': float(last_terms['mean_site_position_error_m']),
                'last_upright_dot': float(last_terms['upright_dot']),
                'last_forward_speed_m_s': float(last_terms['actual_forward_speed_m_s']),
                'accepted_events': number_accepted_events,
                'observed_events': number_events,
                'max_normalization_clipped_features': max_clip_count,
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'reason': last_info.get('termination_reason') or last_info.get('truncation_reason') or 'unknown',
            })
    finally:
        env.close()
    if steps:
        with (output/'dynamic_steps.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(steps[0]))
            writer.writeheader()
            writer.writerows(steps)
    with (output/'episodes.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        'health_checks_passed': True,
        'mode': 'MEP_only' if policy is None else 'trained_policy',
        'physics_integrated': True,
        'reference_kinematics_not_imposed_after_reset': True,
        'settings': asdict(settings),
        'episodes': summaries,
        'summary': {
            'episodes': len(summaries),
            'mean_episode_seconds': float(np.mean([s['simulated_seconds'] for s in summaries])),
            'mean_episode_return': float(np.mean([s['return'] for s in summaries])),
            'terminated': sum(s['terminated'] for s in summaries),
            'truncated': sum(s['truncated'] for s in summaries),
            'mean_end_upright_dot': float(np.mean([s['last_upright_dot'] for s in summaries])),
            'max_normalization_clipped_features': max(s['max_normalization_clipped_features'] for s in summaries),
        },
        'interpretation': 'Safety checks passing does not mean the MEP baseline walks stably. Falls, short episodes and large tracking errors must be inspected before trusting PPO results.',
    }
    (output/'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ('model','references','mep','lookup','normalization','reward-config'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('runs/mep_smoke'))
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--seconds', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--cache-size', type=int, default=16)
    p.add_argument('--disable-event-sync', action='store_true')
    p.add_argument('--enable-unmapped', action='store_true')
    p.add_argument('--disable-tracking-termination', action='store_true')
    return p


def settings_from_args(args: argparse.Namespace) -> EnvironmentSettings:
    return EnvironmentSettings(
        model=str(args.model.resolve()), references=str(args.references.resolve()),
        mep=str(args.mep.resolve()), lookup=str(args.lookup.resolve()),
        normalization=str(args.normalization.resolve()),
        reward_config=str(args.reward_config.resolve()), seed=args.seed,
        max_episode_seconds=args.seconds,
        reference_cache_size=args.cache_size,
        event_sync=not args.disable_event_sync,
        enable_unmapped=args.enable_unmapped,
        enable_tracking_termination=not args.disable_tracking_termination,
    )


def main() -> None:
    args = build_parser().parse_args()
    settings = settings_from_args(args)
    settings.verify_paths()
    report = run_dynamic_episodes(settings, episodes=args.episodes, output=args.output)
    print(json.dumps(report['summary'], indent=2))
    print('Saved', args.output/'dynamic_steps.csv', args.output/'episodes.csv', args.output/'summary.json')
    print('MEP-only rollout: falls may be expected; inspect the distributions and reasons.')


if __name__ == '__main__':
    main()
