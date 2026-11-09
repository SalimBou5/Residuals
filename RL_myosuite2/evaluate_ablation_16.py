#!/usr/bin/env python3
r"""Headless, paired 4-mode evaluation of the SAME MyoLeg residual PPO policy.

Uses the exact 16 reference paths/reset phases from a Step-6 evaluation JSON.
Does not render, train, disable event synchronization, or relax termination.
Produces CSVs for both individual episodes and aggregate performance.

Example (PowerShell, from the RL_myosuite2 project root):

python evaluate_ablation_16.py `
  --summary evaluation_resets_10m.json `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from ppo_companion import create_environment, EnvironmentSettings, neutral_mep_action, validate_dynamic_step

MODES = ('mep', 'phase-only', 'muscle-only', 'full')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--summary', type=Path, required=True,
                   help='The existing 16-episode evaluation JSON (same as viewer --summary)')
    p.add_argument('--checkpoint', type=Path, required=True, help='Trained SB3 PPO .zip')
    p.add_argument('--output-dir', type=Path, default=Path('runs/ppo_ablation_16'))
    p.add_argument('--episodes', type=int, default=16)
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    return p


def policy_action(mode, model, obs, neutral):
    if mode == 'mep':
        return neutral.copy()
    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if action.shape != neutral.shape:
        raise ValueError(f'Policy produces {action.shape}, expected {neutral.shape}')
    np.clip(action, -1.0, 1.0, out=action)
    if mode == 'muscle-only':
        action[-1] = neutral[-1]
    elif mode == 'phase-only':
        action[:-1] = neutral[:-1]
    return action


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parser().parse_args()
    if args.episodes < 1:
        raise ValueError('--episodes must be at least 1')
    supplied = json.loads(args.summary.read_text(encoding='utf-8'))
    if not isinstance(supplied.get('episodes'), list) or 'settings' not in supplied:
        raise ValueError('Expected evaluation report with settings and episodes')
    if args.episodes > len(supplied['episodes']):
        raise ValueError(f'Report contains only {len(supplied["episodes"])} episodes')
    selected = supplied['episodes'][:args.episodes]
    settings = EnvironmentSettings(**supplied['settings'])
    settings.verify_paths()
    from stable_baselines3 import PPO
    policy = PPO.load(str(args.checkpoint), device='cpu')
    env = create_environment(settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    try:
        if env.action_space.shape != policy.action_space.shape or env.observation_space.shape != policy.observation_space.shape:
            raise ValueError('Checkpoint action/observation spaces do not match this environment')
        print(f'Event sync: {settings.event_sync} | Tracking termination: {settings.enable_tracking_termination}')
        print(f'Using {args.episodes} recorded reset phases/references across {len(args.modes)} modes; deterministic policy')
        for mode in args.modes:
            print(f'\n===== {mode} =====', flush=True)
            for ep, record in enumerate(selected):
                desired_ref = Path(record['reference'])
                desired_phase = float(record['reset_phase'])
                obs, reset_info = env.reset(
                    seed=int(settings.seed) + ep,
                    options={'reference': str(desired_ref), 'phase': desired_phase})
                actual_ref = Path(reset_info['reference'])
                if actual_ref.resolve() != desired_ref.resolve():
                    raise AssertionError(f'Wrong reference: {actual_ref} != {desired_ref}')
                if abs(float(reset_info['reset_phase']) - desired_phase) > 1e-9:
                    raise AssertionError('Reset phase was not reproduced exactly')
                neutral = neutral_mep_action(env)
                rewards = []
                site_errors = []
                root_errors = []
                speeds = []
                upright = []
                residuals = []
                observed_events = accepted_events = 0
                reason = None
                t = 0.0
                for step in range(1, int(np.ceil(settings.max_episode_seconds / env.core.config.control_dt_s)) + 4):
                    action = policy_action(mode, policy, obs, neutral)
                    obs, reward, terminated, truncated, info = env.step(action)
                    validate_dynamic_step(env, obs, reward, info, baseline_only=(mode == 'mep'))
                    terms = info['reward_terms']
                    rewards.append(float(reward))
                    site_errors.append(float(terms['mean_site_position_error_m']))
                    root_errors.append(float(terms['world_root_position_error_m']))
                    speeds.append(float(terms['actual_forward_speed_m_s']))
                    upright.append(float(terms['upright_dot']))
                    residuals.append(float(terms['actual_applied_residual_rms']))
                    observed_events += len(info.get('events', ()))
                    accepted_events += sum(bool(e.get('accepted', False)) for e in info.get('events', ()))
                    t = float(info['time_s'])
                    if terminated or truncated:
                        reason = info.get('termination_reason') or info.get('truncation_reason') or 'unknown'
                        break
                else:
                    raise RuntimeError('Episode exceeded time limit without termination/truncation')
                row = {
                    'mode': mode,
                    'episode': ep,
                    'reference': str(desired_ref),
                    'reset_phase': desired_phase,
                    'duration_s': t,
                    'steps': len(rewards),
                    'total_reward': sum(rewards),
                    'mean_reward_per_step': mean(rewards),
                    'mean_site_error_m': mean(site_errors),
                    'final_site_error_m': site_errors[-1],
                    'mean_world_root_error_m': mean(root_errors),
                    'mean_forward_speed_m_s': mean(speeds),
                    'mean_upright_dot': mean(upright),
                    'mean_applied_residual_rms': mean(residuals),
                    'events_observed': observed_events,
                    'events_accepted': accepted_events,
                    'termination_reason': reason,
                    'reached_time_limit': (reason == 'episode_time_limit'),
                }
                episodes.append(row)
                print(f'  episode {ep:2d}: {t:5.3f} s  return {sum(rewards):7.2f}  '
                      f'site {mean(site_errors):.3f} m  speed {mean(speeds):.2f} m/s  {reason}', flush=True)
    finally:
        env.close()
    write_csv(args.output_dir / 'episodes.csv', episodes)
    aggregate = []
    for mode in args.modes:
        group = [x for x in episodes if x['mode'] == mode]
        times = [x['duration_s'] for x in group]
        aggregate.append({
            'mode': mode,
            'n': len(group),
            'mean_duration_s': mean(times),
            'sd_duration_s': stdev(times) if len(times) >= 2 else 0.0,
            'min_duration_s': min(times),
            'max_duration_s': max(times),
            'mean_total_reward': mean(x['total_reward'] for x in group),
            'mean_site_error_m': mean(x['mean_site_error_m'] for x in group),
            'mean_world_root_error_m': mean(x['mean_world_root_error_m'] for x in group),
            'mean_forward_speed_m_s': mean(x['mean_forward_speed_m_s'] for x in group),
            'mean_upright_dot': mean(x['mean_upright_dot'] for x in group),
            'mean_applied_residual_rms': mean(x['mean_applied_residual_rms'] for x in group),
            'time_limit_reached_count': sum(bool(x['reached_time_limit']) for x in group),
        })
    write_csv(args.output_dir / 'summary.csv', aggregate)
    provenance = {'checkpoint': str(args.checkpoint), 'summary': str(args.summary),
                  'modes': args.modes, 'episodes': args.episodes,
                  'event_sync': settings.event_sync,
                  'enable_tracking_termination': settings.enable_tracking_termination,
                  'deterministic_actions': True,
                  'note': 'Ablation masks outputs of one trained policy; this is not independent retraining.'}
    (args.output_dir / 'metadata.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('\nCOMPARISON (paired episode identities)')
    print(f'{"MODE":13} {"survival mean±SD (s)":>22} {"return":>9} {"site err (m)":>13} {"speed (m/s)":>13}')
    for row in aggregate:
        print(f'{row["mode"]:13} {row["mean_duration_s"]:7.3f} ± {row["sd_duration_s"]:5.3f} '
              f'{row["mean_total_reward"]:9.2f} {row["mean_site_error_m"]:13.3f} {row["mean_forward_speed_m_s"]:13.3f}')
    print(f'\nSaved {args.output_dir / "episodes.csv"} and {args.output_dir / "summary.csv"}')


if __name__ == '__main__':
    main()
