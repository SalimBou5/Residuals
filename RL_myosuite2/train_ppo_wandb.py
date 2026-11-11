#!/usr/bin/env python3
"""Step 6: CPU-parallel SB3 PPO with native online Weights & Biases logging.

Windows safe: imports and PPO workers are spawned ONLY within main().
All MuJoCo models and reference caches are created inside each worker.
No TensorBoard dependency, no running VecNormalize, no changes to reward.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ppo_companion  import EnvironmentSettings, TrainingDiagnostics, create_environment
from smoke_test_mep import run_dynamic_episodes

import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ('model','references','mep','lookup','normalization','reward-config'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--run-dir', type=Path, default=Path('runs/ppo_myoleg'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-envs', type=int, default=8)
    p.add_argument('--total-steps', type=int, default=2_000_000)
    p.add_argument('--n-steps', type=int, default=1024, help='Transitions per worker per PPO rollout')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--learning-rate', type=float, default=2e-5)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-range', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.0)
    p.add_argument('--initial-action-std', type=float, default=0.10,
                   help='Initial unsquashed Gaussian std; controls are still bounded by Gym Box')
    p.add_argument('--max-seconds', type=float, default=10.0)
    p.add_argument('--cache-size', type=int, default=16)
    p.add_argument('--checkpoint-every', type=int, default=100_000,
                   help='Total environment transitions; divided by n-envs for callbacks')
    p.add_argument('--eval-every', type=int, default=100_000,
                   help='Total environment transitions; 0 disables evaluation')
    p.add_argument('--eval-episodes', type=int, default=8)
    p.add_argument('--preflight-episodes', type=int, default=3,
                   help='MEP-only dynamic episodes before starting PPO; 0 explicitly skips')
    p.add_argument('--preflight-seconds', type=float, default=1.5)
    p.add_argument('--device', default='auto', choices=('auto','cpu','cuda'))
    p.add_argument('--resume', type=Path, default=None, help='Load a previous SB3 PPO .zip checkpoint')
    p.add_argument('--wandb-project', default='myoleg-residual-gait')
    p.add_argument('--wandb-entity', default=None)
    p.add_argument('--wandb-name', default=None)
    p.add_argument('--wandb-mode', choices=('online','offline','disabled'), default='online')
    p.add_argument('--upload-final-model', action='store_true',
                   help='Upload final/best SB3 zip to W&B as model artifacts (can be large)')
    p.add_argument('--enable-unmapped', action='store_true',
                   help='81-D direct-unmapped actions; needs 505-D fitted normalizer')
    p.add_argument('--disable-event-sync', action='store_true')
    p.add_argument('--disable-tracking-termination', action='store_true',
                   help='Debug only: retain physical terminations')
    return p


def validate_train_args(args: argparse.Namespace) -> None:
    if args.n_envs < 1 or args.total_steps < 1 or args.n_steps < 2:
        raise ValueError('n-envs, total-steps and n-steps must be positive')
    if args.batch_size < 2 or args.epochs < 1 or args.eval_episodes < 1:
        raise ValueError('Invalid PPO minibatch, epochs or evaluation count')
    if (args.n_envs * args.n_steps) % args.batch_size != 0:
        raise ValueError('n-envs * n-steps must be divisible by batch-size')
    for key in ('learning_rate','clip_range','initial_action_std'):
        x = float(getattr(args, key))
        if not math.isfinite(x) or x <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('gamma','gae_lambda'):
        x = float(getattr(args, key))
        if not 0 < x <= 1:
            raise ValueError(f'{key} must be in (0,1]')
    if args.ent_coef < 0 or not math.isfinite(args.ent_coef):
        raise ValueError('ent-coef must be finite and nonnegative')
    if args.checkpoint_every < 0 or args.eval_every < 0 or args.preflight_episodes < 0:
        raise ValueError('Checkpoint/eval/preflight counts cannot be negative')
    if args.preflight_episodes and args.preflight_seconds <= 0:
        raise ValueError('preflight-seconds must be positive')


def build_settings(args: argparse.Namespace) -> EnvironmentSettings:
    return EnvironmentSettings(
        model=str(args.model.resolve()), references=str(args.references.resolve()),
        mep=str(args.mep.resolve()), lookup=str(args.lookup.resolve()),
        normalization=str(args.normalization.resolve()), reward_config=str(args.reward_config.resolve()),
        seed=args.seed, max_episode_seconds=args.max_seconds,
        reference_cache_size=args.cache_size,
        event_sync=not args.disable_event_sync,
        enable_unmapped=args.enable_unmapped,
        enable_tracking_termination=not args.disable_tracking_termination,
    )


def _make_worker(settings: EnvironmentSettings, rank: int):
    """Module-level function, to keep spawn/cloudpickle behavior predictable."""
    # Prevent 8 x implicit OpenMP thread oversubscription in MuJoCo workers.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    return create_environment(settings, worker_index=rank)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = build_parser().parse_args()
    validate_train_args(args)
    settings = build_settings(args)
    settings.verify_paths()

    # Parent-only imports: no W&B run or Torch CUDA context in spawned workers.
    try:
        import torch
        import wandb
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
        from stable_baselines3.common.logger import KVWriter, configure
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
        import gymnasium as gym
    except ImportError as exc:
        raise SystemExit('Missing Step-6 dependency. Install: pip install -r requirements_step6.txt') from exc

    if args.device == 'cpu':
        torch.set_num_threads(1)
    available_cores = os.cpu_count()
    if available_cores is not None and args.n_envs > available_cores:
        print('WARNING: More PPO workers than available logical CPU cores.')
    # Fixed seed, small initial motor noise: zero action means the MEP baseline.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.run_dir.resolve() / (args.wandb_name or f'ppo_{stamp}_seed{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir/'checkpoints').mkdir()
    (run_dir/'evaluation').mkdir()
    local_manifest = {
        'settings': asdict(settings),
        'train_args': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'normalization_sha256': _sha256(settings.normalization),
        'reward_config_sha256': _sha256(settings.reward_config),
        'note': 'Model / reference paths stored locally only, never in W&B hyperparameters',
    }
    (run_dir/'run_manifest.json').write_text(json.dumps(local_manifest,indent=2)+'\n',encoding='utf-8')

    if args.preflight_episodes:
        print(f'[Step 6] Real MuJoCo MEP-only preflight: {args.preflight_episodes} random phase episodes')
        preflight = run_dynamic_episodes(
            replace(settings, max_episode_seconds=args.preflight_seconds),
            episodes=args.preflight_episodes, output=run_dir/'mep_preflight')
        print('[Step 6] Preflight:', json.dumps(preflight['summary'],indent=2))
        print('[Step 6] A fall is not an infrastructure failure. Review the preflight CSV.')
    else:
        print('WARNING: Dynamic preflight explicitly skipped. No MEP-only baseline was simulated.')

    # No absolute local paths in W&B run config.
    public_config: dict[str, Any] = {
        'seed': args.seed, 'n_envs': args.n_envs,
        'total_steps_requested': args.total_steps,
        'n_steps': args.n_steps, 'batch_size': args.batch_size,
        'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'gamma': args.gamma, 'gae_lambda': args.gae_lambda,
        'clip_range': args.clip_range, 'ent_coef': args.ent_coef,
        'initial_action_std': args.initial_action_std,
        'episode_limit_s': args.max_seconds, 'physics_hz': 1000, 'policy_hz': 200,
        'reference_library_cycles': (sum(1 for _ in Path(settings.references).rglob('*_loopable.npz'))
                                     if Path(settings.references).is_dir() else 1),
        'reference_cache_size_per_worker': args.cache_size,
        'normalization_hash': local_manifest['normalization_sha256'],
        'reward_hash': local_manifest['reward_config_sha256'],
        'event_sync': settings.event_sync,
        'enable_unmapped': settings.enable_unmapped,
        'enable_tracking_termination': settings.enable_tracking_termination,
    }

    class WandbOutput(KVWriter):
        """Send SB3 native learning metrics without TensorBoard integration."""
        def __init__(self, run):
            self.run = run
        def write(self, key_values, key_excluded, step=0):
            payload = {'global_step': int(step)}
            for k,v in key_values.items():
                exclude = key_excluded.get(k, ()) or ()
                if 'wandb' in exclude:
                    continue
                if isinstance(v, (int,float,np.integer,np.floating)) and math.isfinite(float(v)):
                    payload['sb3/' + str(k)] = float(v)
            if len(payload) > 1:
                self.run.log(payload)
        def close(self):
            pass

    class WandbDiagnosticsCallback(BaseCallback):
        def __init__(self, run, n_envs, observation_dim):
            super().__init__(verbose=0)
            self.run = run
            self.aggregator = TrainingDiagnostics(n_envs,observation_dim)
        def _on_step(self):
            infos = self.locals.get('infos', [])
            rewards = self.locals.get('rewards', [])
            dones = self.locals.get('dones', [])
            actions = self.locals.get('actions', None)
            for idx,(info,r,done) in enumerate(zip(infos,rewards,dones)):
                action = None if actions is None else actions[idx]
                self.aggregator.observe(idx, float(r), info, bool(done), action)
            return True
        def _on_rollout_end(self):
            payload = self.aggregator.snapshot(reset=True)
            payload['global_step'] = int(self.num_timesteps)
            self.run.log(payload)

    class WandbEvalCallback(EvalCallback):
        def __init__(self, run, *a, **kw):
            self.wandb_run = run
            super().__init__(*a, **kw)
        def _on_step(self):
            keep_going = super()._on_step()
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.wandb_run.log({
                    'global_step': int(self.num_timesteps),
                    'eval/mean_episode_return': float(self.last_mean_reward),
                    'eval/best_mean_episode_return': float(self.best_mean_reward),
                })
            return keep_going

    class FixedEvaluationCases(gym.Wrapper):
        """Use the same valid (reference, RSI phase) cases at each evaluation."""
        def __init__(self, env, n_cases, seed):
            super().__init__(env)
            self.cases = []
            for i in range(n_cases):
                _, info = env.reset(seed=seed + i)
                self.cases.append((info['reference'], info['reset_phase']))
            self.index = 0
        def reset(self, *, seed=None, options=None):
            path, phase = self.cases[self.index % len(self.cases)]
            self.index += 1
            return self.env.reset(seed=seed, options={'reference':path,'phase':phase})

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        name=args.wandb_name or run_dir.name, mode=args.wandb_mode,
        dir=str(run_dir), config=public_config, save_code=False,
    )
    venv = eval_env = None
    try:
        run.define_metric('global_step')
        for namespace in ('reward/*','reward_terms/*','state/*','action/*',
                          'episodes/*','events/*','references/*','rollout/*','normalization/*',
                          'eval/*','sb3/*'):
            run.define_metric(namespace, step_metric='global_step')
        if args.preflight_episodes:
            run.log({'global_step': 0, 'preflight/mean_episode_seconds':preflight['summary']['mean_episode_seconds'],
                     'preflight/fall_count':preflight['summary']['terminated'],
                     'preflight/time_limit_count':preflight['summary']['truncated']})

        # On Windows use spawn (not fork). NEVER create the model in the parent.
        env_funcs = [partial(_make_worker, settings, rank) for rank in range(args.n_envs)]
        venv = (SubprocVecEnv(env_funcs, start_method='spawn') if args.n_envs > 1
                else DummyVecEnv(env_funcs))
        venv = VecMonitor(venv, filename=str(run_dir/'episode_monitor.csv'))
        obs_dim = int(venv.observation_space.shape[0])
        action_dim = int(venv.action_space.shape[0])
        if obs_dim != (505 if args.enable_unmapped else 485):
            raise ValueError(f'Unexpected observation dimension {obs_dim}')
        if action_dim != (81 if args.enable_unmapped else 71):
            raise ValueError(f'Unexpected action dimension {action_dim}')
        print(f'[Step 6] Observation={obs_dim}, action={action_dim}, workers={args.n_envs}')
        run.log({'global_step':0, 'model/obs_dim':obs_dim, 'model/action_dim':action_dim})

        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 256, 128],
                vf=[512, 256, 128],
            ),
            activation_fn=torch.nn.Tanh,
            log_std_init=float(np.log(args.initial_action_std)),
        )
        if args.resume:
            model = PPO.load(str(args.resume), env=venv, device=args.device)
            print('[Step 6] Resumed PPO checkpoint:', args.resume)
            print('[Step 6] Prior optimizer and schedule are restored; new CLI optimizer values are not reapplied.')
        else:
            model = PPO(
                'MlpPolicy', venv, policy_kwargs=policy_kwargs,
                n_steps=args.n_steps, batch_size=args.batch_size,
                n_epochs=args.epochs, learning_rate=args.learning_rate,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip_range=args.clip_range, ent_coef=args.ent_coef,
                target_kl=0.025, verbose=1, seed=args.seed, device=args.device,
                vf_coef=0.5, max_grad_norm=0.5
            )
        logger = configure(str(run_dir/'sb3_logs'), ['stdout','csv'])
        logger.output_formats.append(WandbOutput(run))
        model.set_logger(logger)
        callbacks = [WandbDiagnosticsCallback(run,args.n_envs,obs_dim)]
        if args.checkpoint_every:
            callbacks.append(CheckpointCallback(
                save_freq=max(1,math.ceil(args.checkpoint_every/args.n_envs)),
                save_path=str(run_dir/'checkpoints'),
                name_prefix='myoleg_ppo',
            ))
        if args.eval_every:
            fixed_env = FixedEvaluationCases(
                create_environment(replace(settings, seed=args.seed+50_000),worker_index=0),
                n_cases=args.eval_episodes,seed=args.seed+50_000)
            eval_env = Monitor(fixed_env)
            callbacks.append(WandbEvalCallback(
                run, eval_env,
                eval_freq=max(1,math.ceil(args.eval_every/args.n_envs)),
                n_eval_episodes=args.eval_episodes,deterministic=True,
                best_model_save_path=str(run_dir/'evaluation'),
                log_path=str(run_dir/'evaluation'),
                verbose=1,
            ))
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=args.resume is None,
            log_interval=1,
        )
        model.save(str(run_dir/'final_model'))
        run.summary['final_timesteps'] = int(model.num_timesteps)
        print('[Step 6] Final PPO checkpoint:', run_dir/'final_model.zip')
        if args.upload_final_model and args.wandb_mode != 'disabled':
            for label,path in (
                ('final',run_dir/'final_model.zip'),
                ('best',run_dir/'evaluation'/'best_model.zip'),
            ):
                if path.is_file():
                    artifact = wandb.Artifact(f'{run.id}-myoleg-ppo-{label}',type='model')
                    artifact.add_file(str(path))
                    run.log_artifact(artifact)
    finally:
        if eval_env is not None:
            eval_env.close()
        if venv is not None:
            venv.close()
        run.finish()


if __name__ == '__main__':
    main()
