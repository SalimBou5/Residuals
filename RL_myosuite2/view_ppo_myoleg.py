#!/usr/bin/env python3
"""Watch genuine Step-6 MyoLeg MEP+PPO physics in MuJoCo's interactive viewer.

Works with the existing Step-6 environment, fixed normalizer, MEP, activation
lookup, gait reference library and SB3 checkpoint. No kinematic replay.

For an EXACT previously evaluated initialization, use --summary and --episode;
this passes that episode's recorded reference filepath and reset phase to RSI.

Optional --record FILE.mp4 records the same physical state from an independent
off-screen camera, while the native interactive viewer is also open.

New: optional live/offline diagnostic plots for
- muscle excitations: applied total control vs MEP baseline
- kinematics: actual nonroot joint positions vs reference joint positions
- phase: phase, theta, drift and phase-rate history
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import time

import numpy as np

from ppo_companion import EnvironmentSettings, create_environment, neutral_mep_action, validate_dynamic_step


PATH_ARGS = ('model', 'references', 'mep', 'lookup', 'normalization', 'reward_config')


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', type=Path, help='SB3 PPO final_model.zip or evaluation/best_model.zip')
    p.add_argument('--summary', type=Path, help='summary.json produced by evaluate_ppo.py; supplies all six input paths and exact evaluated reset states')
    for name in PATH_ARGS:
        p.add_argument('--' + name.replace('_', '-'), type=Path, help=f'Input {name} (unneeded when --summary is used)')
    p.add_argument('--mode', choices=('full', 'mep', 'muscle-only', 'phase-only'), default='full',
                   help='Mask outputs of the SAME trained policy; no retraining')
    p.add_argument('--episode', type=int, default=0,
                   help='Index into summary.json episodes, or seed offset when no summary is supplied')
    p.add_argument('--episodes', type=int, default=1, help='Number of consecutive episodes to watch')
    p.add_argument('--seed', type=int, default=17, help='Only used without a summary')
    p.add_argument('--seconds', type=float, default=None, help='Override max episode duration in seconds')
    p.add_argument('--speed', type=float, default=0.5, help='Playback speed relative to real time: 0.5 is half speed')
    p.add_argument('--hold-start', type=float, default=1.5, help='Seconds to inspect the stationary RSI reset state')
    p.add_argument('--hold-end', type=float, default=4.0, help='Seconds to inspect the final failure pose')
    p.add_argument('--stay-open', action='store_true', help='After last episode, keep viewer open until you close its window')
    p.add_argument('--stochastic', action='store_true', help='Sample the PPO action distribution instead of policy mean')
    p.add_argument('--no-follow', action='store_true', help='Leave native MuJoCo free camera instead of tracking pelvis')
    p.add_argument('--report-dir', type=Path, default=Path('runs/ppo_viewer'),
                   help='Save step-level CSV and episode JSON here')
    p.add_argument('--record', type=Path, default=None, help='Optionally record one episode to MP4 (requires imageio and imageio-ffmpeg)')
    p.add_argument('--video-fps', type=int, default=50)
    p.add_argument('--video-width', type=int, default=960)
    p.add_argument('--video-height', type=int, default=540)
    p.add_argument('--disable-event-sync', action='store_true', help='Only relevant without --summary; for testing')
    p.add_argument('--disable-tracking-termination', action='store_true', help='Only relevant without --summary; for testing')
    p.add_argument('--enable-unmapped', action='store_true', help='Only relevant without --summary (requires 505-D normalizer)')
    p.add_argument('--plot-mode', choices=('none', 'live', 'save', 'both'), default='save',
                   help='Visualization mode for excitations / kinematics / phase')
    p.add_argument('--plot-update-every', type=int, default=10,
                   help='Update live matplotlib plots every N control steps')
    p.add_argument('--plot-max-muscles-per-side', type=int, default=0,
                   help='Limit number of muscles plotted per side (0 = plot all side muscles)')
    p.add_argument('--plot-max-joints-per-side', type=int, default=0,
                   help='Limit number of joints plotted per side (0 = plot all side joints)')
    p.add_argument('--muscles-per-panel', type=int, default=3,
                   help='Maximum number of muscles on one saved subplot, split within anatomical groups')
    return p


def load_setup(args: argparse.Namespace) -> tuple[EnvironmentSettings, list[dict] | None]:
    if args.summary is not None:
        contents = json.loads(args.summary.read_text(encoding='utf-8'))
        if 'settings' not in contents or 'episodes' not in contents:
            raise ValueError('Expected evaluate_ppo.py summary.json with keys: settings and episodes')
        settings = EnvironmentSettings(**contents['settings'])
        previous_episodes = contents['episodes']
        if not isinstance(previous_episodes, list):
            raise ValueError('summary.json episodes must be a list')
        if any(getattr(args, name) is not None for name in PATH_ARGS):
            raise ValueError('Use --summary OR the six individual input paths, not both')
    else:
        missing = [name for name in PATH_ARGS if getattr(args, name) is None]
        if missing:
            raise ValueError('Provide --summary existing_evaluation/summary.json, OR all six input paths. Missing: ' + ', '.join(missing))
        settings = EnvironmentSettings(
            model=str(args.model.resolve()), references=str(args.references.resolve()),
            mep=str(args.mep.resolve()), lookup=str(args.lookup.resolve()),
            normalization=str(args.normalization.resolve()), reward_config=str(args.reward_config.resolve()),
            seed=args.seed, max_episode_seconds=4.0,
            event_sync=not args.disable_event_sync,
            enable_tracking_termination=not args.disable_tracking_termination,
            enable_unmapped=args.enable_unmapped,
        )
        previous_episodes = None
    if args.seconds is not None:
        settings = replace(settings, max_episode_seconds=args.seconds)
    settings.verify_paths()
    return settings, previous_episodes


def set_tracking_camera(mj, handle, model, data, *, enabled: bool) -> int:
    pelvis_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'pelvis'))
    if enabled and pelvis_id >= 0:
        with handle.lock():
            handle.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            handle.cam.trackbodyid = pelvis_id
            handle.cam.distance = 2.7
            handle.cam.elevation = -15.0
            handle.cam.azimuth = 145.0
            handle.cam.lookat[:] = data.xpos[pelvis_id]
    return pelvis_id


def init_video(args, mj, model, data, pelvis_id):
    if args.record is None:
        return None, None, None
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError('Recording requires: pip install imageio imageio-ffmpeg') from exc
    args.record.parent.mkdir(parents=True, exist_ok=True)
    try:
        renderer = mj.Renderer(model, height=args.video_height, width=args.video_width)
    except Exception as exc:
        raise RuntimeError('MuJoCo off-screen renderer could not initialize; run without --record to use the native viewer') from exc
    camera = mj.MjvCamera()
    if not args.no_follow and pelvis_id >= 0:
        camera.type = mj.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = pelvis_id
        camera.distance = 2.7
        camera.elevation = -15.0
        camera.azimuth = 145.0
        camera.lookat[:] = data.xpos[pelvis_id]
    else:
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.distance = 3.0
        camera.azimuth = 145.0
        camera.elevation = -15.0
        camera.lookat[:] = data.qpos[:3]
    try:
        writer = imageio.get_writer(str(args.record), fps=args.video_fps, codec='libx264', quality=8)
    except Exception:
        renderer.close()
        raise
    return renderer, writer, camera


def capture(renderer, writer, camera, data):
    renderer.update_scene(data, camera=camera)
    writer.append_data(renderer.render())


def wait_live(viewer, seconds: float) -> None:
    finish = time.monotonic() + seconds
    while viewer.is_running() and time.monotonic() < finish:
        viewer.sync()
        time.sleep(min(0.03, max(0.0, finish - time.monotonic())))


def evaluate_actions(model, obs, neutral, mode: str, *, stochastic: bool) -> np.ndarray:
    if mode == 'mep':
        return neutral.copy()
    action = np.asarray(model.predict(obs, deterministic=not stochastic)[0], dtype=np.float32).copy()
    if action.shape != neutral.shape:
        raise ValueError(f'Action shape {action.shape} does not match {neutral.shape}')
    np.clip(action, -1.0, 1.0, out=action)
    if mode == 'muscle-only':
        action[-1] = neutral[-1]
    elif mode == 'phase-only':
        action[:-1] = neutral[:-1]
    return action


def _safe_name(x: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(x))


def _slice_side(names: list[str], side: str) -> list[int]:
    tag = f'_{side}'
    alt = f'/{side}'
    idx = [i for i, name in enumerate(names) if name.endswith(tag) or tag in name or alt in name]
    return idx


def _maybe_limit(idxs: list[int], limit: int) -> list[int]:
    if limit and limit > 0:
        return idxs[:limit]
    return idxs


def _strip_side_suffix(name: str) -> str:
    for suf in ('_r', '_l', '/r', '/l'):
        if name.endswith(suf):
            return name[:-2]
    return name


def _muscle_group_name(raw_name: str) -> str:
    """Anatomical plotting families; multi-joint muscles appear only once.

    The family labels are for plot organization, not claims of unique function.
    Unknown names are retained in 'Other muscles'.
    """
    name = _strip_side_suffix(raw_name).lower().replace('-', '').replace('_', '')
    if name.startswith(('glmax', 'glutmax')):
        return 'Gluteus maximus'
    if name.startswith(('glmed', 'glmin', 'glutmed', 'glutmin')):
        return 'Gluteus medius / minimus'
    if name.startswith(('psoas', 'iliacus', 'iliopsoas', 'tfl', 'sart', 'pect')):
        return 'Hip flexor family'
    if name.startswith(('add', 'grac')):
        return 'Hip adductor family'
    if name.startswith(('piri', 'gem', 'obt', 'quadfem')):
        return 'Hip rotator family'
    if name.startswith(('vas', 'recfem', 'rectfem')):
        return 'Quadriceps'
    if name.startswith(('bflh', 'bfsh', 'semiten', 'semimem', 'bicepsfem')):
        return 'Hamstrings'
    if name.startswith(('gas', 'soleus', 'sol')):
        return 'Triceps surae'
    if name.startswith(('per', 'fib', 'tibpost', 'plant')):
        return 'Lateral / deep calf'
    if name.startswith(('tibant', 'tibialisant', 'edl', 'ehl')):
        return 'Anterior leg / toe extensors'
    if name.startswith(('fdl', 'fhl', 'flex')):
        return 'Toe flexors'
    if name.startswith(('ext',)):
        return 'Toe extensors'
    return 'Other muscles'


def _group_indices_by_name(names: list[str], idxs: list[int],
                           max_per_panel: int = 3) -> list[tuple[str, list[int]]]:
    """Preserve all muscles, split long families into panels of <= max_per_panel."""
    if max_per_panel < 1:
        raise ValueError('--muscles-per-panel must be at least 1')
    order = [
        'Gluteus maximus', 'Gluteus medius / minimus',
        'Hip flexor family', 'Hip adductor family', 'Hip rotator family',
        'Quadriceps', 'Hamstrings', 'Triceps surae', 'Lateral / deep calf',
        'Anterior leg / toe extensors', 'Toe flexors', 'Toe extensors',
        'Other muscles',
    ]
    buckets = {k: [] for k in order}
    for idx in idxs:
        buckets[_muscle_group_name(names[idx])].append(idx)
    grouped = []
    for family in order:
        ids = buckets[family]
        for k in range(0, len(ids), max_per_panel):
            block = ids[k:k + max_per_panel]
            label = family if len(ids) <= max_per_panel else f'{family} ({k // max_per_panel + 1})'
            grouped.append((label, block))
    return grouped


def _joint_group_name(raw_name: str) -> str:
    name = _strip_side_suffix(raw_name).lower()
    if 'hip' in name:
        return 'Hip'
    if 'knee' in name:
        return 'Knee'
    if 'ankle' in name or 'subtalar' in name:
        return 'Ankle'
    if 'mtp' in name or 'toe' in name:
        return 'Toes'
    return 'Other joints'


def _group_joints(names: list[str], idxs: list[int], max_per_panel: int = 3):
    families = ('Hip', 'Knee', 'Ankle', 'Toes', 'Other joints')
    grouped = []
    for family in families:
        ids = [i for i in idxs if _joint_group_name(names[i]) == family]
        for k in range(0, len(ids), max_per_panel):
            label = family if len(ids) <= max_per_panel else f'{family} ({k // max_per_panel + 1})'
            grouped.append((label, ids[k:k + max_per_panel]))
    return grouped


def actuator_names(env, mj):
    names = []
    for i in range(int(env.model.nu)):
        name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_ACTUATOR, i) or f'actuator_{i}'
        names.append(str(name))
    return names


def nonroot_joint_metadata(env, mj):
    names, qpos_ids = [], []
    for j in range(int(env.model.njnt)):
        qadr = int(env.model.jnt_qposadr[j])
        if qadr < 7:
            continue
        name = mj.mj_id2name(env.model, mj.mjtObj.mjOBJ_JOINT, j) or f'joint_{j}'
        names.append(str(name))
        qpos_ids.append(qadr - 7)
    return names, qpos_ids


class EpisodePlotter:
    def __init__(self, env, mj, obs_slices, prefix: Path, mode: str,
                 plot_mode: str = 'save', plot_update_every: int = 10,
                 max_muscles_per_side: int = 0, max_joints_per_side: int = 0,
                 muscles_per_panel: int = 3):
        self.env = env
        self.mj = mj
        self.obs_slices = obs_slices
        self.prefix = prefix
        self.plot_mode = plot_mode
        self.plot_live = plot_mode in ('live', 'both')
        self.plot_save = plot_mode in ('save', 'both')
        self.plot_update_every = max(1, int(plot_update_every))
        self.mode = mode
        self._mpl = None
        self._fig_exc = self._axs_exc = None
        self._fig_kin = self._axs_kin = None
        self._fig_phase = self._axs_phase = None
        self.times = []
        self.phase = []
        self.theta = []
        self.phase_drift = []
        self.phase_rate_hz = []
        self.ctrl = []
        self.baseline = []
        self.joint_actual = []
        self.joint_reference = []
        self.actuator_names = actuator_names(env, mj)
        self.joint_names, self.joint_qpos_ids = nonroot_joint_metadata(env, mj)
        self.right_muscles = _maybe_limit(_slice_side(self.actuator_names, 'r'), max_muscles_per_side)
        self.left_muscles = _maybe_limit(_slice_side(self.actuator_names, 'l'), max_muscles_per_side)
        self.right_muscle_groups = _group_indices_by_name(self.actuator_names, self.right_muscles, muscles_per_panel)
        self.left_muscle_groups = _group_indices_by_name(self.actuator_names, self.left_muscles, muscles_per_panel)
        self.right_joints = _maybe_limit(_slice_side(self.joint_names, 'r'), max_joints_per_side)
        self.left_joints = _maybe_limit(_slice_side(self.joint_names, 'l'), max_joints_per_side)
        self.right_joint_groups = _group_joints(self.joint_names, self.right_joints)
        self.left_joint_groups = _group_joints(self.joint_names, self.left_joints)
        self._init_live()

    def _init_live(self):
        if not self.plot_live:
            return
        import matplotlib.pyplot as plt
        self._mpl = plt
        plt.ion()
        self._fig_exc, self._axs_exc = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        self._fig_kin, self._axs_kin = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        self._fig_phase, self._axs_phase = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        self._fig_exc.suptitle(f'Excitations ({self.mode})')
        self._fig_kin.suptitle('Joint kinematics: actual vs reference')
        self._fig_phase.suptitle('Phase diagnostics')
        for fig in (self._fig_exc, self._fig_kin, self._fig_phase):
            fig.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def record(self, t: float, info: dict, obs: np.ndarray):
        self.times.append(float(t))
        self.phase.append(float(info.get('phase', np.nan)))
        self.theta.append(float(info.get('theta', np.nan)))
        self.phase_drift.append(float(info.get('phase_drift', np.nan)))
        self.phase_rate_hz.append(float(info.get('phase_rate_hz', np.nan)))
        self.ctrl.append(np.asarray(self.env.rsi.data.ctrl, dtype=float).copy())
        self.baseline.append(np.asarray(self.env.core.actions.last_baseline, dtype=float).copy())
        joint_actual = np.asarray(self.env.rsi.data.qpos[7:], dtype=float).copy()
        self.joint_actual.append(joint_actual)
        # The PPO obs is normalized; do NOT plot obs['reference_joints'] against raw qpos.
        # Obtain the physical-unit joint reference at the adapted phase directly.
        target = self.env.observations.reference.sample(
            self.env.core.clock.theta, phase_rate=self.env.core.clock.phase_rate_hz)
        self.joint_reference.append(np.asarray(target.qpos[7:], dtype=float).copy())
        if self.plot_live and (len(self.times) == 1 or len(self.times) % self.plot_update_every == 0):
            self.update_live()

    def _draw_excitation_axis(self, ax, side_idxs: list[int], title: str):
        ax.clear()
        ax.set_title(title)
        if not side_idxs or not self.times:
            ax.text(0.5, 0.5, 'No data for this side', ha='center', va='center', transform=ax.transAxes)
            return
        ctrl = np.vstack(self.ctrl)
        baseline = np.vstack(self.baseline)
        x = np.asarray(self.times)
        from matplotlib import pyplot as plt
        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for j, idx in enumerate(side_idxs):
            color = palette[j % len(palette)]
            ax.plot(x, baseline[:, idx], color=color, alpha=0.32, linestyle='--', linewidth=1.2)
            ax.plot(x, ctrl[:, idx], color=color, linewidth=1.7, label=self.actuator_names[idx])
        ax.set_ylabel('excitation')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        if len(side_idxs) <= 14:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)

    def _draw_kin_axis(self, ax, joint_idxs: list[int], title: str):
        ax.clear()
        ax.set_title(title)
        if not joint_idxs or not self.times:
            ax.text(0.5, 0.5, 'No data for this side', ha='center', va='center', transform=ax.transAxes)
            return
        actual = np.vstack(self.joint_actual)
        ref = np.vstack(self.joint_reference)
        x = np.asarray(self.times)
        from matplotlib import pyplot as plt
        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for j, local_idx in enumerate(joint_idxs):
            qpos_idx = self.joint_qpos_ids[local_idx]
            name = self.joint_names[local_idx]
            color = palette[j % len(palette)]
            ax.plot(x, ref[:, qpos_idx], color=color, alpha=0.32, linestyle='--', linewidth=1.2)
            ax.plot(x, actual[:, qpos_idx], color=color, linewidth=1.7, label=name)
        ax.set_ylabel('joint position')
        ax.grid(True, alpha=0.25)
        if len(joint_idxs) <= 14:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)

    def _draw_phase_axes(self, axs):
        x = np.asarray(self.times)
        axs[0].clear(); axs[1].clear(); axs[2].clear()
        axs[0].plot(x, self.phase, label='phase')
        theta_mod = np.mod(self.theta, 1.0)
        axs[0].plot(x, theta_mod, alpha=0.6, linestyle='--', label='theta mod 1')
        axs[0].set_ylabel('phase')
        axs[0].set_ylim(-0.05, 1.05)
        axs[0].grid(True, alpha=0.25)
        axs[0].legend(loc='best')
        axs[1].plot(x, self.phase_drift)
        axs[1].set_ylabel('drift')
        axs[1].grid(True, alpha=0.25)
        axs[2].plot(x, self.phase_rate_hz)
        axs[2].set_ylabel('phase rate (Hz)')
        axs[2].set_xlabel('time (s)')
        axs[2].grid(True, alpha=0.25)

    def _draw_excitation_group_axis(self, ax, idxs: list[int], title: str):
        ax.clear()
        ax.set_title(title)
        if not idxs or not self.times:
            ax.text(0.5, 0.5, 'No data in this group', ha='center', va='center', transform=ax.transAxes)
            return
        ctrl = np.vstack(self.ctrl)
        baseline = np.vstack(self.baseline)
        x = np.asarray(self.times)
        from matplotlib import pyplot as plt
        palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for j, idx in enumerate(idxs):
            color = palette[j % len(palette)]
            ax.plot(x, baseline[:, idx], color=color, alpha=0.32, linestyle='--', linewidth=1.2)
            ax.plot(x, ctrl[:, idx], color=color, linewidth=1.8, label=self.actuator_names[idx])
        ax.set_ylabel('excitation')
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=7, ncol=1)

    def _save_excitation_side_figure(self, side_label: str, grouped: list[tuple[str, list[int]]], prefix_suffix: str):
        import matplotlib.pyplot as plt
        if not grouped:
            return None
        n = len(grouped)
        fig, axs = plt.subplots(n, 1, figsize=(15, max(3.2 * n, 5.0)), sharex=True)
        if n == 1:
            axs = [axs]
        for ax, (group_name, idxs) in zip(axs, grouped):
            self._draw_excitation_group_axis(ax, idxs, f'{side_label} — {group_name}')
        axs[-1].set_xlabel('time (s)')
        fig.suptitle(f'{side_label} muscle excitations: applied (solid) vs MEP baseline (faded dashed)', y=0.999)
        fig.tight_layout(rect=(0, 0, 1, 0.981))
        out = self.prefix.with_name(self.prefix.name + prefix_suffix)
        fig.savefig(out, dpi=180, bbox_inches='tight')
        plt.close(fig)
        return out

    def _save_kinematics_side_figure(self, side_label: str, grouped, prefix_suffix: str):
        import matplotlib.pyplot as plt
        if not grouped or not self.times:
            return None
        n = len(grouped)
        fig, axs = plt.subplots(n, 1, figsize=(15, max(3.0 * n, 5.0)), sharex=True)
        if n == 1:
            axs = [axs]
        for ax, (name, idxs) in zip(axs, grouped):
            self._draw_kin_axis(ax, idxs, f'{side_label} — {name}')
        axs[-1].set_xlabel('time (s)')
        fig.suptitle(f'{side_label} joint kinematics: actual (solid) vs reference (faded dashed)', y=0.999)
        fig.tight_layout(rect=(0, 0, 1, 0.981))
        out = self.prefix.with_name(self.prefix.name + prefix_suffix)
        fig.savefig(out, dpi=180, bbox_inches='tight')
        plt.close(fig)
        return out

    def update_live(self):
        if not self.plot_live:
            return
        self._draw_excitation_axis(self._axs_exc[0], self.right_muscles, 'Right muscles: applied total (solid) vs baseline MEP (faded)')
        self._draw_excitation_axis(self._axs_exc[1], self.left_muscles, 'Left muscles: applied total (solid) vs baseline MEP (faded)')
        self._axs_exc[1].set_xlabel('time (s)')
        self._draw_kin_axis(self._axs_kin[0], self.right_joints, 'Right joints: actual (solid) vs reference (faded)')
        self._draw_kin_axis(self._axs_kin[1], self.left_joints, 'Left joints: actual (solid) vs reference (faded)')
        self._axs_kin[1].set_xlabel('time (s)')
        self._draw_phase_axes(self._axs_phase)
        for fig in (self._fig_exc, self._fig_kin, self._fig_phase):
            fig.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def save(self):
        if not self.plot_save or not self.times:
            return []
        import matplotlib.pyplot as plt
        saved = []

        p = self._save_excitation_side_figure('Right side', self.right_muscle_groups, '_excitations_right.png')
        if p is not None:
            saved.append(p)
        p = self._save_excitation_side_figure('Left side', self.left_muscle_groups, '_excitations_left.png')
        if p is not None:
            saved.append(p)

        p = self._save_kinematics_side_figure('Right side', self.right_joint_groups, '_kinematics_right.png')
        if p is not None:
            saved.append(p)
        p = self._save_kinematics_side_figure('Left side', self.left_joint_groups, '_kinematics_left.png')
        if p is not None:
            saved.append(p)

        fig3, axs3 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        self._draw_phase_axes(axs3)
        fig3.suptitle('Phase diagnostics')
        fig3.tight_layout()
        p3 = self.prefix.with_name(self.prefix.name + '_phase.png')
        fig3.savefig(p3, dpi=180, bbox_inches='tight')
        plt.close(fig3)
        saved.append(p3)
        return saved

    def close(self):
        if self._mpl is None:
            return
        self._mpl.ioff()


def run(args: argparse.Namespace) -> None:
    if args.episode < 0 or args.episodes < 1 or args.speed <= 0 or args.video_fps < 1:
        raise ValueError('episode must be >= 0; episodes, speed and video-fps must be positive')
    if args.hold_start < 0 or args.hold_end < 0:
        raise ValueError('Hold durations must be nonnegative')
    if args.record is not None and args.episodes != 1:
        raise ValueError('--record currently supports one episode at a time (use --episodes 1)')
    if args.record is not None and args.record.suffix.lower() != '.mp4':
        raise ValueError('--record filename must end in .mp4')
    if args.muscles_per_panel < 1:
        raise ValueError('--muscles-per-panel must be at least 1')
    settings, old_episodes = load_setup(args)
    if old_episodes is not None and args.episode + args.episodes > len(old_episodes):
        raise ValueError(f'Summary has {len(old_episodes)} episodes. Requested episode {args.episode} + {args.episodes} episodes.')
    if args.checkpoint is None and args.mode != 'mep':
        raise ValueError('--checkpoint is required for modes other than mep')

    try:
        import mujoco as mj
        import mujoco.viewer
    except ImportError as exc:
        raise SystemExit('Install MuJoCo in the myosuite environment: pip install mujoco') from exc
    model = None
    if args.mode != 'mep':
        from stable_baselines3 import PPO
        model = PPO.load(str(args.checkpoint), device='cpu')
    env = create_environment(settings)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    try:
        expected_act = env.action_space.shape
        expected_obs = env.observation_space.shape
        if model is not None and (model.action_space.shape != expected_act or model.observation_space.shape != expected_obs):
            raise ValueError(f'Checkpoint has action/obs {model.action_space.shape}/{model.observation_space.shape}, '
                             f'but environment expects {expected_act}/{expected_obs}.')
        dt = env.core.config.control_dt_s
        print(f'Viewer: actual MyoLeg MuJoCo dynamics, {dt * 1000:.1f} ms PPO action period, '
              f'{env.core.config.physics_dt_s * 1000:.1f} ms physics period')
        print(f'Mode: {args.mode}; event sync: {settings.event_sync}; tracking termination: {settings.enable_tracking_termination}')
        print('The reference qpos/qvel are applied ONLY at reset. Thereafter env.step() integrates muscle-driven physics.')
        print('Plot mode:', args.plot_mode)

        with mj.viewer.launch_passive(env.model, env.rsi.data) as viewer:
            for ep in range(args.episode, args.episode + args.episodes):
                if not viewer.is_running():
                    break
                if old_episodes is not None:
                    prior = old_episodes[ep]
                    options = {'reference': prior['reference'], 'phase': float(prior['reset_phase'])}
                else:
                    options = None
                obs, reset_info = env.reset(seed=settings.seed + ep, options=options)
                start_phase = float(reset_info['reset_phase'])
                ref_path = str(reset_info['reference'])
                if options is not None:
                    if not np.isclose(start_phase, options['phase'], rtol=0, atol=1e-9):
                        raise AssertionError('Reset phase does not match recorded episode')
                    if Path(ref_path).resolve() != Path(options['reference']).resolve():
                        raise AssertionError('Reset reference does not match recorded episode')
                neutral = neutral_mep_action(env)
                pelvis_id = set_tracking_camera(mj, viewer, env.model, env.rsi.data, enabled=not args.no_follow)
                viewer.sync()
                print(f'\nEpisode {ep}: reference={Path(ref_path).name}; reset_phase={start_phase:.6f}; initial floor penetration={reset_info["max_reset_penetration_m"]*1000:.1f} mm')
                wait_live(viewer, args.hold_start)

                writer = renderer = camera = None
                rows: list[dict] = []
                total_reward = 0.0
                last_info = None
                next_video_time = 0.0
                prefix = args.report_dir / f'episode_{ep:02d}_{args.mode.replace("-", "_")}'
                plotter = EpisodePlotter(env, mj, env.observation_slices, prefix,
                                         args.mode, args.plot_mode, args.plot_update_every,
                                         args.plot_max_muscles_per_side, args.plot_max_joints_per_side,
                                         args.muscles_per_panel)
                try:
                    renderer, writer, camera = init_video(args, mj, env.model, env.rsi.data, pelvis_id)
                    if writer is not None:
                        capture(renderer, writer, camera, env.rsi.data)
                        next_video_time = 1.0 / args.video_fps
                    wall_start = time.monotonic()
                    next_console_time = 0.0
                    while viewer.is_running():
                        action = evaluate_actions(model, obs, neutral, args.mode, stochastic=args.stochastic)
                        obs, reward, terminated, truncated, info = env.step(action)
                        validate_dynamic_step(env, obs, reward, info, baseline_only=args.mode == 'mep')
                        last_info = info
                        total_reward += float(reward)
                        terms = info['reward_terms']
                        t = float(info['time_s'])
                        plotter.record(t, info, obs)
                        row = {
                            'episode': ep, 'time_s': t, 'phase': float(info['phase']),
                            'theta': float(info.get('theta', np.nan)),
                            'phase_drift': float(info.get('phase_drift', np.nan)),
                            'phase_rate_hz': float(info['phase_rate_hz']),
                            'reward': float(reward), 'return_so_far': total_reward,
                            'site_error_m': float(terms['mean_site_position_error_m']),
                            'root_error_m': float(terms['world_root_position_error_m']),
                            'actual_speed_m_s': float(terms['actual_forward_speed_m_s']),
                            'desired_speed_m_s': float(terms['desired_forward_speed_m_s']),
                            'upright_dot': float(terms['upright_dot']),
                            'residual_rms': float(terms['actual_applied_residual_rms']),
                            'events': '|'.join(('+' if e['accepted'] else '-') + str(e['event']) for e in info.get('events', ())),
                            'done': bool(terminated or truncated),
                            'reason': info.get('termination_reason') or info.get('truncation_reason') or '',
                        }
                        rows.append(row)
                        if t >= next_console_time or terminated or truncated:
                            print(f't={t:5.3f}s  site={row["site_error_m"]:5.3f}m  '
                                  f'speed={row["actual_speed_m_s"]:5.2f}/{row["desired_speed_m_s"]:5.2f}m/s  '
                                  f'upright={row["upright_dot"]:5.2f}  phase={row["phase"]:5.2f}  '
                                  f'events={row["events"] or "-"}')
                            next_console_time += 0.2
                        if writer is not None:
                            while t + 1e-10 >= next_video_time:
                                capture(renderer, writer, camera, env.rsi.data)
                                next_video_time += 1.0 / args.video_fps
                        viewer.sync()
                        if terminated or truncated:
                            print(f'END: {row["reason"]}, duration={t:.3f}s, steps={len(rows)}, return={total_reward:.2f}')
                            break
                        target_wall = wall_start + t / args.speed
                        ahead = target_wall - time.monotonic()
                        if ahead > 0:
                            time.sleep(ahead)
                finally:
                    if writer is not None:
                        writer.close()
                    if renderer is not None:
                        renderer.close()

                if rows:
                    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
                        out = csv.DictWriter(f, fieldnames=list(rows[0]))
                        out.writeheader()
                        out.writerows(rows)
                outcome = {
                    'episode': ep, 'mode': args.mode, 'checkpoint': str(args.checkpoint),
                    'reference': ref_path, 'reset_phase': start_phase,
                    'seconds': rows[-1]['time_s'] if rows else 0.0,
                    'steps': len(rows), 'return': total_reward,
                    'reason': rows[-1]['reason'] if rows else 'viewer_closed',
                    'event_sync': settings.event_sync,
                    'tracking_termination': settings.enable_tracking_termination,
                    'physics_integrated': True,
                    'reset_only_kinematics': True,
                }
                prefix.with_suffix('.json').write_text(json.dumps(outcome, indent=2) + '\n', encoding='utf-8')
                saved_plots = plotter.save()
                plotter.close()
                print(f'Saved: {prefix.with_suffix(".csv")}, {prefix.with_suffix(".json")}')
                for p in saved_plots:
                    print(f'Saved plot: {p}')
                if args.record is not None:
                    print(f'Saved MP4: {args.record.resolve()}')
                if viewer.is_running():
                    wait_live(viewer, args.hold_end)
            if args.stay_open and viewer.is_running():
                print('Final pose will remain in the viewer. Close the viewer window to exit.')
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.03)
    finally:
        env.close()


def main() -> None:
    try:
        run(make_parser().parse_args())
    except KeyboardInterrupt:
        print('\nViewer interrupted by user.')


if __name__ == '__main__':
    main()
