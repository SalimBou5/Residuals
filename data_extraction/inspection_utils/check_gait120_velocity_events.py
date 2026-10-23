"""Compare rapid Gait120 ankle/subtalar motions with recorded ground forces.

This is a read-only diagnostic; it does not filter or modify source data.
Use MOT (joint angles), MOT (ground forces), and TRC from the SAME trial.
Only force-channel labels (not left/right feet) are assigned automatically.

python check_gait120_velocity_events.py --joint-mot "C:\Users\BoussofaraS\Downloads\Gait120_011_to_020\S015\JointAngle\LevelWalking\Trial01\step01.mot" --grf-mot "C:\Users\BoussofaraS\Downloads\Gait120_011_to_020\S015\MotionCapture\LevelWalking\MOT\Trial01\step01.mot" --trc "C:\Users\BoussofaraS\Downloads\Gait120_011_to_020\S015\MotionCapture\LevelWalking\TRC\Trial01\step01.trc"
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_mot(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    end = next((i for i, s in enumerate(lines) if s.strip().lower() == 'endheader'), None)
    if end is None:
        raise ValueError(f'OpenSim endheader missing: {path}')
    return pd.read_csv(path, sep=r'\s+', skiprows=end + 1)


def read_trc(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    marker_names = [s.strip() for s in lines[3].split('\t')[2:] if s.strip()]
    keys = lines[1].split('\t')
    vals = lines[2].split('\t')
    units = dict(zip(keys, vals)).get('Units', '').strip()
    scale = {'mm': 1e-3, 'm': 1}.get(units)
    if scale is None:
        raise ValueError(f'Unexpected TRC units: {units}')
    times, frames = [], []
    for line in lines[5:]:
        if not line.strip():
            continue
        fields = line.split('\t')
        times.append(float(fields[1]))
        frames.append([float(x) if x.strip() else np.nan for x in fields[2:2+3*len(marker_names)]])
    xyz = np.asarray(frames).reshape(-1, len(marker_names), 3) * scale
    return np.asarray(times), marker_names, xyz


def get_crossings(t, f, threshold):
    active = f >= threshold
    changes = np.flatnonzero(active[1:] != active[:-1]) + 1
    crossings = []
    for k in changes:
        # Linear interpolation of the 20 N crossing between recorded force samples.
        before, after = f[k-1], f[k]
        fraction = (threshold-before)/(after-before) if after != before else 1
        tc = t[k-1] + np.clip(fraction, 0, 1) * (t[k]-t[k-1])
        crossings.append((float(tc), 'on' if active[k] else 'off'))
    return crossings, bool(active[0]), bool(active[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--joint-mot', required=True)
    ap.add_argument('--grf-mot', required=True)
    ap.add_argument('--trc', required=True)
    ap.add_argument('--output', default='gait120_velocity_events.png')
    args = ap.parse_args()

    joints, grf = read_mot(args.joint_mot), read_mot(args.grf_mot)
    t = joints['time'].to_numpy(float)
    tf = grf['time'].to_numpy(float)
    tm, names, xyz = read_trc(args.trc)
    if not (np.all(np.diff(t) > 0) and np.all(np.diff(tf) > 0)):
        raise ValueError('Non-increasing timestamps')
    if len(tm) != len(t) or not np.allclose(tm, t, atol=1e-5):
        raise ValueError('Joint-MOT/TRC timestamps do not match')
    if t[0] < tf[0] - 0.002 or t[-1] > tf[-1] + 0.002:
        raise ValueError('GRF file does not cover the same trial interval')

    event_times = []
    for channel in ('ground_force1_vy', 'ground_force2_vy'):
        if channel not in grf:
            raise ValueError(f'Missing vertical force column {channel}')
        events, active_start, active_end = get_crossings(tf, grf[channel].to_numpy(float), 20)
        relevant = [(when, kind) for when, kind in events if t[0] <= when <= t[-1]]
        event_times += [(when, channel, kind) for when, kind in relevant]
        print(f'{channel}: starts above 20N = {active_start}; ends above 20N = {active_end}')
        for when, kind in relevant:
            print(f'  {kind:>3s} at {when:.4f}s (stride {when-t[0]:.4f}s)')

    watch = ('ankle_angle_l', 'ankle_angle_r', 'subtalar_angle_r')
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for channel in ('ground_force1_vy', 'ground_force2_vy'):
        axes[0].plot(tf-t[0], grf[channel].to_numpy(float), label=channel)
    axes[0].axhline(20, color='grey', ls='--', alpha=.7, label='20 N')
    axes[0].set_ylabel('Vertical GRF (N)')
    axes[0].legend(fontsize=8)

    for joint in watch:
        if joint not in joints:
            raise ValueError(f'Missing joint column {joint}')
        angles = joints[joint].to_numpy(float)
        velocities = np.gradient(angles, t)
        axes[1].plot(t-t[0], angles, label=joint)
        axes[2].plot(t-t[0], np.abs(velocities), label=joint)
        k = int(np.argmax(abs(velocities)))
        peak_time = float(t[k])
        nearest = min(event_times, key=lambda e: abs(e[0]-peak_time)) if event_times else None
        message = (f'nearest {nearest[1]} {nearest[2]} at Δt={peak_time-nearest[0]:+.3f}s'
                   if nearest else 'no GRF threshold crossing inside segment')
        print(f'{joint}: peak |velocity|={abs(velocities[k]):.1f} deg/s at '
              f'{peak_time:.4f}s (stride {peak_time-t[0]:.3f}s); {message}')
    axes[1].set_ylabel('Joint angle (deg)')
    axes[2].set_ylabel('|Angular speed| (deg/s)')
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)

    for marker in ('LHEE', 'LTOE', 'RHEE', 'RTOE'):
        if marker in names:
            axes[3].plot(tm-t[0], 1000*xyz[:, names.index(marker), 1], label=marker)
    axes[3].set_ylabel('Measured marker height (mm)')
    axes[3].set_xlabel('Time since trial start (s)')
    axes[3].legend(fontsize=8)

    for when, channel, kind in event_times:
        for axis in axes:
            axis.axvline(when-t[0], color='black', ls=':' if kind == 'on' else '--', alpha=.35)
    for axis in axes:
        axis.set_xlim(0, t[-1]-t[0])
        axis.grid(alpha=.2)
    fig.suptitle('Gait120: velocity peaks and force-contact transitions (unmodified data)')
    fig.tight_layout()
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(f'Plot saved: {args.output}')
    print('NOTE: force channels are not automatically assigned to left/right feet.')


if __name__ == '__main__':
    main()
