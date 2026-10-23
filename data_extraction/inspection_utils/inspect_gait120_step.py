"""Inspect one Gait120 trial without changing any data.

Usage:
  python inspect_gait120_step.py --joint-mot 'step01(1).mot' \
    --grf-mot 'Step01(2).mot' --trc Step01.trc --output-dir .

Dependencies: numpy, pandas, matplotlib
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_mot(path):
    """Read OpenSim .mot with an endheader line."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    try:
        end = next(i for i, line in enumerate(lines)
                   if line.strip().lower() == 'endheader')
    except StopIteration as exc:
        raise ValueError(f'No endheader in {path}') from exc
    metadata = lines[:end]
    df = pd.read_csv(path, sep=r'\s+', skiprows=end + 1)
    return df, metadata


def read_trc(path):
    """Read a conventional OpenSim TRC: positions are converted to meters."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    names = [name.strip() for name in lines[3].split('\t')[2:] if name.strip()]
    meta_keys = lines[1].split('\t')
    meta_vals = lines[2].split('\t')
    meta = dict(zip(meta_keys, meta_vals))
    units = meta.get('Units', '').strip()
    if units not in ('mm', 'm'):
        raise ValueError(f'Unexpected TRC units: {units!r}')
    frames, times, values = [], [], []
    for line in lines[5:]:
        if not line.strip():
            continue
        fields = line.split('\t')
        frames.append(int(fields[0]))
        times.append(float(fields[1]))
        coords = fields[2:2 + 3 * len(names)]
        values.append([float(v) if v.strip() else np.nan for v in coords])
    xyz = np.asarray(values).reshape(len(times), len(names), 3)
    if units == 'mm':
        xyz /= 1000.0
    return np.asarray(times), names, xyz, meta


def describe_time(label, t, values):
    dt = np.diff(t)
    assert np.all(dt > 0), f'{label}: time must increase'
    print(f'{label}: {len(t)} frames; {t[0]:.4f}–{t[-1]:.4f} s; '
          f'fs≈{1 / np.median(dt):.0f} Hz; missing={np.isnan(values).sum()}')


def contact_intervals(t, vertical_force, threshold=20):
    """Return threshold crossings, keeping boundary-truncated intervals explicit."""
    mask = np.asarray(vertical_force) > threshold
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [(float(t[s]), float(t[e]), s == 0, e == len(mask) - 1)
            for s, e in zip(starts, ends)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint-mot', required=True)
    ap.add_argument('--grf-mot', required=True)
    ap.add_argument('--trc', required=True)
    ap.add_argument('--output-dir', default='.')
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    joint, joint_meta = read_mot(args.joint_mot)
    grf, grf_meta = read_mot(args.grf_mot)
    tm, names, xyz, trc_meta = read_trc(args.trc)
    tj, tg = joint['time'].to_numpy(), grf['time'].to_numpy()

    print('=== Raw file inspection (no data modifications) ===')
    describe_time('Joint angles', tj, joint.iloc[:, 1:].to_numpy())
    describe_time('TRC markers', tm, xyz)
    describe_time('Ground forces', tg, grf.iloc[:, 1:].to_numpy())
    print(f'Markers: {len(names)}; coordinates: {len(joint.columns) - 1}; '
          f'GRF signals: {len(grf.columns) - 1}')
    print('Joint MOT inDegrees:', next((s for s in joint_meta
          if s.lower().startswith('indegrees')), 'not specified'))
    print('TRC units:', trc_meta.get('Units'))
    print('Joint and marker timestamps match:',
          len(tj) == len(tm) and np.allclose(tj, tm, atol=1e-9))
    print('Force interval matches kinematic interval:',
          abs(tg[0] - tj[0]) < 1e-8 and abs(tg[-1] - tj[-1]) <= .001)

    print('\n=== Knee angles (original units; no sign conversion) ===')
    for side in ('r', 'l'):
        col = f'knee_angle_{side}'
        print(f'{col}: min={joint[col].min():.2f}, max={joint[col].max():.2f} deg')

    # Independent sanity check: IK pelvis translation vs measured ASIS midpoint.
    # These are different anatomical points; a constant offset is expected.
    pelvis = joint[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy()
    asis = xyz[:, [names.index('LASI'), names.index('RASI')], :].mean(axis=1)
    offset = asis - pelvis
    print('\n=== Root translation sanity check ===')
    print('IK pelvis displacement XYZ (m):', np.round(pelvis[-1] - pelvis[0], 4))
    print('ASIS-midpoint displacement XYZ (m):', np.round(asis[-1] - asis[0], 4))
    print('ASIS - IK pelvis offset mean XYZ (m):', np.round(offset.mean(axis=0), 4))
    print('ASIS - IK pelvis offset std XYZ (m):', np.round(offset.std(axis=0), 4))

    print('\n=== Vertical GRF > 20 N ===')
    intervals_by_plate = {}
    for plate in (1, 2):
        col = f'ground_force{plate}_vy'  # OpenSim Y = vertical
        force = grf[col].to_numpy()
        intervals = contact_intervals(tg, force)
        intervals_by_plate[plate] = intervals
        print(f'Plate {plate}: max={force.max():.1f} N; intervals={intervals}')
        for start, end, left_cut, right_cut in intervals:
            idx = np.argmin(np.abs(tg - start))
            cop_x = float(grf[f'ground_force{plate}_px'].iloc[idx])
            # Only a geometric clue, NOT a verified plate-to-foot mapping.
            heel_x = {side: float(np.interp(start, tm, xyz[:, names.index(side+"HEE"), 0]))
                      for side in ('L', 'R')}
            closest = min(heel_x, key=lambda s: abs(heel_x[s] - cop_x))
            print(f'  onset t={start:.4f}: CoP X={cop_x:.3f} m, '
                  f'LHEE X={heel_x["L"]:.3f}, RHEE X={heel_x["R"]:.3f}; '
                  f'X-only closest heel={closest}; still active at clip end={right_cut}')

    fig, ax = plt.subplots(figsize=(10, 4))
    for plate in (1, 2):
        ax.plot(tg, grf[f'ground_force{plate}_vy'], label=f'Force {plate}: vertical (Y)')
    ax.axhline(20, color='gray', linewidth=1, linestyle='--', label='20 N threshold')
    ax.set(xlabel='Absolute time (s)', ylabel='Vertical GRF (N)',
           title='Gait120: force channels (unmodified)')
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'gait120_grf.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for side in ('L', 'R'):
        for part in ('HEE', 'TOE'):
            y = xyz[:, names.index(side + part), 1]  # OpenSim Y = vertical
            ax.plot(tm, y, label=f'{side}{part} Y')
    for plate, intervals in intervals_by_plate.items():
        for start, _, _, _ in intervals:
            ax.axvline(start, linewidth=1, linestyle=':', alpha=.7,
                       label=f'Force {plate} > 20 N at {start:.4f} s')
    ax.set(xlabel='Absolute time (s)', ylabel='Marker height (m)',
           title='Gait120: measured heel/toe marker heights')
    ax.grid(alpha=.2)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'gait120_heel_heights.png', dpi=170)
    plt.close(fig)
    print('\nSaved plots to:', out.resolve())


if __name__ == '__main__':
    main()
