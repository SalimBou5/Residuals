#!/usr/bin/env python3
"""Plot real per-feature observation histograms from a recorded-state sweep.

Requires the observation_samples.npz produced by:
  python inspect_observation_ranges.py ... --plots

This script does NOT invoke MuJoCo, change normalization, or classify any
value as an error automatically. It ranks tail-heavy features for inspection.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def rank_features(values: np.ndarray, names: list[str], units: list[str],
                  reference_ids: np.ndarray, phases: np.ndarray,
                  penetrations_mm: np.ndarray, paths: list[str]) -> list[dict]:
    """Compute an interpretable tail score and the location of sample extrema."""
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError('values must be (samples, features), with one name per feature')
    if values.shape[0] != len(reference_ids) or values.shape[0] != len(phases):
        raise ValueError('sample provenance must align with observation rows')
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite observations')
    if not (len(units) == len(names) and len(penetrations_mm) == len(phases)):
        raise ValueError('feature units / sample penetration metadata missing')
    q = np.percentile(values, [1, 5, 50, 95, 99], axis=0)
    min_idx = np.argmin(values, axis=0)
    max_idx = np.argmax(values, axis=0)
    result = []
    for j, name in enumerate(names):
        lo, hi = float(values[min_idx[j], j]), float(values[max_idx[j], j])
        p01, p05, median, p95, p99 = (float(a) for a in q[:, j])
        middle_width = p99 - p01
        lower_tail = p01 - lo
        upper_tail = hi - p99
        tail_score = (max(lower_tail, upper_tail) / middle_width
                      if middle_width > 1e-5 else 0.0)
        beyond_99 = int(np.count_nonzero((values[:, j] < p01) | (values[:, j] > p99)))
        mi, ma = int(min_idx[j]), int(max_idx[j])
        result.append({
            'index': j, 'block': name.split('/', 1)[0], 'feature': name, 'unit': units[j],
            'min': lo, 'p01': p01, 'p05': p05, 'median': median,
            'p95': p95, 'p99': p99, 'max': hi,
            'tail_ratio': float(tail_score), 'outside_p01_p99_count': beyond_99,
            'min_reference': paths[int(reference_ids[mi])], 'min_phase': float(phases[mi]),
            'min_penetration_mm': float(penetrations_mm[mi]),
            'max_reference': paths[int(reference_ids[ma])], 'max_phase': float(phases[ma]),
            'max_penetration_mm': float(penetrations_mm[ma]),
        })
    return sorted(result, key=lambda x: x['tail_ratio'], reverse=True)


def plot_distributions(samples_path: str | Path, output_dir: str | Path, *,
                       top_n: int = 24, plot_all: bool = False,
                       contains: str | None = None) -> list[Path]:
    """Save tail-ranking CSV and full/zoomed histograms for selected channels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    samples_path, output_dir = Path(samples_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(samples_path, allow_pickle=False) as d:
        values = np.asarray(d['observations'], dtype=np.float32)
        names = [str(a) for a in d['feature_names']]
        units = [str(a) for a in d['units']]
        reference_ids = np.asarray(d['reference_ids'], dtype=int)
        phases = np.asarray(d['phases'], dtype=float)
        penetrations = np.asarray(d['penetration_mm'], dtype=float)
        paths = [str(a) for a in d['reference_paths']]

    if not (values.ndim == 2 and len(values) > 1 and values.shape[1] == len(names)):
        raise ValueError('No valid observations in the saved sample NPZ')
    rows = rank_features(values, names, units, reference_ids, phases, penetrations, paths)
    with (output_dir / 'outlier_candidates.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if contains:
        selected = [r for r in rows if contains.lower() in r['feature'].lower()]
    elif plot_all:
        selected = rows
    else:
        # During imposed replay the actual/ref site distributions are usually identical;
        # avoid showing the same peak twice in the default diagnostic plots.
        selected = [r for r in rows
                    if not r['feature'].startswith('reference/')
                    and r['feature'] not in ('wrapped_phase_drift',
                                             'remaining_event_correction',
                                             'event_correction_active')
                    and r['tail_ratio'] > 0][:top_n]
    selected = sorted(selected, key=lambda r: r['tail_ratio'], reverse=True)

    saved = []
    # Each row gets a full-scale and a P01–P99 zoomed histogram.
    n_per_figure = 6
    for part, start in enumerate(range(0, len(selected), n_per_figure), 1):
        subset = selected[start:start+n_per_figure]
        fig, axes = plt.subplots(len(subset), 2, figsize=(14, 2.7*len(subset)),
                                 squeeze=False, constrained_layout=True)
        for row, record in enumerate(subset):
            v = values[:, int(record['index'])]
            p01 = record['p01']; p99 = record['p99']
            for column, ax in enumerate(axes[row]):
                if column == 0:
                    ax.hist(v, bins=75)
                    ax.set_title('Full range', fontsize=9)
                else:
                    central = v[(v >= p01) & (v <= p99)]
                    if central.size and p99 > p01:
                        ax.hist(central, bins=65)
                        ax.set_xlim(p01, p99)
                    else:
                        ax.hist(v, bins=30)
                    ax.set_title('Central P01–P99 zoom', fontsize=9)
                ax.axvline(record['median'], linestyle='--', linewidth=1.1, color='black')
                ax.tick_params(labelsize=8)
                ax.set_xlabel(f"{record['unit']} (median dashed)", fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
            label = record['feature']
            axes[row, 0].text(0.02, .97,
                f"{label}\nmin {record['min']:.3g} | P99 {p99:.3g} | max {record['max']:.3g}"
                f" | tail score {record['tail_ratio']:.2f}",
                transform=axes[row, 0].transAxes, va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=.83, edgecolor='none'))
        fig.suptitle(f'Observation distributions: part {part} — {len(values):,} sampled poses',
                     fontsize=12)
        filepath = output_dir / f'distributions_{part:02d}.png'
        fig.savefig(filepath, dpi=140)
        plt.close(fig)
        saved.append(filepath)

    # Foot-contact forces deserve their own plot, because this is an imposed
    # kinematic replay where penetration can create nonphysical force spikes.
    contacts = [r for r in rows if r['feature'].startswith(('right_heel/scaled',
                                                            'right_toe/scaled',
                                                            'left_heel/scaled',
                                                            'left_toe/scaled'))]
    if len(contacts) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for ax, rec in zip(axes.flat, contacts):
            v = values[:, int(rec['index'])]
            ax.hist(v, bins=85)
            ax.set_title(f"{rec['feature']}  (P99={rec['p99']:.3g}, max={rec['max']:.3g} kN)")
            ax.set_xlabel('kN; imposed poses, NOT experimental GRFs')
            ax.set_ylabel('Sample count')
        filename = output_dir / 'contact_forces_distribution.png'
        fig.savefig(filename, dpi=140)
        plt.close(fig)
        saved.append(filename)
    return saved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', type=Path, help='observation_samples.npz from inspector --plots')
    p.add_argument('--output', type=Path, default=Path('observation_distributions'))
    p.add_argument('--top', type=int, default=24)
    p.add_argument('--all', action='store_true', help='Plot EVERY feature (many PNG pages)')
    p.add_argument('--contains', type=str, default=None,
                   help='Plot only features containing given text, e.g. mtp_angle')
    args = p.parse_args()
    if args.top < 1:
        p.error('--top must be >= 1')
    saved = plot_distributions(args.samples, args.output, top_n=args.top,
                               plot_all=args.all, contains=args.contains)
    print(f"Saved {len(saved)} plot(s) and {args.output/'outlier_candidates.csv'}")


if __name__ == '__main__':
    main()
