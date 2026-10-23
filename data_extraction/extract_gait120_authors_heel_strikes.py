"""Export the right-heel-strike boundaries chosen by the Gait120 authors.

Each original LevelWalking/TrialXX/StepYY.mot contains one right-HS-to-right-HS
cycle according to Boo et al., Scientific Data (2025), DOI 10.1038/s41597-025-05391-0.
The first and final frame timestamps are the *released segmentation frame times*,
not sub-frame 20 N threshold crossing estimates.

No left heel strikes or toe-offs are inferred and no data are changed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def step_times(mot_path: Path) -> tuple[float, float, int]:
    """Read the first and last timestamps of an OpenSim .mot, without pandas."""
    after_header = False
    first = last = None
    n = 0
    with mot_path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            stripped = line.strip()
            if not after_header:
                if stripped.lower() == 'endheader':
                    after_header = True
                continue
            if not stripped or stripped.lower().startswith('time'):
                continue
            parts = stripped.split()
            try:
                t = float(parts[0])
            except ValueError:
                raise ValueError(f'Invalid time in {mot_path}: {stripped[:100]}')
            if first is None:
                first = t
            if last is not None and t <= last:
                raise ValueError(f'Non-increasing time in {mot_path}')
            last = t
            n += 1
    if not after_header or n < 2:
        raise ValueError(f'No valid OpenSim time series in {mot_path}')
    return first, last, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True,
                   help='Folder containing Sxxx, or Gait120_*/Sxxx folders')
    p.add_argument('--output', type=Path, default=Path('authors_right_heel_strikes.csv'))
    p.add_argument('--start-subject', type=int, default=11)
    p.add_argument('--end-subject', type=int, default=120)
    a = p.parse_args()
    root = a.data_root
    if not root.exists():
        p.error(f'Data root not found: {root}')

    subjects = ([root] if root.name.startswith('S') and root.name[1:].isdigit()
                else list(root.glob('S[0-9][0-9][0-9]'))
                     + list(root.glob('Gait120_*/S[0-9][0-9][0-9]')))
    seen = set()
    rows = []
    for subject in sorted(subjects):
        snum = int(subject.name[1:])
        if not a.start_subject <= snum <= a.end_subject:
            continue
        for trial_dir in sorted((subject/'JointAngle'/'LevelWalking').glob('Trial*')):
            for mot in sorted(trial_dir.iterdir() if trial_dir.is_dir() else []):
                if mot.suffix.lower() != '.mot' or not mot.stem.lower().startswith('step'):
                    continue
                key = (subject.name, trial_dir.name.lower(), mot.stem.lower())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    t0, t1, nframes = step_times(mot)
                except Exception as exc:
                    raise RuntimeError(f'Could not read {mot}') from exc
                rows.append({
                    'subject': subject.name, 'trial': trial_dir.name, 'step': mot.stem,
                    'right_hs_start_s': t0, 'right_hs_end_s': t1,
                    'stride_duration_s': t1 - t0, 'n_frames': nframes,
                    'source': 'Gait120 author-selected right-HS segmentation frames',
                    'source_mot': str(mot),
                })

    if not rows:
        p.error('No LevelWalking JointAngle .mot steps found')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} steps to {a.output}')
    print('These timestamps are right-HS segmentation frame boundaries, not newly detected GRF events.')


if __name__ == '__main__':
    main()
