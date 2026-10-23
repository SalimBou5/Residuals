"""Apply a 20 ms event refractory period to existing Gait120 references.

Only edits *_events.csv and *_manifest.json for affected strides. Does not
change qpos, qvel, phase arrays, original GRF, or any NPZ file.

Each force plate has its own refractory clock; a detected 20 N threshold
crossing suppresses any later HS/TO crossing on THAT plate within the interval.

Usage:
    python apply_gait120_event_refractory.py --root gait120_references_forward_2
    python apply_gait120_event_refractory.py --root gait120_references_forward_2 --apply
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import re


def filter_events(rows, refractory_s):
    """Keep the earliest event, discard any on the same plate within cooldown."""
    last_accepted = {}
    kept, dropped = [], []
    for row in sorted(rows, key=lambda r: float(r['time_s'])):
        plate = str(row.get('plate') or row.get('force_channel', '')).strip()
        t = float(row['time_s'])
        if plate in last_accepted and t - last_accepted[plate] < refractory_s:
            dropped.append(row)
        else:
            kept.append(row)
            last_accepted[plate] = t
    return kept, dropped


def process(csv_path, refractory_s, apply):
    base = csv_path.name.replace('_events.csv', '')
    manifest_path = csv_path.with_name(base + '_manifest.json')
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with csv_path.open(newline='', encoding='utf-8-sig') as fp:
        reader = csv.DictReader(fp)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or not {'time_s', 'plate', 'event', 'foot_side'}.issubset(fields):
        raise ValueError(f'Unexpected event CSV format: {csv_path}')
    kept, dropped = filter_events(rows, refractory_s)
    if not dropped:
        return 0

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    events = manifest['events']
    # This small filter must never silently invalidate previously selected HS phase.
    if str(events.get('left_hs_source', '')).startswith('observed'):
        selected = float(events['left_hs_time_s'])
        for e in dropped:
            if (e['event'] == 'HS_candidate' and e['foot_side'] == 'left'
                    and abs(float(e['time_s']) - selected) < 0.003):
                raise ValueError(f'Refractory would remove selected left HS: {csv_path}')

    events['observed_left_toe_off_s'] = [float(e['time_s']) for e in kept
                                         if e['event'] == 'TO_candidate' and e['foot_side'] == 'left']
    events['observed_right_toe_off_s'] = [float(e['time_s']) for e in kept
                                          if e['event'] == 'TO_candidate' and e['foot_side'] == 'right']
    for plate in ('1', '2'):
        if plate in manifest.get('contacts_by_plate', {}):
            manifest['contacts_by_plate'][plate]['number_threshold_crossings'] = sum(
                str(e['plate']) == plate for e in kept)
    manifest['event_refractory'] = {
        'duration_s': refractory_s,
        'scope': 'per_force_plate',
        'method': 'Ignore HS and TO crossings within duration_s after any accepted crossing',
        'suppressed_events': dropped,
        'raw_grf_preserved_in_npz': True,
    }
    if apply:
        backups = [p.with_name(p.name + '.before_refractory.bak')
                   for p in (csv_path, manifest_path)]
        if any(p.exists() for p in backups):
            raise FileExistsError(f'Refractory backup already exists: {csv_path}')
        shutil.copy2(csv_path, backups[0])
        shutil.copy2(manifest_path, backups[1])
        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(kept)
        temp_csv = csv_path.with_name(csv_path.name + '.tmp')
        temp_manifest = manifest_path.with_name(manifest_path.name + '.tmp')
        try:
            temp_csv.write_text(buf.getvalue(), encoding='utf-8')
            temp_manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            temp_csv.replace(csv_path)
            temp_manifest.replace(manifest_path)
        finally:
            temp_csv.unlink(missing_ok=True)
            temp_manifest.unlink(missing_ok=True)
    return len(dropped)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--refractory-ms', type=float, default=20.0)
    ap.add_argument('--apply', action='store_true', help='Otherwise report only')
    args = ap.parse_args()
    if args.refractory_ms < 0:
        ap.error('--refractory-ms must be nonnegative')
    paths = sorted(
        p for p in args.root.glob('S[0-9][0-9][0-9]/Trial*/step*_events.csv')
        if re.fullmatch(r'step[0-9]+_events\.csv', p.name)
    )
    if not paths:
        ap.error(f'No Gait120 event CSVs found under {args.root}')
    changed = removed = 0
    for path in paths:
        count = process(path, args.refractory_ms / 1000.0, args.apply)
        if count:
            changed += 1
            removed += count
            print(f'{path.relative_to(args.root)}: suppressed {count} crossing(s)')
    print(f'{len(paths)} strides checked; {changed} affected; '
          f'{removed} crossings suppressed; mode={"APPLIED" if args.apply else "DRY RUN"}.')
    print('NPZ (kinematics, reference phase, and raw GRF) unchanged.')


if __name__ == '__main__':
    main()
