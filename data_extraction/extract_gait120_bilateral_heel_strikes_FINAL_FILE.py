"""Use Gait120 authors' right-HS stride boundaries and detect left HS.

Input: the NPZ references generated from the original Gait120 MOT/TRC data.
All force and marker signals are read from the original, preserved arrays in NPZ.
Force plate events are detected at TRIAL level, before assigning them to strides.

Priority:
  1. 20 N vertical-force onset, assigned to LEFT using loaded-bout CoP + markers.
  2. Optional 8 N onset with peak 15-20 N (flagged LOW_FORCE, NOT paper method).
  3. Observed left-heel marker's landing-like minimum, calibrated against 20 N
     left HS events from this dataset (ESTIMATE, NOT measured GRF).

Right HS: exported stride first and last timestamp (authors' segmentation).
Never invent a GRF crossing. Do not search for / modify toe-off events.
New output folder; input references are never modified.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


MIN_DWELL_S = 0.020
PHASE_WINDOW = (0.20, 0.80)  # Conservative left-HS candidate QC, not an author rule.


def runs(states):
    changes = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    return [(int(a), int(b), bool(states[a])) for a, b in zip(changes[:-1], changes[1:])]


def clean_contacts(t, force, threshold, dwell=MIN_DWELL_S):
    """Remove isolated *interior* contact-state blips shorter than 20 ms."""
    state = np.asarray(force >= threshold, dtype=bool).copy()
    for _ in range(100):
        rr = runs(state)
        changed = False
        for j in range(1, len(rr) - 1):
            a, b, val = rr[j]
            if rr[j - 1][2] == rr[j + 1][2] != val:
                duration = t[b] - t[a] if b < len(t) else t[-1] - t[a]
                if duration < dwell:
                    state[a:b] = not val
                    changed = True
                    break
        if not changed:
            break
    return state


def interpolate_crossing(t, f, index, threshold):
    a, b = float(f[index - 1]), float(f[index])
    frac = (threshold - a) / (b - a) if b != a else 0.5
    return float(t[index - 1] + np.clip(frac, 0., 1.) * (t[index] - t[index - 1]))


def combine(refs, name_t, name_v, name_cols):
    """Concatenate source data from all steps of one trial; deduplicate overlaps."""
    chunks_t, chunks_v, columns = [], [], None
    for r in refs:
        with np.load(r, allow_pickle=False) as d:
            tt = np.asarray(d[name_t], dtype=float)
            v = np.asarray(d[name_v], dtype=float)
            cc = [str(x) for x in d[name_cols].tolist()]
        if columns is not None and cc != columns:
            raise ValueError(f'Different {name_cols} across {refs}')
        columns = cc
        chunks_t.append(tt)
        chunks_v.append(v)
    tt = np.concatenate(chunks_t)
    vv = np.concatenate(chunks_v)
    order = np.argsort(tt, kind='stable')
    tt, vv = tt[order], vv[order]
    keep = np.r_[True, np.diff(tt) > 1e-7]
    return tt[keep], vv[keep], columns


def interpolate_markers(t, markers, query):
    if query < t[0] - .015 or query > t[-1] + .015:
        return None
    k = int(np.searchsorted(t, query))
    if k == 0:
        return markers[0] if abs(t[0] - query) < .015 else None
    if k == len(t):
        return markers[-1] if abs(t[-1] - query) < .015 else None
    if t[k] - t[k - 1] > .04:
        return None
    a = float((query - t[k - 1]) / (t[k] - t[k - 1]))
    return markers[k - 1] * (1 - a) + markers[k] * a


def point_segment_distance(point, a, b):
    ab = b - a
    u = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(point - (a + u * ab)))


def label_bout_side(ft, grf, cols, plate, start, end, mt, markers, names, threshold):
    needed = ('LHEE', 'LTOE', 'RHEE', 'RTOE')
    if any(n not in names for n in needed):
        return 'unassigned', np.nan, np.nan
    cx, cz, force = (f'ground_force{plate}_{s}' for s in ('px', 'pz', 'vy'))
    if any(n not in cols for n in (cx, cz, force)):
        return 'unassigned', np.nan, np.nan
    f = grf[start:end, cols.index(force)]
    peak = float(f.max())
    good = np.flatnonzero(f >= max(threshold, 0.25 * peak))
    if not len(good):
        return 'unassigned', np.nan, np.nan
    # Only use CoP while the plate is loaded; at low force CoP can be unreliable.
    good = good[np.unique(np.linspace(0, len(good) - 1, min(9, len(good))).astype(int))] + start
    distances = {'left': [], 'right': []}
    for idx in good:
        xyz = interpolate_markers(mt, markers, float(ft[idx]))
        if xyz is None:
            continue
        point = grf[idx, [cols.index(cx), cols.index(cz)]]
        if not np.isfinite(point).all():
            continue
        for side, prefix in [('left', 'L'), ('right', 'R')]:
            a, b = xyz[names.index(prefix + 'HEE'), [0, 2]], xyz[names.index(prefix + 'TOE'), [0, 2]]
            distances[side].append(point_segment_distance(point, a, b))
    if not all(distances.values()):
        return 'unassigned', np.nan, np.nan
    m = {side: float(np.median(ds)) for side, ds in distances.items()}
    selected = min(m, key=m.get)
    other = 'right' if selected == 'left' else 'left'
    margin = m[other] - m[selected]
    # These are our practical geometry-QC tolerances, not part of the Gait120 paper.
    if m[selected] <= .18 and margin >= .06:
        return selected, m[selected], margin
    return 'unassigned', m[selected], margin


def force_hs_trial(refs):
    ft, grf, cols = combine(refs, 'grf_time_s', 'grf_values', 'grf_channel_names')
    mt, markers, names = combine(refs, 'marker_time_s', 'marker_xyz_os_m', 'marker_names')
    if not all(n in cols for n in ('ground_force1_vy', 'ground_force2_vy')):
        raise ValueError(f'Missing vertical force columns: {cols}')
    candidates = []
    med_dt = float(np.median(np.diff(ft)))
    gap_idx = np.r_[0, np.flatnonzero(np.diff(ft) > max(.01, 5 * med_dt)) + 1, len(ft)]
    for plate in (1, 2):
        fidx = cols.index(f'ground_force{plate}_vy')
        for a, b in zip(gap_idx[:-1], gap_idx[1:]):
            if b - a < 3:
                continue
            tt, ff = ft[a:b], grf[a:b, fidx]
            for threshold, label in [(20., 'GRF_20N'), (8., 'LOW_FORCE_GRF_CANDIDATE')]:
                state = clean_contacts(tt, ff, threshold)
                for s, e, loaded in runs(state):
                    if not loaded or s == 0:
                        continue  # Beginning already loaded: onset was not observed.
                    peak = float(np.max(ff[s:e]))
                    duration = float(tt[e-1] - tt[s])
                    if label == 'GRF_20N' and (peak < 20 or duration < .025):
                        continue
                    if label != 'GRF_20N' and not (15 <= peak < 20 and duration >= .025):
                        continue
                    moment = interpolate_crossing(tt, ff, s, threshold)
                    side, dist, margin = label_bout_side(ft, grf, cols, plate,
                                                         a+s, a+e, mt, markers, names, threshold)
                    candidates.append(dict(time_s=moment, side=side, source=label,
                                           plate=plate, bout_peak_N=peak,
                                           cop_distance_m=dist, cop_margin_m=margin))
    return sorted(candidates, key=lambda x: x['time_s'])


def smooth(x, dt):
    win = max(5, int(round(.07 / dt)) | 1)
    win = min(win, len(x) if len(x) % 2 else len(x) - 1)
    return savgol_filter(x, win, 2) if win >= 5 else x


def heel_marker_landing(t, name, xyz):
    """Approximate foot landing from marker HEIGHT, not a measured HS."""
    if 'LHEE' not in name or len(t) < 12:
        return np.nan, 'NO_LEFT_HEEL_MARKER'
    h = xyz[:, name.index('LHEE'), 1]  # Original OpenSim +Y is vertical
    if not np.isfinite(h).all():
        return np.nan, 'INVALID_LEFT_HEEL_HEIGHT'
    h = smooth(h, float(np.median(np.diff(t))))
    phase = (t - t[0]) / (t[-1] - t[0])
    mask = (phase >= .3) & (phase <= .7)
    inds = np.flatnonzero(mask)
    if len(inds) < 5:
        return np.nan, 'TOO_FEW_MIDSTRIDE_MARKERS'
    # Height minimum is a kinematic proxy. Require visible swing-to-ground descent.
    low_idx = inds[int(np.argmin(h[inds]))]
    prev = max(0, low_idx - int(round(.2 / np.median(np.diff(t)))))
    prior_peak = float(np.max(h[prev:low_idx+1]))
    if prior_peak - h[low_idx] < .03:
        return np.nan, 'NO_CLEAR_LEFT_HEEL_DESCENT'
    if low_idx <= inds[1] or low_idx >= inds[-2]:
        return np.nan, 'HEEL_MINIMUM_AT_WINDOW_EDGE'
    return float(t[low_idx]), 'LEFT_HEEL_HEIGHT_MINIMUM'


def step_metadata(path):
    with np.load(path, allow_pickle=False) as x:
        t = np.asarray(x['time_s'], dtype=float)
        mt = np.asarray(x['marker_time_s'], dtype=float)
        names = [str(a) for a in x['marker_names']]
        xyz = np.asarray(x['marker_xyz_os_m'], dtype=float)
    return t, heel_marker_landing(mt, names, xyz)


def choose_left(t, candidates, heel_time):
    start, end = float(t[0]), float(t[-1])
    duration = end - start
    viable = []
    for c in candidates:
        p = (c['time_s'] - start) / duration
        if c['side'] != 'left' or not (PHASE_WINDOW[0] <= p <= PHASE_WINDOW[1]):
            continue
        if np.isfinite(heel_time) and abs(c['time_s'] - heel_time) > .13:
            continue
        viable.append(c)
    # A strong, geometrically assigned 20 N event takes precedence over weak-force proposals.
    viable.sort(key=lambda c: (c['source'] != 'GRF_20N',
                               abs(c['time_s'] - heel_time) if np.isfinite(heel_time) else abs((c['time_s']-start)/duration-.5),
                               -c['bout_peak_N']))
    return (viable[0] if viable else None), len(viable)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def save_reference(src, dest, left_time, source, note, row):
    with np.load(src, allow_pickle=False) as f:
        d = {key: f[key].copy() for key in f.files}
    t = np.asarray(d['time_s'], dtype=float)
    phase_right = (t-t[0]) / (t[-1]-t[0])
    phase_left = np.mod(phase_right - (left_time-t[0])/(t[-1]-t[0]), 1.) if np.isfinite(left_time) else np.full_like(t, np.nan)
    d.update(phase_right_01=phase_right, phase_right_rad=2*np.pi*phase_right,
             phase_left_01=phase_left, phase_left_rad=2*np.pi*phase_left,
             left_hs_s=np.float64(left_time))
    old_manifest = src.with_name(src.stem+'_manifest.json')
    manifest = json.loads(old_manifest.read_text(encoding='utf-8')) if old_manifest.exists() else {}
    ev = manifest.setdefault('events', {})
    ev.update(right_hs_boundaries_s=[float(t[0]),float(t[-1])],
              right_hs_boundary_source='Gait120_authors_stride_segmentation',
              left_hs_time_s=float(left_time) if np.isfinite(left_time) else None,
              left_hs_source=source, left_hs_method_note=note,
              left_hs_phase_in_right_stride=float((left_time-t[0])/(t[-1]-t[0])) if np.isfinite(left_time) else None)
    manifest['bilateral_HS_extraction'] = row
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **d)
    dest.with_name(dest.stem+'_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    # Do not preserve old candidate CSV with misleading markers. The existing
    # replay can draw a *measured* left HS from one row; marker estimates are
    # rendered amber via the manifest's ASSUMED source label.
    measured = source.startswith(('GRF_20N', 'LOW_FORCE'))
    event_rows = ([dict(time_s=float(left_time), event='HS_candidate', foot_side='left',
                        plate=row['force_plate'], event_source=source)] if measured else [])
    write_csv(dest.with_name(dest.stem+'_events.csv'), event_rows,
              ['time_s', 'event', 'foot_side', 'plate', 'event_source'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference-root', type=Path, required=True,
                    help='Original raw-derived NPZ references (not previously re-estimated event files)')
    ap.add_argument('--output-root', type=Path, required=True,
                    help='New folder: summary and optionally corrected NPZ copies')
    ap.add_argument('--csv-only', action='store_true', help='Only export CSVs; do not copy NPZ')
    ap.add_argument('--no-marker-fallback', action='store_true', help='Leave left HS missing if no GRF onset is available')
    ap.add_argument('--start-subject', type=int, default=11)
    ap.add_argument('--end-subject', type=int, default=120)
    ap.add_argument('--limit', type=int, default=0, help='Process first N complete trials (0=all)')
    args = ap.parse_args()
    if not args.reference_root.is_dir():
        ap.error(f'Missing --reference-root: {args.reference_root}')
    if args.reference_root.resolve() == args.output_root.resolve() or args.reference_root.resolve() in args.output_root.resolve().parents:
        ap.error('Output must differ from input and must not be inside input')
    groups = defaultdict(list)
    for ref in sorted(args.reference_root.glob('S[0-9][0-9][0-9]/Trial*/step*.npz')):
        subject = int(ref.parent.parent.name[1:])
        if args.start_subject <= subject <= args.end_subject:
            groups[(ref.parent.parent.name, ref.parent.name)].append(ref)
    keys = sorted(groups)
    if args.limit:
        keys = keys[:args.limit]
    if not keys:
        ap.error('No step*.npz found under subject/trial folders')
    print(f'Found {len(keys)} trials and {sum(map(lambda k:len(groups[k]),keys))} strides')
    recs, candidate_rows, errors = [], [], []
    for j, key in enumerate(keys, 1):
        refs = sorted(groups[key])
        try:
            all_candidates = force_hs_trial(refs)
            prelim = []
            for ref in refs:
                t, (heel_t, heel_status) = step_metadata(ref)
                chosen, n = choose_left(t, all_candidates, heel_t)
                prelim.append((ref, t, heel_t, heel_status, chosen, n))
            # Calibrate marker-minimum -> GRF HS offset, first within trial.
            for ref, t, heel_t, status, chosen, n in prelim:
                if chosen and chosen['source'] == 'GRF_20N' and np.isfinite(heel_t):
                    delta = chosen['time_s'] - heel_t
                    if abs(delta) < .12:
                        candidate_rows.append(dict(subject=key[0], trial=key[1],
                                                   offset_s=float(delta)))
            for ref, t, heel_t, heel_status, chosen, n in prelim:
                recs.append(dict(ref=ref, t=t, heel_t=heel_t, heel_status=heel_status,
                                 chosen=chosen, n_candidates=n, subject=key[0], trial=key[1]))
        except Exception as e:
            errors.append(dict(subject=key[0], trial=key[1], error=f'{type(e).__name__}: {e}'))
        if j % 50 == 0 or j == len(keys):
            print(f'Inspected {j}/{len(keys)} trials')
    offsets = np.asarray([r['offset_s'] for r in candidate_rows], dtype=float)
    global_offset = float(np.median(offsets)) if len(offsets) else 0.
    subject_offsets = defaultdict(list)
    for r in candidate_rows:
        subject_offsets[r['subject']].append(r['offset_s'])
    rows = []
    for r in recs:
        t, chosen, heel_t = r['t'], r['chosen'], r['heel_t']
        t0, t1 = float(t[0]), float(t[-1])
        if chosen:
            hs, source = float(chosen['time_s']), chosen['source']
            note = 'Measured GRF onset' if source == 'GRF_20N' else 'WEAK GRF below paper 20N threshold; review'
            plate = str(chosen['plate'])
        elif np.isfinite(heel_t) and not args.no_marker_fallback:
            ss = subject_offsets[r['subject']]
            offset = float(np.median(ss)) if ss else global_offset
            hs = float(np.clip(heel_t + offset, t0+.2*(t1-t0), t0+.8*(t1-t0)))
            source, plate = 'ASSUMED_marker_kinematic_estimate_NOT_GRF', ''
            note = f'Heel vertical minimum + measured-data median offset {offset:+.4f}s; NOT a GRF observation'
        else:
            hs, source, plate = float('nan'), 'MISSING_NO_OBSERVABLE_EVENT', ''
            note = r['heel_status']
        row = dict(subject=r['subject'], trial=r['trial'], step=r['ref'].stem,
                   right_hs_start_s=t0, right_hs_end_s=t1,
                   right_hs_source='AUTHORS_SEGMENTATION', stride_duration_s=t1-t0,
                   left_hs_s=hs if np.isfinite(hs) else '',
                   left_hs_phase=(hs-t0)/(t1-t0) if np.isfinite(hs) else '',
                   left_hs_source=source, force_plate=plate,
                   marker_heel_min_s=float(heel_t) if np.isfinite(heel_t) else '',
                   n_valid_left_grf_candidates=r['n_candidates'],
                   evidence_note=note)
        rows.append(row)
        if not args.csv_only:
            dst = args.output_root / r['ref'].relative_to(args.reference_root)
            save_reference(r['ref'], dst, hs, source, note, row)
    fields = ['subject','trial','step','right_hs_start_s','right_hs_end_s','right_hs_source',
              'stride_duration_s','left_hs_s','left_hs_phase','left_hs_source',
              'force_plate','marker_heel_min_s','n_valid_left_grf_candidates','evidence_note']
    write_csv(args.output_root/'bilateral_heel_strikes.csv', rows, fields)
    write_csv(args.output_root/'calibration_offsets.csv', candidate_rows,
              ['subject','trial','offset_s'])
    write_csv(args.output_root/'heel_strike_failures.csv', errors,
              ['subject','trial','error'])
    n20 = sum(x['left_hs_source']=='GRF_20N' for x in rows)
    nweak = sum(x['left_hs_source']=='LOW_FORCE_GRF_CANDIDATE' for x in rows)
    nmarker = sum(x['left_hs_source'].startswith('ASSUMED') for x in rows)
    missing = len(rows) - n20 - nweak - nmarker
    print(f'Strides: {len(rows)} | Left HS: 20N={n20}, weak GRF={nweak}, '
          f'marker estimate={nmarker}, missing={missing} | trial errors={len(errors)}')
    print(f'Marker-to-force calibration: {len(offsets)} samples, median {global_offset:+.4f}s')
    print(f'CSV: {args.output_root / "bilateral_heel_strikes.csv"}')


if __name__ == '__main__':
    main()
