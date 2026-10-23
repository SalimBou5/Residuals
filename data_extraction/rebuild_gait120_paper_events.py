"""Rebuild Gait120 contact annotations from untouched GRFs stored in the .npz files.

Paper rule (Boo et al., Scientific Data 2025, Data processing): for right
foot on level walking, heel strike = upward 20 N vertical-GRF crossing,
toe-off = downward 20 N crossing. A cycle is segmented right HS -> right HS.

Extras NOT in the paper:
- user-requested 20 ms refractory period per plate, avoiding millisecond chatter
- optional left-foot crossing labels obtained from CoP-to-foot marker proximity
- explicit exclusion of two known false left-HS candidates, S029 and S115

NO imputation/phase templates. Absent events remain NaN/unobserved. Read-only
with respect to the input; writes a separate output tree.

The existing heel-strike boundary of each published step is preserved even
when its 20 N crossing isn't present in the saved (cropped) GRF array.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

THRESHOLD_N = 20.0
# Previous visual QC: transient upward crossings at toe-off are not left HS.
MANUAL_REJECTIONS = {
    ('S029', 'Trial05', 'step01'): (1, 0.10181995784713907),
    ('S115', 'Trial03', 'step02'): (1, 0.11788464413966933),
}
EVENT_FIELDS = [
    'time_s', 'time_since_start_s', 'normalized_phase', 'plate',
    'force_channel', 'threshold_N', 'event', 'foot_side', 'side_method',
    'nearest_distance_m', 'side_margin_m', 'event_source',
]
RAW_FIELDS = ['reference', 'plate', 'time_s', 'normalized_phase', 'event',
              'foot_side', 'status', 'reason']


def crossings_from_force(time_s, force_N):
    """All interpolated upward/downward 20N threshold crossings, no smoothing."""
    t = np.asarray(time_s, dtype=float)
    f = np.asarray(force_N, dtype=float)
    above = f >= THRESHOLD_N
    indices = np.flatnonzero(above[1:] != above[:-1]) + 1
    crossings = []
    for i in indices:
        frac = ((THRESHOLD_N - f[i-1]) / (f[i] - f[i-1]))
        when = float(t[i-1] + np.clip(frac, 0., 1.) * (t[i] - t[i-1]))
        crossings.append({'time_s': when, 'kind': 'HS' if above[i] else 'TO'})
    return bool(above[0]), crossings


def refractory_filter(starts_loaded, crossings, refractory_s, manual_phase=None, t0=None, T=None):
    """Ignore subsequent edges for refractory_s; keep logical state coherent."""
    state = bool(starts_loaded)
    last_event = -np.inf
    for row in crossings:
        kind, when = row['kind'], row['time_s']
        expected_kind = 'TO' if state else 'HS'
        if (manual_phase is not None and kind == 'HS'
                and abs((when-t0)/T-manual_phase) < .025):
            row['status'] = 'rejected'
            row['reason'] = 'manual_QC_false_left_HS'
        elif when - last_event < refractory_s - 1.e-12:
            row['status'] = 'rejected'
            row['reason'] = 'refractory'
        elif kind != expected_kind:
            # When a transition was suppressed, the following raw edge may
            # no longer constitute a transition of the accepted contact state.
            row['status'] = 'rejected'
            row['reason'] = 'no_change_of_accepted_contact_state'
        else:
            row['status'] = 'accepted'
            row['reason'] = ''
            last_event = when
            state = (kind == 'HS')
    return crossings


def assign_side(when, plate, kind, gtime, names, forces, mtime, mnames, mxyz):
    """Extra (not described by paper): infer side via CoP vs foot heel-to-toe."""
    px, pz = f'ground_force{plate}_px', f'ground_force{plate}_pz'
    if px not in names or pz not in names:
        return 'unassigned', 'missing_cop', np.nan, np.nan
    if not all(n in mnames for n in ('LHEE', 'RHEE', 'LTOE', 'RTOE')):
        return 'unassigned', 'missing_foot_markers', np.nan, np.nan
    gi = int(np.argmin(np.abs(gtime - when)))
    mi = int(np.argmin(np.abs(mtime - when)))
    cop = forces[gi, [names.index(px), names.index(pz)]]
    scores = {}
    for side, code in [('left', 'L'), ('right', 'R')]:
        heel = mxyz[mi, mnames.index(code + 'HEE'), [0, 2]]
        toe = mxyz[mi, mnames.index(code + 'TOE'), [0, 2]]
        delta = toe - heel
        frac = np.clip(np.dot(cop - heel, delta) / max(np.dot(delta, delta), 1.e-12), 0., 1.)
        segment_dist = float(np.linalg.norm(cop - (heel + frac * delta)))
        heel_dist = float(np.linalg.norm(cop - heel))
        scores[side] = (heel_dist, segment_dist)
    component = 0 if kind == 'HS' else 1
    order = sorted(scores, key=lambda s: scores[s][component])
    a, b = order
    nearest, margin = scores[a][component], scores[b][component] - scores[a][component]
    close = nearest <= (0.18 if kind == 'HS' else 0.14)
    if close and margin >= 0.07:
        return a, 'cop_marker_geometry', float(nearest), float(margin)
    return 'unassigned', 'ambiguous_cop_marker_geometry', float(nearest), float(margin)


def collect_one(src: Path, in_root: Path, refractory_s: float):
    with np.load(src, allow_pickle=False) as npz:
        arrays = {key: np.array(npz[key]) for key in npz.files}
    required = {'time_s', 'grf_time_s', 'grf_values', 'grf_channel_names',
                'marker_time_s', 'marker_xyz_os_m', 'marker_names', 'qpos', 'qvel'}
    if required - arrays.keys():
        raise ValueError(f'Missing NPZ fields: {sorted(required - arrays.keys())}')
    time = np.asarray(arrays['time_s'], float)
    gt = np.asarray(arrays['grf_time_s'], float)
    gv = np.asarray(arrays['grf_values'], float)
    mt = np.asarray(arrays['marker_time_s'], float)
    mx = np.asarray(arrays['marker_xyz_os_m'], float)
    names = [str(x) for x in arrays['grf_channel_names']]
    mnames = [str(x) for x in arrays['marker_names']]
    if any(len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.diff(x)>0)
           for x in (time, gt, mt)):
        raise ValueError('Missing, nonfinite or unsorted times')
    if gv.shape != (len(gt), len(names)) or mx.shape != (len(mt), len(mnames), 3):
        raise ValueError('GRF/marker shape mismatch')
    if not np.isfinite(gv).all() or not np.isfinite(mx).all():
        raise ValueError('Nonfinite GRF or marker data')
    t0, t1 = float(time[0]), float(time[-1])
    T = t1 - t0
    key = (src.parent.parent.name, src.parent.name, src.stem)
    manual_rule = MANUAL_REJECTIONS.get(key)
    kept, all_raw = [], []
    for plate in (1, 2):
        channel = f'ground_force{plate}_vy'
        if channel not in names:
            raise ValueError(f'No {channel} in NPZ')
        starts_loaded, crossings = crossings_from_force(gt, gv[:, names.index(channel)])
        manual_phase = manual_rule[1] if manual_rule is not None and plate == manual_rule[0] else None
        refractory_filter(starts_loaded, crossings, refractory_s, manual_phase, t0, T)
        # If the force plate starts loaded, the next TO belongs to the same
        # already-ongoing contact. Infer side from CoP at the beginning.
        side = 'unassigned'
        if starts_loaded:
            side, _, _, _ = assign_side(gt[0], plate, 'TO', gt, names, gv, mt, mnames, mx)
        for row in crossings:
            when, kind = row['time_s'], row['kind']
            if not (t0 <= when <= t1):
                continue
            status, reason = row['status'], row['reason']
            d = margin = np.nan
            method = 'unassigned'
            current_side = 'unassigned'
            if status == 'accepted':
                if kind == 'HS':
                    current_side, method, d, margin = assign_side(
                        when, plate, 'HS', gt, names, gv, mt, mnames, mx)
                    side = current_side
                else:
                    current_side = side
                    method = 'same_plate_contact_bout'
                    if current_side == 'unassigned':
                        current_side, method, d, margin = assign_side(
                            when, plate, 'TO', gt, names, gv, mt, mnames, mx)
                    side = 'unassigned'
                phase = (when-t0)/T
                kept.append({
                        'time_s': f'{when:.12f}',
                        'time_since_start_s': f'{when-t0:.12f}',
                        'normalized_phase': f'{phase:.12f}',
                        'plate': plate, 'force_channel':channel, 'threshold_N':THRESHOLD_N,
                        'event':kind+'_candidate', 'foot_side':current_side,
                        'side_method': method,
                        'nearest_distance_m':float(d), 'side_margin_m':float(margin),
                        'event_source':'observed_20N_crossing',
                })
            all_raw.append({
                'reference':str(src.relative_to(in_root)).replace('\\', '/'),
                'plate':plate, 'time_s':f'{when:.12f}',
                'normalized_phase':f'{(when-t0)/T:.12f}', 'event':kind,
                'foot_side':current_side, 'status':status, 'reason':reason})
    kept.sort(key=lambda e:float(e['time_s']))
    all_raw.sort(key=lambda e:float(e['time_s']))
    return arrays, kept, all_raw


def unique_time(events, side, kind):
    selected = [float(e['time_s']) for e in events
                if e['foot_side'] == side and e['event'] == kind+'_candidate']
    return selected[0] if len(selected) == 1 else float('nan')


def save_one(src, input_root, output_root, refractory_s):
    arrays, events, raw = collect_one(src, input_root, refractory_s)
    time = np.asarray(arrays['time_s'], float)
    t0, t1 = float(time[0]), float(time[-1]); T = t1-t0
    lhs = unique_time(events,'left','HS')
    lto = unique_time(events,'left','TO')
    rto = unique_time(events,'right','TO')
    phase_r = (time-t0)/T
    arrays['phase_right_01'] = phase_r
    arrays['phase_right_rad'] = phase_r * (2*np.pi)
    # Do not keep the previous guessed half-cycle left phase when HS is absent.
    if np.isfinite(lhs):
        left_offset = (lhs-t0)/T
        arrays['phase_left_01'] = np.mod(phase_r-left_offset, 1.)
    else:
        arrays['phase_left_01'] = np.full(time.shape, np.nan)
    arrays['phase_left_rad'] = arrays['phase_left_01'] * (2*np.pi)
    arrays['left_hs_s'] = np.float64(lhs)
    arrays['left_to_s'] = np.float64(lto)
    arrays['right_to_s'] = np.float64(rto)
    arrays['left_to_phase_in_right_stride'] = np.float64((lto-t0)/T if np.isfinite(lto) else np.nan)
    arrays['right_to_phase_in_right_stride'] = np.float64((rto-t0)/T if np.isfinite(rto) else np.nan)
    arrays['left_to_phase_in_left_stride'] = np.float64(((lto-lhs)/T) % 1 if np.isfinite(lto) and np.isfinite(lhs) else np.nan)
    arrays['left_hs_source'] = np.str_('observed_20N_crossing' if np.isfinite(lhs) else 'not_observed')
    arrays['left_to_source'] = np.str_('observed_20N_crossing' if np.isfinite(lto) else 'not_observed')
    arrays['right_to_source'] = np.str_('observed_20N_crossing' if np.isfinite(rto) else 'not_observed')
    # Check no kinematics, original GRF, timestamps or markers are modified.
    with np.load(src, allow_pickle=False) as orig:
        for field in ('qpos', 'qvel', 'time_s', 'grf_time_s', 'grf_values', 'marker_xyz_os_m'):
            if not np.array_equal(arrays[field], orig[field]):
                raise AssertionError(f'{field} changed unexpectedly')
    counts=Counter((e['foot_side'], e['event']) for e in events)
    rel=src.relative_to(input_root)
    dest=output_root/rel
    manifest_src=src.with_name(src.stem+'_manifest.json')
    manifest=json.loads(manifest_src.read_text(encoding='utf8')) if manifest_src.exists() else {}
    old_events=manifest.pop('events', None)
    manifest.pop('event_finalization', None)
    manifest.pop('two_step_event_fix', None)
    manifest.pop('event_refractory', None)
    manifest['events']={
        'method':'Boo et al 2025 (right foot): rising 20N vertical-GRF crossing=HS; falling=TO',
        'stride_boundary':'Published right-HS-to-right-HS segmentation (not necessarily visible in cropped GRF)',
        'stride_start_right_hs_s':t0, 'stride_end_right_hs_s':t1,
        'right_hs_boundary_sources':[
            'observed_20N_crossing_near_boundary' if any(e['foot_side']=='right' and e['event']=='HS_candidate'
                            and abs(float(e['time_s'])-tb)<=.03 for e in events)
            else 'published_right_HS_segmentation' for tb in (t0,t1)],
        'left_hs_time_s':None if not np.isfinite(lhs) else lhs,
        'left_hs_source':'observed_20N_crossing' if np.isfinite(lhs) else 'not_observed',
        'left_hs_phase_in_right_stride':None if not np.isfinite(lhs) else (lhs-t0)/T,
        'left_to_time_s':None if not np.isfinite(lto) else lto,
        'right_to_time_s':None if not np.isfinite(rto) else rto,
        'left_to_source':'observed_20N_crossing' if np.isfinite(lto) else 'not_observed',
        'right_to_source':'observed_20N_crossing' if np.isfinite(rto) else 'not_observed',
        'observed_left_toe_off_s':[float(e['time_s']) for e in events if e['foot_side']=='left' and e['event']=='TO_candidate'],
        'observed_right_toe_off_s':[float(e['time_s']) for e in events if e['foot_side']=='right' and e['event']=='TO_candidate'],
    }
    manifest['contact_event_rebuild']={
        'status':'derived_from_npz_original_GRF_not_existing_event_CSV',
        'paper_definition':'right HS / TO at 20 N vertical GRF (level walking)',
        'additional_user_qc':f'{refractory_s*1000:g} ms per plate refractory; S029/S115 false-HS exclusions; CoP side classification',
        'no_estimated_events':True,
        'left_side_events':'extra inferred from CoP-to-foot-markers; NOT explicitly described by paper',
        'original_force_and_kinematics_unchanged':True,
        'supersedes_legacy_event_annotations':True,
    }
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    np.savez_compressed(dest, **arrays)
    with dest.with_name(dest.stem+'_events.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=EVENT_FIELDS);w.writeheader();w.writerows(events)
    dest.with_name(dest.stem+'_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf8')
    rej=Counter(r['reason'] for r in raw if r['status']=='rejected')
    summary={
        'reference':str(rel).replace('\\','/'), 'status':'OK',
        'left_HS_observed':int(np.isfinite(lhs)), 'left_TO_observed':int(np.isfinite(lto)),
        'right_TO_observed':int(np.isfinite(rto)),
        'left_HS_count':counts['left','HS_candidate'],
        'left_TO_count':counts['left','TO_candidate'],
        'right_HS_count':counts['right','HS_candidate'],
        'right_TO_count':counts['right','TO_candidate'],
        'unassigned_count':sum(e['foot_side']=='unassigned' for e in events),
        'refractory_rejections':rej['refractory'],
        'manual_QC_rejections':rej['manual_QC_false_left_HS'],
        'other_rejections':sum(v for k,v in rej.items() if k not in ('refractory','manual_QC_false_left_HS')),
        'warnings':';'.join([f'MULTIPLE_{s.upper()}_{k}' for (s,k),n in counts.items()
                           if s!='unassigned' and n>1]),
    }
    return summary, raw


def write_csv(path, rows, columns):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-root',required=True,type=Path)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--refractory-ms',type=float,default=20.)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if not args.input_root.is_dir():p.error('Input directory does not exist')
    if args.input_root.resolve()==args.output_root.resolve() or args.input_root.resolve() in args.output_root.resolve().parents:
        p.error('Output must be separate and outside input')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        p.error('Output folder must be new/empty (no overwriting)')
    if args.refractory_ms<0:p.error('--refractory-ms must be nonnegative')
    refs=[f for f in sorted(args.input_root.glob('S[0-9][0-9][0-9]/Trial*/step*.npz'))
          if re.fullmatch(r'step\d+\.npz',f.name)]
    if args.limit:refs=refs[:args.limit]
    if not refs:p.error('No stepXX.npz under input directory')
    print(f'Found {len(refs)} steps. Recomputing 20N crossings from original force arrays...')
    results, raw_all=[],[]
    for i,src in enumerate(refs,1):
        try:
            s, raw=save_one(src,args.input_root,args.output_root,args.refractory_ms/1000)
            results.append(s);raw_all.extend(raw)
        except Exception as ex:
            results.append({'reference':str(src.relative_to(args.input_root)).replace('\\','/'),
                            'status':'FAILED','warnings':f'{type(ex).__name__}: {ex}'})
        if i%50==0 or i==len(refs):print(f'[{i}/{len(refs)}] processed')
    args.output_root.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for row in results for k in row))
    write_csv(args.output_root/'paper_event_summary.csv',results,cols)
    write_csv(args.output_root/'paper_event_raw_crossings.csv',raw_all,RAW_FIELDS)
    counts=Counter(row['status'] for row in results)
    print(f'Finished: {counts["OK"]}/{len(refs)} successful, {counts["FAILED"]} failures')
    print('Missing measured events are stored as NaN, NOT estimated.')
    for key in ('left_HS_observed','left_TO_observed','right_TO_observed'):
        print(f'{key}: {sum(row.get(key,0) for row in results)}')
    print('QC rejection reasons:',dict(Counter(row['reason'] for row in raw_all if row['status']=='rejected')))
    print('Saved:',args.output_root/'paper_event_summary.csv')


if __name__=='__main__':
    main()
