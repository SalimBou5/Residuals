#!/usr/bin/env python3
"""Finalize Gait120 reference foot events WITHOUT modifying source references.

Input:  heading-aligned gait120_references_forward/S###/Trial##/step##.npz,
        corresponding *_events.csv and *_manifest.json.
Output: NEW reference tree containing identical kinematics/GRFs, updated gait
        phase metadata, one selected HS/TO per foot per stride, explicit source
        (observed vs estimated), and *_selected_events.csv for replay/training.

Why estimates? The two force plates do not capture all foot-contact transitions
in every stride. In particular many step01s show HS but lack both TO crossings.
A missing GRF threshold crossing does NOT demonstrate a missing biomechanical TO.

The 20 N crossings are kept in *_events.csv for traceability. A per-plate 20 ms
refractory filter removes threshold chatter from candidate selection only.
Selected timing and provenance live in *_selected_events.csv and the manifest.
Right HS stride boundaries are the dataset segmentation, not independently
confirmed GRF HS unless a matching right-side crossing is found close by.

This is a KINEMATIC reference dataset with event annotations, not evidence that
force/muscle dynamics of MyoLeg reproduce the human recordings.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

LEFT_HS_WINDOW = (0.30, 0.70)
LEFT_TO_WINDOW = (0.03, 0.25)
RIGHT_TO_WINDOW = (0.45, 0.80)
REFERENCE_PHASES = {'left_TO': 0.10, 'right_TO': 0.605}
EVENT_COLUMNS = ["time_s", "time_since_start_s", "normalized_phase", "foot_side", "event", "event_source", "plate", "force_channel", "selection_note"]


def csv_rows(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def kind_of(row):
    v = str(row.get('event', '')).strip().lower()
    return {'hs_candidate': 'HS', 'hs': 'HS', 'on': 'HS',
            'to_candidate': 'TO', 'to': 'TO', 'off': 'TO'}.get(v)


def plate_of(row):
    v = str(row.get('plate', '') or row.get('force_channel', '')).strip()
    m = re.search(r'(?:ground_force)?(\d+)', v)
    return m.group(1) if m else v


def clean_candidates(rows, refractory_s, t0, t1):
    """Per-plate refractory on the saved measured threshold crossings.

    Earlier manual corrections may already have removed two false HS from
    *_events.csv. Never reconstruct them from the retained raw GRF.
    """
    valid = []
    for r in rows:
        kind = kind_of(r)
        if not kind:
            continue
        try:
            t = float(r['time_s'])
        except (KeyError, ValueError, TypeError):
            raise ValueError(f'Invalid event time: {r}')
        if not math.isfinite(t) or t < t0 - .003 or t > t1 + .003:
            continue
        side = str(r.get('foot_side', 'unassigned')).strip().lower()
        valid.append({'time_s':t, 'kind':kind, 'side':side,
                      'plate':plate_of(r), 'force_channel':str(r.get('force_channel', '')),
                      'row':r})
    last = {}
    kept, suppressed = [], []
    for r in sorted(valid, key=lambda x:(x['time_s'], x['plate'])):
        p = r['plate']
        if p in last and r['time_s'] - last[p] < refractory_s - 1e-10:
            suppressed.append(r)
        else:
            kept.append(r)
            last[p] = r['time_s']
    return kept, suppressed


def within(c, t0, T, window):
    p = (c['time_s'] - t0)/T
    return window[0] <= p <= window[1]


def select_measured_to(candidates, side, center, window, t0, T):
    matches = [r for r in candidates if r['side'] == side and r['kind'] == 'TO'
               and within(r, t0, T, window)]
    return (min(matches, key=lambda r:abs((r['time_s']-t0)/T-center)) if matches else None), len(matches)


def selected_row(t0, T, side, kind, phase, source, plate='', channel='', note=''):
    time = t0 + T*phase
    return {'time_s':f'{time:.12f}', 'time_since_start_s':f'{T*phase:.12f}',
            'normalized_phase':f'{phase:.12f}', 'foot_side':side, 'event':kind,
            'event_source':source, 'plate':plate, 'force_channel':channel,
            'selection_note':note}


def collect_metadata(path: Path, refractory_s):
    base = path.with_suffix('')
    manifest_file = Path(str(base) + '_manifest.json')
    event_file = Path(str(base) + '_events.csv')
    if not manifest_file.is_file() or not event_file.is_file():
        raise FileNotFoundError(f'Missing manifest/events for {path}')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    # Read only time array in the first pass; no need to materialize all frames.
    with np.load(path, allow_pickle=False) as npz:
        ts = np.asarray(npz['time_s'], dtype=float)
        if not {'qpos','qvel','phase_right_01','phase_left_01','left_hs_s'}.issubset(npz.files):
            raise ValueError(f'Missing required NPZ fields in {path}')
    if len(ts) < 2 or not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0):
        raise ValueError(f'Invalid times in {path}')
    t0, t1 = float(ts[0]), float(ts[-1])
    T = t1 - t0
    candidates, suppressed = clean_candidates(csv_rows(event_file), refractory_s, t0, t1)
    if not T > 0:
        raise ValueError(f'Invalid duration in {path}')
    return {'path':path, 'manifest':manifest, 't0':t0, 't1':t1, 'T':T,
            'candidates':candidates, 'suppressed':suppressed}


def estimate_templates(infos):
    """Use reliable observations to derive dataset-specific fallback TO phases."""
    phases = {'left_TO': [], 'right_TO': []}
    for info in infos:
        t0, T = info['t0'], info['T']
        for side, key, window in [('left', 'left_TO', LEFT_TO_WINDOW),
                                  ('right', 'right_TO', RIGHT_TO_WINDOW)]:
            choice, _ = select_measured_to(info['candidates'],side,REFERENCE_PHASES[key],window,t0,T)
            if choice is not None:
                phases[key].append((choice['time_s']-t0)/T)
    centers={}
    for key in phases:
        # Full dataset uses its own medians. Very small pilots explicitly
        # use prior audit medians rather than an unstable sample estimate.
        if len(phases[key]) < 10:
            print(f'WARNING: only {len(phases[key])} measured {key} candidate(s); '
                  f'using prior-audit default {REFERENCE_PHASES[key]:.3f}.')
            centers[key]=REFERENCE_PHASES[key]
        else:
            centers[key]=float(np.median(phases[key]))
    return centers, {k:len(v) for k,v in phases.items()}


def choose_annotations(info, centers):
    t0, T, cand = info['t0'], info['T'], info['candidates']
    em = info['manifest'].get('events',{})
    source = str(em.get('left_hs_source',''))
    left_hs = None
    # Honor existing QC/manual decisions: only treat the preselected measured
    # LHS as reliable when present in saved data and in plausible phase range.
    if source.lower().startswith('observed'):
        try:
            previously_selected = float(em['left_hs_time_s'])
        except (KeyError, ValueError, TypeError):
            previously_selected = math.nan
        matching = [c for c in cand if c['side']=='left' and c['kind']=='HS'
                    and within(c,t0,T,LEFT_HS_WINDOW)
                    and abs(c['time_s']-previously_selected) <= .003]
        if matching:
            left_hs=min(matching,key=lambda c:abs(c['time_s']-previously_selected))
    lhs_phase = (left_hs['time_s']-t0)/T if left_hs else .5

    lto, lto_choices = select_measured_to(cand,'left',centers['left_TO'],LEFT_TO_WINDOW,t0,T)
    rto, rto_choices = select_measured_to(cand,'right',centers['right_TO'],RIGHT_TO_WINDOW,t0,T)
    warnings=[]
    if lto_choices > 1:
        warnings.append(f'LEFT_TO_MULTIPLE_PLAUSIBLE_{lto_choices}')
    if rto_choices > 1:
        warnings.append(f'RIGHT_TO_MULTIPLE_PLAUSIBLE_{rto_choices}')
    if not left_hs and source.lower().startswith('observed'):
        warnings.append('PRIOR_LEFT_HS_MEASUREMENT_REJECTED_UNVERIFIED')

    # Timing consistency across one nominal right-HS-to-right-HS cycle.
    # When violated, do not mislabel a crossing as a real TO; use a template.
    if lto is not None and not (0 < (lto['time_s']-t0)/T < lhs_phase):
        lto=None
        warnings.append('LEFT_TO_ORDER_REJECTED')
    if rto is not None and not (lhs_phase < (rto['time_s']-t0)/T < 1):
        rto=None
        warnings.append('RIGHT_TO_ORDER_REJECTED')
    if not (0 < centers['left_TO'] < lhs_phase < centers['right_TO'] < 1):
        raise ValueError('Inconsistent inferred events or templates')

    rows=[]
    for phase, location in ((0.0,'start'),(1.0,'end')):
        tb=t0+T*phase
        nearby=[c for c in cand if c['side']=='right' and c['kind']=='HS'
                and abs(c['time_s']-tb)<=.03]
        label='dataset_right_HS_boundary_GRF_corrob' if nearby else 'dataset_right_HS_boundary_inferred'
        rows.append(selected_row(t0,T,'right','HS',phase,label,
                                note=f'{location} of right-HS-to-right-HS segmented stride'))
    if left_hs:
        rows.append(selected_row(t0,T,'left','HS',lhs_phase,'observed_20N_crossing',
                                left_hs['plate'],left_hs['force_channel'],'selected from manifest and observed crossing'))
    else:
        rows.append(selected_row(t0,T,'left','HS',lhs_phase,'estimated_half_stride',
                                note='not observed; exactly half the right stride'))

    for side, key, match in [('left','left_TO',lto),('right','right_TO',rto)]:
        phase = (match['time_s']-t0)/T if match else centers[key]
        label='observed_20N_crossing' if match else 'estimated_cohort_median'
        rows.append(selected_row(t0,T,side,'TO',phase,label,
                                match['plate'] if match else '',
                                match['force_channel'] if match else '',
                                'threshold crossing candidate' if match else
                                'no credible measured crossing; median of measured strides'))
    rows.sort(key=lambda row:(float(row['time_s']),row['foot_side'],row['event']))
    assert len(rows)==5
    return rows, warnings


def validate_selected(rows, t0, t1):
    by={(r['foot_side'],r['event']):[] for r in rows}
    for r in rows:
        if not np.isfinite(float(r['time_s'])) or not t0-1e-8 <= float(r['time_s']) <= t1+1e-8:
            raise ValueError(f'Event outside stride: {r}')
        by[(r['foot_side'],r['event'])].append(r)
    if not (len(by['right','HS'])==2 and len(by['left','HS'])==1
            and len(by['left','TO'])==1 and len(by['right','TO'])==1):
        raise ValueError('Missing/duplicate selected HS/TO')
    t = {k:float(v[0]['time_s']) for k,v in by.items() if k!=('right','HS')}
    if not (t0 < t['left','TO'] < t['left','HS'] < t['right','TO'] < t1):
        raise ValueError('Incorrect gait-event temporal ordering')


def write_csv(path:Path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def finalize_one(info, centers, input_root, output_root):
    src=info['path']
    dest=output_root/src.relative_to(input_root)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Destination reference exists: {dest}; choose an empty --output-root')
    rows, warnings=choose_annotations(info,centers)
    validate_selected(rows,info['t0'],info['t1'])
    by={(r['foot_side'],r['event']):r for r in rows if not (r['foot_side']=='right' and r['event']=='HS')}
    lhs=by['left','HS']; lto=by['left','TO']; rto=by['right','TO']
    t0,T=info['t0'],info['T']
    with np.load(src,allow_pickle=False) as npz:
        arrays={k:np.array(npz[k]) for k in npz.files}
    time=np.asarray(arrays['time_s'],float)
    phase_right=(time-t0)/T
    original_qpos=arrays['qpos'].copy()
    original_qvel=arrays['qvel'].copy()
    left_offset=float(lhs['normalized_phase'])
    arrays['phase_right_01']=phase_right
    arrays['phase_right_rad']=2*np.pi*phase_right
    arrays['phase_left_01']=np.mod(phase_right-left_offset,1.0)
    arrays['phase_left_rad']=2*np.pi*arrays['phase_left_01']
    arrays['left_hs_s']=np.float64(lhs['time_s'])
    arrays['left_to_s']=np.float64(lto['time_s'])
    arrays['right_to_s']=np.float64(rto['time_s'])
    arrays['left_to_phase_in_right_stride']=np.float64(lto['normalized_phase'])
    arrays['right_to_phase_in_right_stride']=np.float64(rto['normalized_phase'])
    arrays['left_hs_source']=np.str_(lhs['event_source'])
    arrays['left_to_source']=np.str_(lto['event_source'])
    arrays['right_to_source']=np.str_(rto['event_source'])
    # Important: phase in the current leg cycle starts at that leg's own HS.
    arrays['left_to_phase_in_left_stride']=np.float64(
        (float(lto['normalized_phase'])-left_offset)%1.0)
    assert np.array_equal(arrays['qpos'],original_qpos)
    assert np.array_equal(arrays['qvel'],original_qvel)

    manifest=info['manifest'].copy()
    events=manifest.setdefault('events',{})
    original_events=dict(events)
    events['left_hs_time_s']=float(lhs['time_s'])
    events['left_hs_phase_in_right_stride']=left_offset
    events['left_hs_source']=lhs['event_source']
    events['left_to_time_s']=float(lto['time_s'])
    events['right_to_time_s']=float(rto['time_s'])
    events['left_to_phase_in_right_stride']=float(lto['normalized_phase'])
    events['right_to_phase_in_right_stride']=float(rto['normalized_phase'])
    events['left_to_phase_in_left_stride']=float(arrays['left_to_phase_in_left_stride'])
    events['left_to_source']=lto['event_source']
    events['right_to_source']=rto['event_source']
    # Legacy observed lists must not retain rejected duplicate crossings.
    # Preserve their previous contents under event_finalization below.
    events['observed_left_toe_off_s']=(
        [float(lto['time_s'])] if lto['event_source'].startswith('observed') else [])
    events['observed_right_toe_off_s']=(
        [float(rto['time_s'])] if rto['event_source'].startswith('observed') else [])
    events['right_hs_boundary_sources']=[
        r['event_source'] for r in rows if r['event']=='HS' and r['foot_side']=='right']
    # Previous raw-observation entries are preserved unchanged, but do not
    # interpret their whole candidate list as the curated selected events.
    manifest['event_finalization']={
        'version':'1.0', 'method':'20 N observed GRF crossings, per-plate 20ms refractory, phase/QC gating, estimate missing TO from reliable cohort medians',
        'raw_events_csv':'*_events.csv (copied unchanged; observations/candidates, not final events)',
        'selected_events_csv':'*_selected_events.csv (one left HS, one left TO, one right TO, two right HS boundaries)',
        'reference_event_phases_right_stride':centers,
        'original_selected_event_metadata_before_finalization':original_events,
        'suppressed_saved_crossing_count':len(info['suppressed']),
        'warnings':warnings,
        'phase_arrays_updated_for_left_HS':True,
        'qpos_qvel_original_grf_unchanged':True,
        'estimates_are_not_measured':True,
    }
    # Copy raw candidate events, so user can always inspect original evidence.
    src_events=src.with_name(src.stem+'_events.csv')
    dst_events=dest.with_name(dest.stem+'_events.csv')
    target_manifest=dest.with_name(dest.stem+'_manifest.json')
    target_selected=dest.with_name(dest.stem+'_selected_events.csv')
    # Stage all outputs before final rename. Output root is required empty.
    tmp_npz=dest.with_name(dest.stem+'.events_tmp.npz')
    try:
        np.savez_compressed(tmp_npz,**arrays)
        with np.load(tmp_npz,allow_pickle=False) as audit:
            for k in ('qpos','qvel','grf_time_s','grf_values'):
                if k in arrays and not np.array_equal(audit[k],arrays[k]):
                    raise AssertionError(f'{k} changed while saving {dest}')
            if not np.isfinite(audit['left_to_s']).all():
                raise AssertionError('Invalid TO array')
        write_csv(target_selected,rows,EVENT_COLUMNS)
        shutil.copy2(src_events,dst_events)
        target_manifest.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        tmp_npz.replace(dest)
    finally:
        tmp_npz.unlink(missing_ok=True)

    return {'reference':str(src.relative_to(input_root)).replace('\\','/'),
            'status':'OK', 'left_HS':lhs['event_source'],
            'left_TO':lto['event_source'], 'right_TO':rto['event_source'],
            'left_HS_phase_pct':round(100*left_offset,3),
            'left_TO_phase_pct':round(100*float(lto['normalized_phase']),3),
            'right_TO_phase_pct':round(100*float(rto['normalized_phase']),3),
            'refractory_suppressed':len(info['suppressed']),
            'warnings':';'.join(warnings)}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--refractory-ms',type=float,default=20.0)
    p.add_argument('--limit',type=int,default=0,help='Pilot: process first N subjects/steps (0=all)')
    args=p.parse_args()
    if not args.input_root.is_dir():
        p.error(f'Input root does not exist: {args.input_root}')
    if args.input_root.resolve()==args.output_root.resolve() or args.input_root.resolve() in args.output_root.resolve().parents:
        p.error('Output directory must be separate, not inside the input directory')
    if args.output_root.exists() and any(args.output_root.iterdir()):
        p.error(f'Output directory must be new/empty: {args.output_root}')
    if args.refractory_ms < 0:
        p.error('refractory-ms must be nonnegative')
    paths=sorted(x for x in args.input_root.glob('S[0-9][0-9][0-9]/Trial*/step*.npz')
                 if re.fullmatch(r'step\d+\.npz',x.name))
    if args.limit:
        paths=paths[:args.limit]
    if not paths:
        p.error('No step##.npz references discovered')
    infos=[]; errors=[]
    for path in paths:
        try:
            infos.append(collect_metadata(path,args.refractory_ms/1000.0))
        except Exception as e:
            errors.append((path,f'{type(e).__name__}: {e}'))
    if errors:
        for path,error in errors[:15]:
            print('INPUT ERROR:',path,error)
        raise SystemExit(f'Refusing incomplete dataset: {len(errors)} input errors. No files written.')
    centers,counts=estimate_templates(infos)
    print(f'Discovered {len(infos)} strides | credible measured TO available: {counts}')
    print('Fallback phases in right stride:',{k:round(v,5) for k,v in centers.items()})
    args.output_root.mkdir(parents=True,exist_ok=True)
    results=[]
    for i,info in enumerate(infos,1):
        try:
            results.append(finalize_one(info,centers,args.input_root,args.output_root))
        except Exception as e:
            results.append({'reference':str(info['path'].relative_to(args.input_root)),
                            'status':'FAILED', 'warnings':f'{type(e).__name__}: {e}'})
        if i%50==0 or i==len(infos):
            print(f'[{i}/{len(infos)}] processed')
    summary=args.output_root/'event_finalization_summary.csv'
    fields=list(dict.fromkeys(k for row in results for k in row))
    write_csv(summary,results,fields)
    counts=Counter(x['status'] for x in results)
    print(f'Finalized: {counts["OK"]}/{len(results)}; failed: {counts["FAILED"]}')
    print('Measured/estimated counts:')
    for field in ('left_HS','left_TO','right_TO'):
        print('  ',field,dict(Counter(row.get(field,'') for row in results if row['status']=='OK')))
    print(f'Output directory: {args.output_root}\nReport: {summary}')
    if counts['FAILED']:
        raise SystemExit('Some conversions failed: consult the summary. Keep your input dataset unchanged.')


if __name__=='__main__':
    main()
