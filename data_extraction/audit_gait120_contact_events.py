"""Read-only audit of Gait120 reference contact events.

Compares raw, saved 20 N GRF crossings with *_events.csv and the selected
left-HS annotation in *_manifest.json. Produces a per-stride CSV, an event CSV,
and diagnostic vertical-GRF plots. Does NOT change reference files.

Examples (PowerShell):
  python audit_gait120_contact_events.py --root gait120_references_forward_2
  python audit_gait120_contact_events.py --root gait120_references_forward_2 --max-plots 25
  python audit_gait120_contact_events.py --reference gait120_references_forward_2/S029/Trial05/step01.npz

Requirements: numpy; matplotlib for PNG plots. No MuJoCo required.

IMPORTANT: Force-plate coverage is incomplete in some files. Missing TO/HS
crossings CANNOT be equated with missing true foot contacts. Raw 20 N crossings
are candidates, not guaranteed biomechanical HS/TO. Threshold chatter checks
are QC heuristics, not automatic corrections.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re
import numpy as np


def crossings(t, f, threshold):
    above = f >= threshold
    out = []
    for i in np.flatnonzero(above[1:] != above[:-1]) + 1:
        t0, t1, y0, y1 = t[i-1], t[i], f[i-1], f[i]
        frac = (threshold-y0)/(y1-y0) if y1 != y0 else 1.
        out.append((float(t0 + np.clip(frac, 0., 1.)*(t1-t0)),
                    'HS' if above[i] else 'TO'))
    return out


def read_csv(path):
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        if cols:
            writer = csv.DictWriter(stream, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)


def get_kind(row):
    raw = str(row.get('event', row.get('transition', ''))).strip().lower()
    return {'hs_candidate': 'HS', 'on': 'HS', 'hs': 'HS',
            'to_candidate': 'TO', 'off': 'TO', 'to': 'TO'}.get(raw, 'UNKNOWN')


def get_plate(row):
    raw = str(row.get('plate', '') or row.get('force_channel', '') or row.get('plate_channel', ''))
    match = re.search(r'ground_force(\d+)', raw)
    if match:
        return int(match.group(1))
    if raw.isdigit():
        return int(raw)
    return None


def collect_reference(path, threshold, min_gap_s, min_bout_s):
    with np.load(path, allow_pickle=False) as a:
        t = np.asarray(a['time_s'], dtype=float)
        gt = np.asarray(a['grf_time_s'], dtype=float)
        gc = [str(x) for x in a['grf_channel_names']]
        gv = np.asarray(a['grf_values'], dtype=float)
        marker_names = [str(x) for x in a['marker_names']] if 'marker_names' in a else []
        marker_xyz = np.asarray(a['marker_xyz_os_m'], dtype=float) if 'marker_xyz_os_m' in a else None
        marker_time = np.asarray(a['marker_time_s'], dtype=float) if 'marker_time_s' in a else None
    if (len(t) < 2 or len(gt) < 2 or not np.all(np.diff(t)>0)
            or not np.all(np.diff(gt)>0)):
        raise ValueError('Invalid timestamps')
    t0, t1, T = float(t[0]), float(t[-1]), float(t[-1]-t[0])
    base = path.with_suffix('')
    evpath = next((p for p in [Path(str(base)+'_events.csv'),
                               Path(str(base)+'_force_events.csv')] if p.exists()), None)
    manpath = Path(str(base)+'_manifest.json')
    manifest = json.loads(manpath.read_text(encoding='utf-8')) if manpath.exists() else {}
    saved = read_csv(evpath) if evpath else []
    annotated = manifest.get('events', {})

    force, expected, flags = {}, [], []
    for plate in (1,2):
        key = f'ground_force{plate}_vy'
        if key not in gc:
            flags.append(f'NO_GRForce_plate{plate}')
            continue
        y = gv[:, gc.index(key)]
        force[plate] = y
        # Match the exporter: extract crossings from the FULL force array,
        # then retain only events within the reference interval +/- 2 ms.
        all_cr = crossings(gt, y, threshold)
        cr = [(when, kind) for when, kind in all_cr if t0-.002 <= when <= t1+.002]
        expected.extend({'time_s':when, 'kind':kind, 'plate':plate} for when,kind in cr)

        # Short gaps or spuriously brief contact bouts may produce extra HS/TO.
        # Restrict to complete bouts/gaps bracketed by observed transitions.
        for (ta, ka), (tb, kb) in zip(all_cr[:-1], all_cr[1:]):
            if tb < t0 or ta > t1:
                continue
            duration = tb-ta
            if ka == 'TO' and kb == 'HS' and duration < min_gap_s:
                flags.append(f'PLATE{plate}_SHORT_UNLOADED_GAP_{duration*1000:.0f}ms')
            elif ka == 'HS' and kb == 'TO' and duration < min_bout_s:
                flags.append(f'PLATE{plate}_SHORT_LOADED_BOUT_{duration*1000:.0f}ms')

    parsed = []
    for r in saved:
        try:
            when = float(r['time_s'])
        except (ValueError, TypeError, KeyError):
            flags.append('CSV_INVALID_EVENT_TIME'); continue
        kind, plate = get_kind(r), get_plate(r)
        side = str(r.get('foot_side') or 'unassigned').strip().lower()
        if side not in ('left','right'):
            side = 'unassigned'
        parsed.append({'time_s':when,'kind':kind,'plate':plate,'side':side,
                       'source':str(r.get('event_source','observed_20N_crossing'))})

    # Saved CSV must be reproducible from the source GRF trace. No assumption
    # about whether a force crossing is actually a biomechanical event.
    unmatched_saved=[]
    for r in parsed:
        if not any(e['plate']==r['plate'] and e['kind']==r['kind']
                   and abs(e['time_s']-r['time_s']) < .003 for e in expected):
            unmatched_saved.append(r)
    unmatched_raw=[]
    for e in expected:
        if not any(e['plate']==r['plate'] and e['kind']==r['kind']
                   and abs(e['time_s']-r['time_s']) < .003 for r in parsed):
            unmatched_raw.append(e)
    if unmatched_saved: flags.append(f'CSV_UNMATCHED_{len(unmatched_saved)}')
    if unmatched_raw: flags.append(f'GRF_CROSSINGS_NOT_IN_CSV_{len(unmatched_raw)}')

    labeled = [r for r in parsed if r['side'] != 'unassigned']
    counts = Counter((r['side'],r['kind']) for r in labeled)
    for side in ('left','right'):
        for kind in ('HS','TO'):
            if counts[(side,kind)]>1:
                flags.append(f'MULTIPLE_{side.upper()}_{kind}_{counts[(side,kind)]}')
    if any(r['side']=='unassigned' for r in parsed):
        flags.append('UNASSIGNED_FORCE_CROSSINGS')

    selected_source = str(annotated.get('left_hs_source',''))
    selected_t = annotated.get('left_hs_time_s')
    try:
        selected_t = float(selected_t) if selected_t is not None else None
    except (TypeError, ValueError):
        selected_t = None
    selected_phase = (selected_t-t0)/T if selected_t is not None else np.nan
    is_selected_measured = selected_source.startswith('observed_')
    if is_selected_measured:
        if not any(r['side']=='left' and r['kind']=='HS' and abs(r['time_s']-selected_t)<.03
                   for r in labeled):
            flags.append('MEASURED_LEFT_HS_NOT_IN_CSV')
        if not 0.30<=selected_phase<=0.70:
            flags.append('SELECTED_LEFT_HS_OUTSIDE_30_70pct')
    # Right-HS at the start/end are supplied by the segmentation, even without
    # actual force crossings in the clipped force samples.

    summary = {
        'reference':str(path), 'stride_s':round(T,5),
        'plate1_starts_loaded':bool(force.get(1,np.array([0]))[np.searchsorted(gt,t0,side='left')] >= threshold) if 1 in force and t0 <= gt[-1] else '',
        'plate2_starts_loaded':bool(force.get(2,np.array([0]))[np.searchsorted(gt,t0,side='left')] >= threshold) if 2 in force and t0 <= gt[-1] else '',
        'left_hs_source':selected_source or 'MISSING',
        'left_hs_selected_pct':round(100*selected_phase,2) if np.isfinite(selected_phase) else '',
        'labeled_left_HS':counts[('left','HS')], 'labeled_left_TO':counts[('left','TO')],
        'labeled_right_HS':counts[('right','HS')], 'labeled_right_TO':counts[('right','TO')],
        'unassigned_crossings':sum(r['side']=='unassigned' for r in parsed),
        'all_saved_crossings':len(parsed), 'raw_crossings_in_interval':len(expected),
        'warning_count':len(flags), 'warnings':';'.join(sorted(set(flags))),
    }
    rows=[]
    for r in parsed:
        rows.append({'reference':str(path), **r,
                     'phase_pct':round(100*(r['time_s']-t0)/T,2),
                     'selected_left_HS':bool(is_selected_measured and r['kind']=='HS' and r['side']=='left'
                                              and selected_t is not None and abs(r['time_s']-selected_t)<.03)})
    markers = {}
    if marker_xyz is not None and marker_time is not None:
        for name in ('LHEE','RHEE','LTOE','RTOE'):
            if name in marker_names:
                yy = marker_xyz[:, marker_names.index(name), 1]  # OpenSim Y-up
                markers[name] = 1000.0*(yy-np.min(yy))  # relative-to-minimum mm, not ground clearance
    return summary, rows, (gt,force,parsed,expected,t0,t1,selected_t,selected_source,marker_time,markers), flags


def make_plot(refpath, out, plotdata, threshold, min_gap_s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gt,force,parsed,expected,t0,t1,selected_t,selected_source,marker_time,markers = plotdata
    fig, axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
    for plate,ax in zip((1,2),axes):
        if plate not in force:
            ax.text(.5,.5,'Missing GRF channel',transform=ax.transAxes,ha='center')
            continue
        mask = (gt>=t0-.05)&(gt<=t1+.05)
        ax.plot(gt[mask]-t0,force[plate][mask],label=f'Plate {plate} vertical GRF',lw=1.25)
        ax.axhline(threshold,ls='--',lw=1,color='black',label=f'{threshold:g} N')
        for e in (x for x in parsed if x['plate']==plate):
            time = e['time_s']-t0
            color = 'green' if e['kind']=='HS' else 'magenta'
            ax.axvline(time,color=color,alpha=.55,lw=1.2)
            ax.text(time,.97 if e['kind']=='HS' else .81,
                    f"{e['kind']}:{e['side'][0].upper()}",color=color,
                    rotation=90,transform=ax.get_xaxis_transform(),
                    fontsize=7,va='top')
        # If QC removed a crossing from *_events.csv, it is still present in
        # the sampled GRF. Show it as grey instead of silently hiding it.
        for e in (e for e in expected if e['plate']==plate):
            if not any(r['plate']==plate and r['kind']==e['kind']
                       and abs(r['time_s']-e['time_s'])<.003 for r in parsed):
                ax.axvline(e['time_s']-t0,color='grey',ls=':',alpha=.7,lw=1.3)
                ax.text(e['time_s']-t0,.58,'raw only',rotation=90,
                        transform=ax.get_xaxis_transform(),va='top',fontsize=7,color='grey')
        ax.set_ylabel('Vertical force (N)')
        ax.grid(alpha=.2)
        ax.legend(loc='upper right',fontsize=8)
    if marker_time is not None and markers:
        for name,mm in markers.items():
            axes[2].plot(marker_time-t0,mm,label=name,lw=1.4)
        axes[2].legend(loc='upper right',fontsize=8,ncol=4)
    else:
        axes[2].text(.5,.5,'No marker trajectories available',
                     ha='center',transform=axes[2].transAxes)
    axes[2].set_ylabel('Marker height above its\nstride minimum (mm)')
    axes[2].grid(alpha=.2)
    if selected_t is not None:
        for ax in axes:
            ax.axvline(selected_t-t0,color='orange',ls=':',lw=1.8)
    for ax in axes:
        ax.axvline(0,color='grey',ls='--',lw=.8)
        ax.axvline(t1-t0,color='grey',ls='--',lw=.8)
    axes[-1].set_xlabel('Seconds from start of reference stride')
    fig.suptitle(f'{refpath.parent.parent.name}/{refpath.parent.name}/{refpath.stem} | '
                 f'Left HS selected: {selected_source or "unknown"}\n'
                 'Green=raw HS crossing, magenta=raw TO crossing, orange dotted=selected left HS\n'
                 'Events denote force crossings, NOT independently confirmed foot contacts',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.90))
    out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=145)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--root',type=Path,help='Directory containing S###/Trial##/step##.npz')
    g.add_argument('--reference',type=Path,help='One exported step##.npz')
    ap.add_argument('--output',type=Path,default=Path('gait120_event_audit'))
    ap.add_argument('--threshold',type=float,default=20.0)
    ap.add_argument('--min-gap-ms',type=float,default=60.0,
                    help='Flag plate unloaded gaps shorter than this (not a correction)')
    ap.add_argument('--min-bout-ms',type=float,default=60.0,
                    help='Flag plate loaded bouts shorter than this (not a correction)')
    ap.add_argument('--max-plots',type=int,default=15,
                    help='Maximum automatically generated plots, preferring flagged strides')
    args=ap.parse_args()
    if args.reference:
        paths = [args.reference]
    else:
        paths = sorted(args.root.glob('S[0-9][0-9][0-9]/Trial*/step*.npz'))
    if not paths:
        ap.error('No reference NPZ files found')
    summaries, all_events, plotted_candidates, failed = [], [], [], []
    for i,p in enumerate(paths,1):
        try:
            s, events, pdata, flags = collect_reference(p,args.threshold,
                args.min_gap_ms/1000,args.min_bout_ms/1000)
            summaries.append(s)
            all_events.extend(events)
            # sort by warnings, prefer duplicate events / short gaps
            severity = (10*sum('SHORT_' in f or 'MULTIPLE_' in f for f in flags)
                        + 3*sum('CSV_' in f or 'GRF_CROSSINGS_' in f for f in flags)
                        + len(flags))
            plotted_candidates.append((severity,p,pdata))
        except Exception as exc:
            failed.append({'reference':str(p),'error':f'{type(exc).__name__}: {exc}'})
    csv_write(args.output/'event_audit_summary.csv', summaries)
    csv_write(args.output/'event_audit_crossings.csv', all_events)
    csv_write(args.output/'event_audit_failures.csv', failed)
    chosen = sorted(plotted_candidates,key=lambda e:-e[0])[:max(0,args.max_plots)]
    for _,path,pdata in chosen:
        rel = Path(path.parent.parent.name)/path.parent.name/(path.stem+'.png')
        make_plot(path,args.output/'plots'/rel,pdata,args.threshold,args.min_gap_ms/1000)
    warnings = Counter()
    for r in summaries:
        for w in filter(None,r['warnings'].split(';')):
            warnings[w.split('_SHORT_')[0] if '_SHORT_' in w else w] += 1
    print(f'Audited {len(summaries)}/{len(paths)} references; read errors={len(failed)}.')
    print(f'Left HS estimated (not observed): {sum("ASSUMED" in s["left_hs_source"].upper() for s in summaries)}')
    print(f'Strides with no LABELED left TO: {sum(s["labeled_left_TO"]==0 for s in summaries)}')
    print(f'Strides with no LABELED right TO: {sum(s["labeled_right_TO"]==0 for s in summaries)}')
    print(f'Strides with repeated labeled HS or TO: '
          f'{sum(any(s[f"labeled_{side}_{kind}"]>1 for side in ("left","right") for kind in ("HS","TO")) for s in summaries)}')
    print('IMPORTANT: a missing force-plate event is not proof of a missing actual foot event.')
    print('Most frequent warnings:',warnings.most_common(10))
    print('Reports:',args.output.resolve())
    print(f'Created {len(chosen)} GRF diagnostic plot(s). No input files were modified.')


if __name__=='__main__':
    main()
