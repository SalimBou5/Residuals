"""Kinematic MyoLeg replay with explicit Gait120 heel-strike/toe-off markers.

Reads a batch-exported ``stepXX.npz``, its ``stepXX_events.csv``, and
``stepXX_manifest.json``. Also accepts older single-step
``*_reference.npz`` / ``*_reference_force_events.csv`` exports.

Green     = observed vertical-GRF 20 N crossing (HS candidate)
Magenta   = observed vertical-GRF 20 N falling crossing (TO candidate)
Amber     = event inferred from dataset segmentation or a half-cycle estimate

Events with unassigned foot sides are printed, but are NOT assigned to a foot.
Right HS at stride boundaries comes from dataset segmentation unless a matching
right-foot force event corroborates it. No unmeasured toe-offs are invented.

ONLY kinematic playback: set qpos/qvel, then mj_forward, never mj_step.

Windows PowerShell example:
  python play_gait120_with_events.py --model "C:\\...\\myolegs.xml" \
    --reference "C:\\...\\gait120_references_full\\S060\\Trial01\\step02.npz" \
    --speed 0.5 --hold 0.8

Run --events-only --reference FILE to audit the annotations without MuJoCo.
"""

import argparse
import csv
import json
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Event:
    time_s: float
    side: str  # 'left', 'right', or 'unassigned'
    kind: str  # 'HS' or 'TO'
    source: str  # 'observed', 'boundary', 'estimated'
    detail: str = ''

    @property
    def description(self):
        side = {'left': 'LEFT', 'right': 'RIGHT', 'unassigned': 'UNASSIGNED'}[self.side]
        evidence = {
            'observed': 'OBSERVED GRF 20N candidate',
            'boundary': 'DATASET stride boundary',
            'estimated': 'ESTIMATED (half stride)',
        }[self.source]
        return f'{side:>10} {self.kind:2}  [{evidence}] {self.detail}'.strip()


def read_event_csv(path):
    if not path.is_file():
        return []
    events = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            raw_kind = (row.get('event') or row.get('transition') or '').strip().lower()
            kind = ('HS' if raw_kind in ('hs_candidate', 'hs', 'on') else
                    'TO' if raw_kind in ('to_candidate', 'to', 'off') else None)
            if kind is None:
                print(f'WARNING: ignoring unrecognized event row {row!r}')
                continue
            side = (row.get('foot_side') or 'unassigned').lower().strip()
            if side not in ('left', 'right'):
                side = 'unassigned'
            detail = str(row.get('force_channel') or row.get('plate_channel') or '')
            events.append(Event(float(row['time_s']), side, kind, 'observed', detail))
    return events


def read_annotations(reference, t):
    """Gather actual 20N crossings and inference-only HS times separately."""
    base = reference.with_suffix('')
    csv_candidates = [Path(str(base) + '_events.csv'),
                      Path(str(base) + '_force_events.csv')]
    event_csv = next((p for p in csv_candidates if p.exists()), None)
    manifest_path = Path(str(base) + '_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    observed = read_event_csv(event_csv) if event_csv else []
    t0, t1 = float(t[0]), float(t[-1])
    T = t1 - t0
    if T <= 0:
        raise ValueError('Stride duration must be positive')

    # Keep all observed crossing records, including multiple crossings and
    # suspect early events; they are flagged as candidates, not certified gait events.
    accepted = []
    unassigned = []
    for event in observed:
        if not t0 - 0.003 <= event.time_s <= t1 + 0.003:
            print(f'WARNING: ignoring GRF event outside reference time range: {event}')
            continue
        if event.side == 'unassigned':
            unassigned.append(event)
        else:
            accepted.append(event)

    # Source labels for stride boundaries: measured only when a correctly
    # identified right-foot HS force crossing is within the tolerance.
    for time_point, label in ((t0, 'start'), (t1, 'end')):
        corroborating = [e for e in accepted if e.side == 'right' and e.kind == 'HS'
                         and abs(e.time_s - time_point) <= 0.03]
        if corroborating:
            # The true force crossing was already added: no duplicate boundary.
            # The event label remains 'OBSERVED' rather than 'boundary'.
            continue
        accepted.append(Event(time_point, 'right', 'HS', 'boundary', label))

    # Left HS estimate from the exporter: add only when source is assumed.
    event_meta = manifest.get('events', {})
    left_source = str(event_meta.get('left_hs_source', ''))
    if 'ASSUMED' in left_source.upper():
        left_time = float(event_meta.get('left_hs_time_s', t0 + 0.5*T))
        if t0 <= left_time <= t1:
            accepted.append(Event(left_time, 'left', 'HS', 'estimated'))
    elif not left_source and not any(e.kind == 'HS' and e.side == 'left' for e in accepted):
        # Supports older NPZs that have no explicit manifest phase provenance.
        accepted.append(Event(t0 + 0.5*T, 'left', 'HS', 'estimated', 'legacy file fallback'))

    accepted.sort(key=lambda e: (e.time_s, e.kind, e.side))
    unassigned.sort(key=lambda e: e.time_s)
    return accepted, unassigned, manifest, event_csv


def print_events(events, unassigned, t, reference, event_csv):
    t0, duration = float(t[0]), float(t[-1] - t[0])
    print(f'\nReference: {reference} | {len(t)} frames | T={duration:.3f} s')
    print(f'Event CSV: {event_csv or "NOT FOUND (only inferred boundaries)"}')
    print('Events (phase 0-100% of the RIGHT stride):')
    for e in events:
        p = (e.time_s - t0) / duration
        print(f'  {e.time_s:8.4f} s | {100*p:6.1f}% | {e.description}')
    if unassigned:
        print(f'\n  {len(unassigned)} observed event(s) with UNASSIGNED foot side are not drawn:')
        for e in unassigned:
            print(f'    {e.time_s:.4f} s | {e.description}')
    if not any(e.kind == 'TO' for e in events):
        print('\n  No foot-labeled, observed toe-off in this recording; none is invented.')
    print('\nWARNING: GRF crossings are candidates; multiple crossings may be artifacts.')


def add_event_sphere(viewer, mujoco, position, rgba):
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.065, 0, 0], dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3, dtype=float).reshape(-1),
        np.array(rgba, dtype=float),
    )
    scn.ngeom += 1


def highlight_events(viewer, model, data, events, at_t, hold_event, flash):
    import mujoco

    # Only geometries from the current instant. No spurious contact state inferred.
    viewer.user_scn.ngeom = 0
    for e in events:
        if e.side == 'unassigned':
            continue
        if abs(at_t - e.time_s) > flash and e is not hold_event:
            continue
        side = 'r' if e.side == 'right' else 'l'
        marker_name = (side.upper() + ('CAL' if e.kind == 'HS' else 'TOE'))
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, marker_name)
        if site_id < 0:
            # MyoLeg asset chain includes RCAL/LCAL and RTOE/LTOE.
            continue
        p = data.site_xpos[site_id].copy()
        p[2] += 0.12  # Float above the heel/toe, so the foot is still visible.
        rgba = ([0.07, 0.92, 0.40, 0.85] if e.source == 'observed' and e.kind == 'HS'
                else [0.95, 0.18, 0.50, 0.85] if e.source == 'observed'
                else [1.0, 0.65, 0.10, 0.85])
        add_event_sphere(viewer, mujoco, p, rgba)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, help='Full path to MyoLeg XML (required for viewer)')
    ap.add_argument('--reference', type=Path, required=True, help='Path to batch stepXX.npz')
    ap.add_argument('--speed', type=float, default=0.5, help='Playback speed, default 0.5x')
    ap.add_argument('--hold', type=float, default=0.7,
                    help='Seconds of wall-time pause at each event, default 0.7; use 0 to disable')
    ap.add_argument('--flash', type=float, default=0.06,
                    help='Visibility window (+/- seconds of dataset time) around events')
    ap.add_argument('--once', action='store_true', help='Pause at final pose instead of looping')
    ap.add_argument('--no-follow', action='store_true', help='Disable pelvis-following camera')
    ap.add_argument('--events-only', action='store_true', help='List event times without opening viewer')
    args = ap.parse_args()
    if args.speed <= 0 or args.hold < 0 or args.flash <= 0:
        ap.error('Require --speed > 0, --hold >= 0, --flash > 0')

    with np.load(args.reference, allow_pickle=False) as ref:
        t = np.asarray(ref['time_s'], dtype=float)
        qpos = np.asarray(ref['qpos'], dtype=float) if not args.events_only else None
        qvel = np.asarray(ref['qvel'], dtype=float) if not args.events_only else None

    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid reference timestamps')
    events, unassigned, manifest, event_csv = read_annotations(args.reference, t)
    print_events(events, unassigned, t, args.reference, event_csv)
    if args.events_only:
        return

    if args.model is None:
        ap.error('--model is required unless --events-only is specified')

    import glfw
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    if qpos.shape != (len(t), model.nq) or qvel.shape != (len(t), model.nv):
        raise ValueError(f'Incompatible qpos/qvel shapes: {qpos.shape}, {qvel.shape}; '
                         f'model nq/nv={model.nq}/{model.nv}')
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ValueError('Nonfinite positions or velocities')
    if np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-4:
        raise ValueError('Invalid root quaternion norms')

    t0, t1 = float(t[0]), float(t[-1]); duration = t1 - t0
    sample_dt = float(np.median(np.diff(t)))
    period = duration + sample_dt
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    if pelvis_id < 0:
        raise ValueError('Missing pelvis body in supplied MyoLeg model')

    # Values written by the key callback are consumed by the main viewer loop.
    state = {
        'play_time': 0.0, 'paused': False, 'quit': False,
        'hold_left': 0.0, 'hold_event': None, 'next_event': 0,
    }
    rel_events = [(e.time_s-t0, e) for e in events]

    def on_key(key):
        if key == glfw.KEY_SPACE:
            state['paused'] = not state['paused']
            print('\nPaused' if state['paused'] else '\nPlaying')
        elif key == glfw.KEY_R:
            state.update(play_time=0.0, paused=False, hold_left=0.0,
                         hold_event=None, next_event=0)
            print('\nRestart')
        elif key == glfw.KEY_Q:
            state['quit'] = True
        elif key in (glfw.KEY_RIGHT, glfw.KEY_LEFT):
            state['paused'] = True
            state['hold_left'] = 0.0
            state['hold_event'] = None
            idx = int(np.searchsorted(t-t0, min(state['play_time'], duration), side='left'))
            idx = max(0, min(len(t)-1, idx + (1 if key == glfw.KEY_RIGHT else -1)))
            state['play_time'] = float(t[idx]-t0)
            state['next_event'] = bisect_right([x[0] for x in rel_events], state['play_time'] + 1e-6)
            print(f'\nFrame {idx}, phase {100*state["play_time"]/duration:.1f}%')

    print('\nViewer: SPACE pause | R restart | LEFT/RIGHT frame | Q quit | mouse orbit/zoom')
    print('Colored spheres above heel/toe appear at events; default event hold 0.7 sec.')
    print('No physics stepping or muscle activation.')
    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -13
            viewer.cam.azimuth = 90

        last_wall = time.perf_counter()
        last_frame = -1
        while viewer.is_running() and not state['quit']:
            now = time.perf_counter()
            wall_dt = min(now-last_wall, 0.15)
            last_wall = now

            if not state['paused']:
                if state['hold_left'] > 0:
                    state['hold_left'] = max(0.0, state['hold_left'] - wall_dt)
                    if state['hold_left'] == 0:
                        state['hold_event'] = None
                else:
                    tentative = state['play_time'] + args.speed*wall_dt
                    if (state['next_event'] < len(rel_events) and
                            rel_events[state['next_event']][0] <= tentative + 1e-10):
                        ev_t, ev = rel_events[state['next_event']]
                        state['play_time'] = max(0.0, float(ev_t))
                        state['next_event'] += 1
                        # Skip any coincident events for the purpose of holding.
                        while (state['next_event'] < len(rel_events) and
                               abs(rel_events[state['next_event']][0] - ev_t) < 0.002):
                            more = rel_events[state['next_event']][1]
                            print(f'  EVENT: {more.description} at {more.time_s:.4f} s')
                            state['next_event'] += 1
                        print(f'\nEVENT: {ev.description} at {ev.time_s:.4f} s '
                              f'({100*state["play_time"]/duration:.1f}% stride)')
                        state['hold_event'] = ev if args.hold > 0 else None
                        state['hold_left'] = args.hold
                    else:
                        state['play_time'] = tentative

                    if state['play_time'] > duration:
                        if args.once:
                            state['play_time'] = duration
                            state['paused'] = True
                            state['hold_left'] = 0.0
                            print('\nEnd of stride. R to restart, LEFT to inspect.')
                        elif state['play_time'] >= period:
                            state['play_time'] = 0.0
                            state['next_event'] = 0
                            state['hold_left'] = 0.0
                            state['hold_event'] = None

            frame = int(np.searchsorted(t-t0, min(state['play_time'], duration), side='right')-1)
            frame = max(0, min(len(t)-1, frame))
            with viewer.lock():
                if frame != last_frame:
                    data.qpos[:] = qpos[frame]
                    data.qvel[:] = qvel[frame]
                    data.time = float(t[frame] - t0)
                    mujoco.mj_forward(model, data)
                    last_frame = frame
                if not args.no_follow:
                    viewer.cam.lookat[:] = data.xpos[pelvis_id]
                highlight_events(viewer, model, data, events,
                                 t0 + min(state['play_time'], duration),
                                 state['hold_event'], args.flash)
            viewer.sync()
            time.sleep(1.0/90.0)


if __name__ == '__main__':
    main()
