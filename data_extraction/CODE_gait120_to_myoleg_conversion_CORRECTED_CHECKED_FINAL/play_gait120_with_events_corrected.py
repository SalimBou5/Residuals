"""Kinematic MyoLeg replay with selected Gait120 gait annotations.

Reads a batch-exported stepXX.npz, its stepXX_events.csv and
stepXX_manifest.json. Also supports older *_reference files.

Default display:
  Green   = selected HS corroborated by a 20 N GRF candidate
  Amber   = HS defined by stride segmentation or explicitly estimated
  Magenta = raw GRF falling crossing (TO candidate, NOT validated)

Unlike the original viewer, unselected raw HS candidates are NEVER shown as
actual heel strikes. They remain in the untouched CSV and can optionally be
inspected with --show-raw-crossings (grey diagnostic markers).

Right HS boundaries follow the Gait120 right-HS-to-right-HS segmentation;
left HS comes from the exporter manifest / NPZ, including its ASSUMED label.
No toe-off times are invented. This is kinematic playback only: mj_forward,
never mj_step.

Example:
  python play_gait120_with_events_corrected.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --reference gait120_references_full/S029/Trial05/step01.npz

Audit labels without a MuJoCo viewer:
  python play_gait120_with_events_corrected.py --events-only --reference FILE
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
    side: str                 # left, right, unassigned
    kind: str                 # HS or TO
    source: str               # source codes documented in description below
    detail: str = ''

    @property
    def description(self):
        side = {'left': 'LEFT', 'right': 'RIGHT', 'unassigned': 'UNASSIGNED'}[self.side]
        evidence = {
            'selected_observed': 'SELECTED HS (20N GRF candidate)',
            'boundary_corroborated': 'DATASET stride boundary (GRF corroborated)',
            'boundary': 'DATASET stride boundary (not GRF-confirmed)',
            'estimated': 'ESTIMATED HS (not measured)',
            'annotated': 'SELECTED HS (no matching GRF candidate found)',
            'source_unknown': 'SAVED HS (measurement provenance unknown)',
            'observed': 'RAW 20N TO candidate (not validated)',
            'raw': 'UNSELECTED raw 20N HS candidate (diagnostic only)',
            'estimated_to': 'ASSUMED TO (not measured)',
            
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


def read_annotations(reference, t, left_hs_s=None, show_raw_crossings=False):
    """Return displayed annotations, leaving all original CSV observations intact.

    HS events are selected using stride boundaries and the exporter-selected
    left HS (manifest/NPZ). Other raw HS crossings are diagnostics only.
    All foot-assigned raw TO crossings are retained but explicitly *candidates*.
    """
    base = reference.with_suffix('')
    csv_candidates = [Path(str(base) + '_events.csv'),
                      Path(str(base) + '_force_events.csv')]
    event_csv = next((p for p in csv_candidates if p.is_file()), None)
    manifest_path = Path(str(base) + '_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    observed = read_event_csv(event_csv) if event_csv else []
    t0, t1 = float(t[0]), float(t[-1])
    duration = t1 - t0
    if duration <= 0:
        raise ValueError('Stride duration must be positive')

    valid, unassigned = [], []
    for e in observed:
        if not t0 - 0.003 <= e.time_s <= t1 + 0.003:
            print(f'WARNING: ignoring GRF crossing outside the stride: {e.description} at {e.time_s:.4f} s')
            continue
        if e.side == 'unassigned':
            unassigned.append(e)
        else:
            valid.append(e)

    # Toe-offs in CSV are *unvalidated* candidates, never selected/proven here.
    visible = [e for e in valid if e.kind == 'TO']
    used_hs = set()

    # Start/end right HS are fixed by Gait120 segmentation, not by the force
    # crossing timestamp. Mark whether a right-foot 20N candidate corroborates.
    for boundary_t, label in ((t0, 'start'), (t1, 'end')):
        matches = [e for e in valid if e.kind == 'HS' and e.side == 'right'
                   and abs(e.time_s - boundary_t) <= 0.03]
        if matches:
            closest = min(matches, key=lambda e: abs(e.time_s - boundary_t))
            used_hs.add(id(closest))
            source = 'boundary_corroborated'
        else:
            source = 'boundary'
        visible.append(Event(boundary_t, 'right', 'HS', source, label))

    # Use the selected left HS that was actually used to define phase_left_01.
    em = manifest.get('events', {})
    left_source = str(em.get('left_hs_source', '')).strip()
    manifest_time = em.get('left_hs_time_s')
    if manifest_time is not None and left_hs_s is not None:
        if abs(float(manifest_time) - float(left_hs_s)) > 0.001:
            raise ValueError('Manifest left_hs_time_s disagrees with NPZ left_hs_s; '
                             'check that the reference/manifest come from the same extraction')
    selected_t = (float(manifest_time) if manifest_time is not None else
                  float(left_hs_s) if left_hs_s is not None else
                  t0 + 0.5 * duration)
    if not t0 <= selected_t <= t1:
        raise ValueError(f'Selected left HS outside reference interval: {selected_t:.4f}')

    if left_source.upper().startswith('ASSUMED') or (not left_source and manifest_time is None and left_hs_s is None):
        visible.append(Event(selected_t, 'left', 'HS', 'estimated',
                             'half-stride fallback' if not left_source else left_source))
    elif left_source.lower().startswith('observed'):
        matches = [e for e in valid if e.kind == 'HS' and e.side == 'left'
                   and abs(e.time_s - selected_t) <= 0.005]
        if matches:
            closest = min(matches, key=lambda e: abs(e.time_s - selected_t))
            used_hs.add(id(closest))
            visible.append(Event(selected_t, 'left', 'HS', 'selected_observed', closest.detail))
        else:
            print('WARNING: left HS marked observed by exporter but no matching raw '
                  'left-HS crossing is present; showing selected time as uncorroborated')
            visible.append(Event(selected_t, 'left', 'HS', 'annotated', left_source))
    else:
        # Legacy/unknown provenance: do not falsely claim measured or estimated.
        visible.append(Event(selected_t, 'left', 'HS', 'source_unknown', left_source))

    # Display the assumed right TO only when the exporter used the 60% fallback.
    if em.get('right_toe_off_source') == 'ASSUMED_0.6_not_measured':
        right_to_phase = float(em['right_toe_off_phase'])
        visible.append(Event(t0 + right_to_phase * duration,
                             'right', 'TO', 'estimated_to',
                             '60% stride fallback'))

    # Display the assumed left TO only when the exporter used the 10% fallback.
    if em.get('left_toe_off_source') == 'ASSUMED_0.1_not_measured':
        left_to_phase = float(em['left_toe_off_phase'])
        visible.append(Event(t0 + left_to_phase * duration,
                             'left', 'TO', 'estimated_to',
                             '10% stride fallback'))
        
    # Unselected raw HS crossings remain intact in the input CSV. By default,
    # exclude them from both the event log and the viewer. Opt-in for diagnosis.
    rejected_hs = [e for e in valid if e.kind == 'HS' and id(e) not in used_hs]
    if rejected_hs:
        print(f'NOTE: {len(rejected_hs)} unselected raw HS candidate(s) hidden from replay '
              '(use --show-raw-crossings to inspect them).')
    if show_raw_crossings:
        visible.extend(Event(e.time_s, e.side, 'HS', 'raw', e.detail) for e in rejected_hs)

    visible.sort(key=lambda e: (e.time_s, e.kind, e.side))
    unassigned.sort(key=lambda e: e.time_s)
    return visible, unassigned, manifest, event_csv


def print_events(events, unassigned, t, reference, event_csv):
    t0, duration = float(t[0]), float(t[-1] - t[0])
    print(f'\nReference: {reference} | {len(t)} frames | T={duration:.3f} s')
    print(f'Event CSV: {event_csv or "NOT FOUND (only selected/inferred annotations)"}')
    print('Displayed events (phase 0-100% of the RIGHT stride):')
    for e in events:
        p = (e.time_s - t0) / duration
        print(f'  {e.time_s:8.4f} s | {100*p:6.1f}% | {e.description}')
    if unassigned:
        print(f'\n  {len(unassigned)} raw crossing(s) with UNASSIGNED foot side not drawn:')
        for e in unassigned:
            print(f'    {e.time_s:.4f} s | {e.description}')
    if not any(e.kind == 'TO' for e in events):
        print('\n  No foot-assigned toe-off candidate available; none is invented.')
    print('\nNOTE: Observed toe-offs are raw 20 N falling crossings, not certified '
          'anatomical toe-offs. Assumed right TO is explicitly marked.')


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
        rgba = ([0.50, 0.54, 0.60, 0.65] if e.source == 'raw'
                else [0.07, 0.92, 0.40, 0.85] if e.source in ('selected_observed', 'boundary_corroborated')
                else [1.0, 0.65, 0.10, 0.85] if e.source == 'estimated_to'
                else [0.95, 0.18, 0.50, 0.85] if e.kind == 'TO'
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
    ap.add_argument('--show-raw-crossings', action='store_true',
                    help='Diagnostic: show unselected raw HS crossings in grey (CSV never modified)')
    args = ap.parse_args()
    if args.speed <= 0 or args.hold < 0 or args.flash <= 0:
        ap.error('Require --speed > 0, --hold >= 0, --flash > 0')

    with np.load(args.reference, allow_pickle=False) as ref:
        t = np.asarray(ref['time_s'], dtype=float)
        qpos = np.asarray(ref['qpos'], dtype=float) if not args.events_only else None
        qvel = np.asarray(ref['qvel'], dtype=float) if not args.events_only else None
        left_hs_s = float(ref['left_hs_s']) if 'left_hs_s' in ref.files else None

    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError('Invalid reference timestamps')
    events, unassigned, manifest, event_csv = read_annotations(
        args.reference, t, left_hs_s=left_hs_s,
        show_raw_crossings=args.show_raw_crossings)
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
