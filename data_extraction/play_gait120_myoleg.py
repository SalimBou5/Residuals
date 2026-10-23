"""Visually replay an already-extracted Gait120 MyoLeg reference in MuJoCo.

No simulation steps, muscle actuation, retargeting, or file modification.
The exported qpos is copied into MyoLeg frame by frame, then mj_forward()
updates the displayed bodies. The camera follows the pelvis by default.

Example (Windows PowerShell):
    python play_gait120_myoleg.py --model "C:\\path\\to\\myolegs.xml" \
        --reference "S012_T01_reference.npz" --speed 0.5

Keys in the viewer: SPACE pause/resume; R restart; RIGHT/LEFT step one frame
while paused; Q quit. You can orbit/zoom using the mouse.

By default it loops. Looping a 1-second measured walking step makes the root
jump back to the beginning after the last sample: that's expected and is NOT
something you should interpret as part of the measured motion. Use --once to
pause at the last frame instead.
"""

import argparse
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, type=Path, help='Full path to MyoLeg model XML')
    ap.add_argument('--reference', required=True, type=Path, help='Exported *_reference.npz file')
    ap.add_argument('--speed', type=float, default=0.5, help='Playback speed (0.5 = half speed)')
    ap.add_argument('--once', action='store_true', help='Play once and pause at the end')
    ap.add_argument('--no-follow', action='store_true', help='Keep the camera fixed instead of following pelvis')
    args = ap.parse_args()

    if args.speed <= 0:
        ap.error('--speed must be positive')

    with np.load(args.reference, allow_pickle=False) as ref:
        t = np.array(ref['time_s'], dtype=np.float64)
        poses = np.array(ref['qpos'], dtype=np.float64)
        velocities = np.array(ref['qvel'], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    if poses.ndim != 2 or poses.shape[1] != model.nq:
        raise ValueError(f'qpos shape {poses.shape}, but this model expects (*, {model.nq})')
    if velocities.shape != (len(t), model.nv):
        raise ValueError(f'qvel shape {velocities.shape}, expected ({len(t)}, {model.nv})')
    if len(t) < 2 or not np.all(np.isfinite(poses)) or not np.all(np.isfinite(velocities)):
        raise ValueError('Reference data must contain at least two finite frames')
    if not np.all(np.diff(t) > 0):
        raise ValueError('Reference timestamps must be strictly increasing')
    if np.max(np.abs(np.linalg.norm(poses[:, 3:7], axis=1) - 1)) > 1e-4:
        raise ValueError('Root quaternion not normalized')

    # Timestamps in the dataset are absolute. Playback starts at t=0.
    rel_t = t - t[0]
    duration = float(rel_t[-1])
    sample_dt = float(np.median(np.diff(rel_t)))
    period = duration + sample_dt  # small slot to show the final sample when looping
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    if pelvis_id < 0:
        raise ValueError('The model does not contain the MyoLeg pelvis body')

    state = {'play_time': 0.0, 'paused': False, 'quit': False}

    def on_key(key):
        if key == glfw.KEY_SPACE:
            state['paused'] = not state['paused']
            print('Paused' if state['paused'] else 'Playing')
        elif key == glfw.KEY_R:
            state['play_time'] = 0.0
            state['paused'] = False
            print('Restarted')
        elif key == glfw.KEY_Q:
            state['quit'] = True
        elif key in (glfw.KEY_RIGHT, glfw.KEY_LEFT):
            state['paused'] = True
            delta = sample_dt if key == glfw.KEY_RIGHT else -sample_dt
            state['play_time'] = float(np.clip(state['play_time'] + delta, 0.0, duration))

    print(f'Reference: {len(t)} frames, {duration:.3f} s at ~{1/sample_dt:.1f} Hz')
    print(f'Model: nq={model.nq}, nv={model.nv}')
    print('Keys: SPACE pause/resume | R restart | LEFT/RIGHT single frame | Q quit')
    print('Note: static kinematic replay only; no physics stepping or control.')
    if not args.once:
        print('Looping: position will jump back to the first frame at each repeat.')

    data.qpos[:] = poses[0]
    data.qvel[:] = velocities[0]
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        # Initial view: roughly from the side. Mouse control remains available.
        with viewer.lock():
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.cam.distance = 2.5
            viewer.cam.elevation = -13
            viewer.cam.azimuth = 90

        last_wall = time.perf_counter()
        last_frame = -1
        while viewer.is_running() and not state['quit']:
            now = time.perf_counter()
            dt = now - last_wall
            last_wall = now

            if not state['paused']:
                state['play_time'] += args.speed * dt
                if args.once and state['play_time'] >= duration:
                    state['play_time'] = duration
                    state['paused'] = True
                    print('End of reference. Press R to replay, or LEFT to inspect frames.')
                elif not args.once:
                    state['play_time'] %= period

            frame = int(np.searchsorted(rel_t, min(state['play_time'], duration), side='right') - 1)
            frame = max(0, min(len(t) - 1, frame))

            if frame != last_frame:
                data.qpos[:] = poses[frame]      # incl. root + dependent knee joints
                data.qvel[:] = velocities[frame]
                data.time = float(rel_t[frame])
                mujoco.mj_forward(model, data)   # static FK only, NOT mj_step()
                last_frame = frame

            if not args.no_follow:
                with viewer.lock():
                    viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.sync()
            time.sleep(1 / 120)


if __name__ == '__main__':
    main()
