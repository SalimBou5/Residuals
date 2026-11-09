# Replay loopable MyoLeg gait with events — plots displayed after playback

Adapted from your `play_gait120_with_events_corrected.py` for the **Step 2** output (`gait_with_landmarks`), which preserves Step 1 loopable kinematics and the original gait-event companion files.

## Install

```powershell
python -m pip install numpy mujoco matplotlib glfw
```

## Run in Windows PowerShell (single line)

```powershell
python replay_loopable_gait_with_events.py --reference "gait_with_landmarks\S012\Trial01\step01_loopable.npz" --model "C:\Users\BoussofaraS\Downloads\myo_sim\myo_sim\leg\myolegs.xml" --output gait_replay_plots --cycles 3 --speed 0.5 --plots-at-end
```

**With `--plots-at-end`, only the MuJoCo viewer runs during replay.** Matplotlib is not imported or updated until the viewer closes, so plot rendering cannot slow down playback. This is also the default mode. When the requested number of strides completes, the viewer closes automatically, and the joint-angle/angular-velocity and pelvis-relative landmark plots are created, saved, and displayed in Matplotlib windows. Closing the viewer early with Q (or the window close button) also shows the plots.

The PNG files are saved as `gait_replay_plots/step01_loopable_angles_velocities.png` and `gait_replay_plots/step01_loopable_sites_positions_velocities.png`. The first contains right/left **hip flexion, knee, and ankle** angles and angular velocities with gait-event phase lines; the second contains pelvis-relative hip, knee, ankle, and toe **x/y/z position and linear velocity** curves.

The saved `qvel` is plotted directly (joint generalized velocity, physical seconds). The model is **never stepped**; this is kinematic animation, not a dynamic walking test.

## Offline plotting (recommended)

Use `--plots-at-end` to save both PNGs and open them **after** the requested number of cycles finishes or after closing the viewer (press Q). If the MuJoCo animation itself still lags, add `--viewer-fps 30` to reduce viewer refresh overhead; this does not change the underlying kinematics.

```powershell
python replay_loopable_gait_with_events.py --reference "gait_with_landmarks\S012\Trial01\step01_loopable.npz" --model "C:\Users\BoussofaraS\Downloads\myo_sim\myo_sim\leg\myolegs.xml" --cycles 3 --plots-at-end --viewer-fps 60
```

To save figures without opening windows, also add `--no-show-plots`. To replay without generating any plots, use `--no-plots`. The script does not implement live-updated Matplotlib figures.

## Options

- `--cycles 3` runs three strides; `--cycles 0` repeats indefinitely (pelvis moves forward every cycle).
- `--speed 0.5` replay at half real time.
- `--hold 0.5` wall-time pause at selected gait events; `--hold 0` disables it.
- `--plots-at-end` saves and displays graphs only after replay (the default).
- `--viewer-fps 30` optionally limits viewer refresh rate to 30 Hz (default 60 Hz).
- `--no-plots` skips all plot generation.
- `--plots-only` saves graph PNGs without starting the viewer (and displays them unless `--no-show-plots` is set).
- `--no-show-plots` saves the PNG files without displaying them afterward.
- `--events-only` prints annotated events without needing a model.
- `--show-raw-crossings` adds raw, **unselected** GRF candidates for diagnostic display.
- Space pauses, R restarts, Left/Right steps by one reference sample, Q exits. Mouse orbits/zooms in the MuJoCo viewer.

## Event meaning

Reads `stepXX_loopable_manifest.json` to retrieve the source-selected left heel strike and left/right toe-off phases. Right HS is the stride start/end. The accompanying `stepXX_loopable_events.csv` is used to corroborate selected events and optionally display unselected diagnostic crossings; raw crossings are **not automatically treated as heel strikes**. Assumed toe-off phases remain labeled **assumed**, not measured. Colors: green = measured/corroborated, orange = segmented or assumed, pink = optional raw candidates. Spheres use `RCAL`/`LCAL` for HS and `RTOE`/`LTOE` for TO where these sites exist in the model. The event lines and console logs still work if these marker sites are absent.

## Important

- Use the **Step 2** NPZ, not the original Gait120 `step01.npz` or the earlier `build_myoleg_gait_library.py` output (different keys).
- No extra XML processing or regeneration of references is performed.
- `qpos[:2]` gains **one whole saved root XY displacement for each completed cycle**; interpolation within the stride retains the already-stored forward motion. The reference is not reset at the boundary.
- Matplotlib is not imported or updated during replay; figures are generated at the end. If `--cycles 0`, the plots appear only after you quit the viewer.
- This script has **not been executed**, in accordance with your preference. Please run it locally and send back screenshots or errors if anything looks wrong.
