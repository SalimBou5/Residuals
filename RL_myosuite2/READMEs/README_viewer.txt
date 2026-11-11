MyoLeg Step 6: Watch trained MEP+PPO gait in MuJoCo
=================================================

This is actual 1000 Hz MuJoCo muscle-driven physics controlled at 200 Hz.
The recorded reference positions and velocities are used to initialize each
run ONCE, at reset, not imposed during dynamic simulation.

Place view_ppo_myoleg.py and evaluation_resets_10m.json at the root of your
RL_myosuite2 repository, alongside evaluate_ppo.py and train_ppo_wandb.py.

Inside your activated myosuite conda environment, PowerShell:

1) Show a specific, reproducible episode from your 16-episode evaluation:

python view_ppo_myoleg.py `
  --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --speed 0.5 --stay-open

Episode 7 was 1.59 s in the 16-episode evaluation. The viewer pauses at the
reset for 1.5 seconds, and after termination for 4 seconds, or indefinitely
with --stay-open. Drag the view with the mouse to inspect gait and failures.

2) Optional MP4 export IN ADDITION TO opening the interactive MuJoCo viewer:

pip install imageio imageio-ffmpeg
python view_ppo_myoleg.py `
  --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --speed 0.5 `
  --record "runs\ppo_viewer\episode_07_full.mp4"

If your GPU/graphics stack cannot initialize a separate offscreen renderer,
run without --record to use the interactive MuJoCo viewer. The GUI viewer is
the primary feature. The MP4 recorder has not been runtime-tested here.

3) Same recorded reset, remove all learned residuals (MEP-only control):

python view_ppo_myoleg.py --summary "evaluation_resets_10m.json" `
  --episode 7 --mode mep --speed 0.5

4) Same checkpoint but mask the phase residual:

python view_ppo_myoleg.py --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --mode muscle-only --speed 0.5

5) Same checkpoint but mask the muscle residuals:

python view_ppo_myoleg.py --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --mode phase-only --speed 0.5

All four modes use the same previously trained PPO model; no retraining.
But masking an output at evaluation changes the state distribution. For a
fully controlled architecture ablation, separately train restricted policies
later if needed.

The live terminal prints every 0.2 simulated seconds: site error, actual and
desired speed, upright score, phase, accepted/rejected contact events, and
final termination reason. Per-step CSV and summary JSON are saved under
runs/ppo_viewer/.

Use --episode 0, 1, ... 15 to select different resets from the supplied
16-episode report. Set --episodes 16 to watch all, sequentially.

To use your own settings and random resets, omit --summary and provide
--model, --references, --mep, --lookup, --normalization, --reward-config;
use --help to see all options.

Checks performed: Python syntax compilation, command line parsing and
static integration with provided Step-6 source files. The native MuJoCo
viewer and MP4 encoder require your local graphics runtime, not present in
this artifact container.


New plotting options
--------------------
The viewer can now also visualize:
- applied muscle excitations (`data.ctrl`) overlaid on the faded MEP baseline (`last_baseline`)
- actual nonroot joint positions overlaid on the faded reference joint positions
- phase diagnostics (`phase`, `theta`, `phase_drift`, `phase_rate_hz`)

Two usage styles are supported:

A) Save figures offline after the episode (default):

python view_ppo_myoleg.py `
  --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --speed 0.5 --plot-mode save

This writes five PNGs next to the per-step CSV/JSON:
- `..._excitations_right.png`
- `..._excitations_left.png`
- `..._kinematics_right.png`
- `..._kinematics_left.png`
- `..._phase.png`

B) Show plots live while MuJoCo is running:

python view_ppo_myoleg.py `
  --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --speed 0.5 --plot-mode live

Or do both:

python view_ppo_myoleg.py `
  --summary "evaluation_resets_10m.json" `
  --checkpoint "runs\ppo_myoleg\ppo_20261108_160227_seed42\final_model.zip" `
  --episode 7 --speed 0.5 --plot-mode both

Useful knobs:
- `--plot-update-every 10` updates the live plots every 10 control steps
- `--muscles-per-panel 3` gives at most three muscles per saved subplot
  (use `--muscles-per-panel 2` to spread the muscles across even more panels)
- `--plot-max-muscles-per-side 12` limits the number of plotted muscles per leg
- `--plot-max-joints-per-side 8` does the same for kinematics

Plot styles and data validity:
- Each muscle's applied excitation and its MEP baseline use precisely the same color.
- Applied excitation is solid; baseline MEP is faint/dashed.
- Each joint's actual qpos and reference joint-angle trace use the same color.
- Actual is solid; reference is faint/dashed.
- Saved figures split muscles into anatomical families, then into panels of up to
  `--muscles-per-panel` traces; right and left sides are independent PNGs.
- Joint kinematics are also split into hip/knee/ankle/toe subplots.
- Kinematic references are now sampled in physical qpos units from the reference
  library at the adapted gait phase; they are NOT taken from normalized PPO inputs.
- Muscles belonging to multiple functional groups are shown only once for clarity.
- In live mode, the same-color paired style is used, but its full-side overview
  panels remain less subdivided than the saved PNGs.
