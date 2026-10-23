# Gait120 → MyoLeg: rebuild from the original recordings

This README describes **`build_gait120_from_source.py`**, which starts from Gait120's original LevelWalking **joint-angle MOT, force-plate MOT, and marker TRC** files. It does **not** repair exported reference files and does **not** depend on the old event-correction scripts.

**Paper:** Boo et al., *Comprehensive human locomotion and electromyography dataset: Gait120*, *Scientific Data* (2025): https://www.nature.com/articles/s41597-025-05391-0

**Scope:** S011–S120, all available `Trial*/step*.mot` files. The first ten participants have force-data limitations noted in the paper and are excluded by default. The existing extraction found 969 usable level-walking strides; re-extraction is expected to find the same input strides if the original data folders are unchanged, but you must verify the new `batch_summary.csv`.

## 1. Explicit adjustment ledger

These are the **only intentional departures/repairs** from the original kinematic dataset. The rightmost column distinguishes project choices from the published method.

| Change | Exact operation | Why | Basis |
|---|---|---|---|
| OpenSim → MuJoCo world coordinates | `T = [[1,0,0],[0,0,-1],[0,1,0]]`; root position `p_MJ = T @ p_OS` | OpenSim vertical is Y; MyoLeg vertical is Z | Our tested model compatibility mapping |
| Root/pelvis orientation | Source intrinsic `ZXY`: `R_OS = Rz(tilt) Rx(list) Ry(rotation)`; root `R_MJ = T R_OS Tᵀ`; quaternion **wxyz** | Gait120 uses Z–X–Y joint-angle convention; MyoLeg pelvis body has fixed rotation relative to root | Paper + marker-rigidity tests + MyoLeg XML |
| Joint angle degrees/signs | Convert degrees to radians. Reverse **both knees**, **left hip adduction**, **left hip rotation**. No other joint signs changed | Opposite positive coordinate conventions | Joint-marker compatibility audits |
| Dependent knees/patellae | Fill 14 dependent coordinates using compiled MuJoCo polynomial joint equality constraints | Setting driver knees alone does not populate dependent qpos | MyoLeg XML / constraint audits |
| S011 | **Do not flip** its left subtalar sign. Retain global mapping; add QC flag | Sign anomaly might be subject-specific gait; evidence was not sufficient to modify data | Audits and visual inspection |
| S060 root quaternions | If neighboring quaternion dot product is negative, flip the sign of the later quaternion; reject physical jumps >15°/frame | `q` and `-q` are the same rotation. Six otherwise valid S060 strides failed because of sign changes | Audit / S060 diagnostic |
| Root heading | One world-Z yaw rotation of each complete stride to align net progression with MuJoCo **+Y**. Initial XY position set to zero, preserve vertical and all joint angles | Avoid learning arbitrary/opposite laboratory travel headings | User's RL objective; previous 969/969 alignment |
| Velocity | Recompute `qvel` from transformed `qpos` via `mujoco.mj_differentiatePos` and the original timestamp differences | Quaternion configuration cannot be differentiated element-by-element | MuJoCo kinematic consistency |
| S029 Trial05 step01 / S115 Trial03 step02 | Exclude only their previously visually verified false *early left HS* annotation if it survives the noise filter. Preserve force and raw crossing in `*_raw_crossings.csv`. **Do not insert a replacement measured event.** | A false green HS appeared during toe-off | Original visualization audits; project-specific |
| Optional small force filter | Remove an **internal force-contact-state excursion shorter than 20 ms**, only when bracketed by the opposite state. No filtering of joint trajectories or GRF values. Default 20 ms; use `--min-dwell-ms 0` to disable | Short `TO→HS→TO` or `HS→TO→HS` force chatter | Project-specific; **NOT specified by the authors** |

There is **no subject rescaling, no retargeted marker fitting, no uniform stride-length/time normalization, no smoothing of qpos/qvel, and no fabricated HS/TO**.

## 2. HS/TO detection: what the paper actually says

For *level walking*, the authors detect **right-foot** heel strike when vertical GRF **first exceeds 20 N**, and right-foot toe-off when it falls **below 20 N**. Steps are extracted as **right HS → next right HS**. The paper does not provide a documented bilateral identification algorithm for the distributed data.

Our bilateral extension uses the **same 20 N condition** for `ground_force1_vy` and `ground_force2_vy`:

- **HS candidate:** plate force changes from `< 20 N` to `≥ 20 N`.
- **TO candidate:** plate force changes from `≥ 20 N` to `< 20 N`.
- The crossing timestamp is linearly interpolated between the two neighboring **original force samples**; the GRF is not resampled to the 100 Hz joint-angle frequency.
- The two plates are **not intrinsically left/right**. A complete nonzero-force bout belongs to one foot: while the plate is loaded, compare its measured center of pressure (CoP, original OpenSim X–Z plane) with `LHEE/LTOE` and `RHEE/RTOE` markers. The best *unambiguous* foot is attached to both the bout's HS and TO.
- **Crucial difference from the old exporter:** CoP at the exact TO crossing may be zero or unstable. We instead determine the foot from reliable **loaded** samples in the same bout; this avoids losing toe-offs purely because the plate unloads.
- A short in-between threshold fluctuation of less than 20 ms is ignored (optional, additional project QC). Original GRF samples are unchanged and every raw crossing is retained for inspection.

### Combine the force data at trial level first

The input `step01.mot` may **end while a foot is still on a plate**. `step02.mot` can contain that same foot's later TO. To avoid losing the event due to artificial step-file boundaries:

1. Read **all** `stepXX` GRF MOTs and TRC marker files in the **same subject/trial**.
2. Sort and deduplicate overlapping samples by their **original absolute time**, verifying duplicate values agree.
3. Detect complete contact bouts across the combined force recording; do **not** interpret a file boundary or a recording gap as a real HS/TO.
4. Assign the time-stamped measured events back to each stride's original time interval.

If a toe-off is outside the **entire recorded trial interval**, or the subject stepped on ground **outside the two plates**, **no GRF-based algorithm can measure it**. The new output marks such events **unobserved**. It does **not** add TO at 10% or 60.5% or left HS at 50%. Those were our earlier estimates, not the paper's measured events.

The script can output **zero, one, or more** observed crossings of a kind for a stride, rather than silently imposing one every cycle. Unassigned CoP events are explicitly labeled `unassigned`. Inspect events with multiple candidates before using them as ground truth.

## 3. Phase and preserved physical timing

For each original nominal right-HS-to-right-HS stride with first and last kinematic times `t0` and `t1`:

`T_stride = t1 - t0` and `phase_right_01 = (t - t0) / T_stride`.

No stride is sped up or slowed down; its original `time_s`, `stride_duration_s`, `qpos`, `qvel`, estimated walking speed, original sampled GRFs and raw marker trajectories are saved.

`phase_left_01` is generated **only when exactly one measured left HS lies within 25–75% of that nominal right stride** (this selection window is our quality-control heuristic, **not** from the paper). Otherwise it is **NaN**, and `left_hs_s` is NaN. This is deliberate: an unknown left HS is not replaced with an invented measurement. For simulated phase control, the policy may use `phase_right_01`; separate clock priors are a training choice, not contact-event data.

## 4. Run from original Gait120 files

Place `build_gait120_from_source.py` anywhere; it includes the conversion and detection helper functions. You only need installed `numpy`, `pandas`, `scipy`, `mujoco`, and access to the MuJoCo XML model and its includes.

### Pilot on two complete trials

```powershell
$model = 'C:\Users\BoussofaraS\OneDrive - University of Twente\Documents\myosuite\myosuite\simhive\myo_sim\leg\myolegs.xml'
$source = 'C:\Users\BoussofaraS\Downloads\Gait120_001_to_120'

python build_gait120_from_source.py `
  --model "$model" `
  --data-root "$source" `
  --output-root gait120_from_raw_pilot `
  --limit 2
```

`--limit` refers to **complete trials**, not isolated steps, so bilateral GRF-event detection always sees all available steps in each selected trial.

### Full extraction

```powershell
python build_gait120_from_source.py `
  --model "$model" `
  --data-root "$source" `
  --output-root gait120_from_raw_clean
```

The output is automatically aligned to **+Y** (same operation as `standardize_gait120_heading.py`). To inspect the original converted laboratory direction instead, append `--no-heading`. To disable the additional 20 ms contact debounce, append `--min-dwell-ms 0`.

Use a **new output directory**; do not overwrite the prior 969-stride dataset or run `fix_two_gait120_steps.py`, `apply_gait120_event_refractory.py`, `finalize_gait120_contact_events.py`, or `standardize_gait120_heading.py` on these newly generated files. The relevant steps are now built in.

## 5. Outputs and verification

For each `Sxxx/Trialyy/stepzz`:

| File | What it contains |
|---|---|
| `stepzz.npz` | `qpos`, `qvel`, original timing, joint columns/values, original GRF/marker samples, stride-specific right phase, measured-only left phase, speed |
| `stepzz_events.csv` | Filtered **observed** plate threshold crossings with L/R/`unassigned` label and exact measured time. **No estimated events.** |
| `stepzz_raw_crossings.csv` | All unfiltered threshold crossings, including short excursions and any narrowly rejected known false HS; reason attached |
| `stepzz_manifest.json` | Model/data conversions, input SHA256 hashes, QC, heading yaw, observed events and event limitations |
| `Trialyy/trial_measured_events.csv` | Accepted events detected on the **merged trial-level GRF**, before step slicing |
| `Trialyy/trial_raw_crossings.csv` | Raw/rejected force crossings from merged force data |
| `batch_summary.csv` | Per-stride extraction success, observed HS/TO counts, unassigned events, excluded jitter, warnings, heading |

For the original kinematic replay viewer (`play_gait120_with_events.py`): **green** = observed HS candidate, **magenta** = observed TO candidate, **amber** = the right-HS **dataset stride boundary** when not corroborated by a GRF crossing. No 50% left-HS estimate is generated in this pipeline.

**Important:** `marker_xyz_os_m` and `grf_values` remain in their **original OpenSim laboratory frame** even when `qpos/qvel` have been heading-aligned. Transform raw markers/forces if directly comparing them with MuJoCo world-frame quantities.

### Tests performed here

- **Actual raw S011 Trial01 Step01**: left HS **5.306647 s**, right HS **5.887732 s**, correct CoP/foot assignment; **no TO** in that particular raw file because both force channels remain loaded at the end.
- **Synthetic adjoining-step test using S011 foot-marker geometry**: two HS in the first step and two TO in the second; detector recovers the left TO and right TO from the **trial union**, and removes an inserted **3 ms false force gap**.
- Python syntax and pure force/marker processing passed. **The full MyoLeg/MuJoCo kinematic export could not be executed in this environment because `mujoco` is not installed here.** Run a pilot and check the produced `batch_summary.csv` before committing the full run.

## 6. Scientific limitations

A 20 N threshold crossing is only observable when a **particular plate actually recorded the contact transition**. Two force plates do not provide a continuous bilateral GRF throughout walking across the laboratory. Therefore, **the paper's event definition cannot guarantee all four measured HS/TO events in every stride**. Recording gaps, plate coverage, and ambiguity must be documented rather than replaced by typical-phase numbers. The kinematic trajectory may nevertheless be an excellent reference for joint-based imitation learning.
