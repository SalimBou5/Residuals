# Gait120 → MyoLeg: final foot-contact event annotations

The input reference dataset already contains the converted MyoLeg joint trajectories, original timing, heading-aligned root motion, GRFs, and OpenSim marker coordinates. **Do not repeat the kinematic export.**

## Why this step is necessary

The Gait120 stride files are segmented from **right heel strike to right heel strike**. The GRFs from the instrumented force plates are not continuous, independent measurements of **both feet throughout every stride**. Depending on the file, the plate data often capture opposite-leg heel strikes (`step01`) or two toe-offs (`step02`), not both. A missing force-plate 20 N crossing is **not evidence that toe-off did not occur**.

Some signals also cross the 20 N threshold repeatedly within a few milliseconds. A 20 ms **refractory period per force plate** suppresses these extra candidate events, without modifying the original GRF data or the 20 N threshold.

## One-time finalization

```powershell
python finalize_gait120_contact_events.py `
  --input-root gait120_references_forward `
  --output-root gait120_references_ready
```

The output directory must be empty/new. Input files are unchanged.

For a quick test using a handful of files:

```powershell
python finalize_gait120_contact_events.py `
  --input-root gait120_references_forward `
  --output-root gait120_references_test `
  --limit 10
```

If a pilot contains fewer than 10 valid measured toe-offs for either leg, the script prints a warning and uses the previous audit's measured medians (left 10%, right 60.5%). The full batch derives the medians from its own credible labeled crossings.

## How the selected events are defined

All phases below are expressed as fractions of **the recorded right-HS-to-right-HS stride**, `p = (t - t0) / (t1 - t0)`. Each stride keeps its *actual* duration and sample times; no time warping is applied.

| Event | Selection | Provenance |
|---|---|---|
| Right HS (beginning and end) | Dataset stride boundaries | `dataset_right_HS_boundary_*`; not necessarily independently observed on a force plate |
| Left HS | Previously selected, CoP-labeled, plausible 20 N upward crossing at 30–70% | `observed_20N_crossing` |
| Left HS if unobserved/unreliable | 50% of right stride | `estimated_half_stride` |
| Left TO | Plausible 20 N downward crossing at 3–25% | `observed_20N_crossing` |
| Right TO | Plausible 20 N downward crossing at 45–80% and following left HS | `observed_20N_crossing` |
| Left/right TO if unobserved/unreliable | Median normalized time among reliable measured toe-offs for that leg; typically 10% and 60.5% | `estimated_cohort_median` |

TO candidates are first filtered with the 20 ms per-force-plate refractory period. If several candidates remain in the relevant window, the crossing closest to the observed cohort median is selected, and a warning is recorded. Temporal ordering is checked: **left TO < left HS < right TO < right HS at the stride end**. These windows are *quality-control heuristics for these level-walking recordings*, **not general biological constraints**. Estimated events are priors, not measured ground-contact labels.

## Output files

For each `S###/Trial##/step##`:

- **`step##.npz`** — Preserves `qpos`, `qvel`, original `grf_values`, markers, original timestamps and stride duration. Adds `left_to_s`, `right_to_s`, `left_to_phase_in_right_stride`, `right_to_phase_in_right_stride`, `left_to_phase_in_left_stride`, and provenance strings. Updates left-HS and left-phase fields consistently.
- **`step##_selected_events.csv`** — **The authoritative selected annotations for replay/training** (five records: right HS at both boundaries; one left HS, left TO, right TO). Every event has an `event_source` marking observed/estimated/boundary status.
- **`step##_events.csv`** — Original/raw force-crossing *candidate* CSV, copied unchanged for traceability; do **not** treat all rows as independent ground contacts.
- **`step##_manifest.json`** — Updated `events` with selected timings and provenance, and an `event_finalization` section preserving older event metadata and all warnings.

At the output root, `event_finalization_summary.csv` lists source labels, timings, suppressed crossing counts, warnings, and any failures. **Expect 969/969 successes** if the 969 source references are intact. Verify the script's actual summary rather than assuming success.

## Replay **selected** events (important)

The **original viewer** reads raw `_events.csv` and is not suitable for validating the new estimated TO markers. Use the matching version:

```powershell
python play_gait120_with_events_final.py `
  --reference "gait120_references_ready\S029\Trial05\step01.npz" `
  --events-only
```

```powershell
python play_gait120_with_events_final.py `
  --model "$model" `
  --reference "gait120_references_ready\S029\Trial05\step01.npz" `
  --speed 0.5 --hold 0.8
```

Color coding: green = observed HS, pink = observed TO, amber = *estimated* event or dataset stride boundary. A colored marker signifies an **annotation**, not contact physics computed by MyoLeg.

## For imitation learning

Read the selected event timestamps and `event_source` from the `.npz`, manifest, or `_selected_events.csv`. If you want **strict contact supervision**, use observed events only (and distinguish the dataset-based right-HS boundaries). Estimated toe-offs can instead serve as initialization or weaker phase references. Do not represent a median-imputed time as a measured toe-off.

The reference phase is available for each leg and is based on the original duration of each stride. In particular, `left_to_phase_in_left_stride` is the toe-off phase relative to the left HS, whereas `left_to_phase_in_right_stride` is the earlier event location in the right stride.

The scripts use NumPy and the Python standard library to finalize event data. MuJoCo and GLFW are needed only for graphical replay (which does **kinematic playback**, not physics stepping).
