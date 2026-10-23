# Gait120 bilateral heel-strike extraction

**Objective:** use the **Gait120 authors' right-HS-to-right-HS stride boundaries** and recover the left HS as faithfully as the *original measured* GRFs and markers permit. Toe-offs are not processed in this script. No original Gait120 files or existing references are modified.

## Inputs

The `--reference-root` folder is the output of our source-based Gait120 → MyoLeg converter, e.g. `gait120_from_raw_clean/`. Each `Sxxx/Trialyy/stepzz.npz` already contains `time_s`, the original `grf_time_s`, `grf_values`, `grf_channel_names`, and the original TRC marker trajectories (`marker_time_s`, `marker_names`, `marker_xyz_os_m`). **The detector ignores all previous event CSVs and previously estimated left-HS timestamps.** The GRF/marker arrays are still the original OpenSim laboratory-frame measurements, even for heading-normalized MyoLeg trajectories.

## Event rules, in priority order

1. **Right HS (authors' segmentation):** `time_s[0]` and `time_s[-1]` for each stride. These are the saved *kinematic frame timestamps* selected by the authors, not a newly calculated sub-frame threshold crossing.
2. **Left HS, 20 N GRF onset:** combine the original GRF signals from steps within each trial; find rising force crossings at 20 N on either plate. Reject brief interior state fluctuations (20 ms QC); label the contact bout left/right from the *loaded* CoP relative to experimental left/right heel and toe markers. A candidate must fall within 20–80% of its right stride, and, when the left heel marker trajectory is usable, within 130 ms of the left-heel landing-like minimum. The phase window and CoP criteria are **our quality-control conventions**, *not* rules claimed by the paper. This measurement is saved as `GRF_20N`.
3. **Optional subthreshold GRF:** for a plate-loading bout that never reaches 20 N, a 8 N crossing is considered only if the bout peaks at 15–20 N, lasts at least 25 ms, is confidently assigned to the left foot, and agrees with the marker timing. Output `LOW_FORCE_GRF_CANDIDATE`; this is **not the paper's 20 N method**, and needs review before being regarded as reliable.
4. **If no credible GRF left HS is available:** find the *observed left heel marker* height minimum in a midstride window, requiring a clear preceding heel descent. Estimate the offset between such minima and measured GRF HS on other strides, using a per-subject median where available or dataset median otherwise. Save as `ASSUMED_marker_kinematic_estimate_NOT_GRF`; this is an estimated *contact time*, **not** a measured force event. No fixed 50% phase assumption is made.
5. If neither force nor marker provides credible timing, the left HS is `MISSING_NO_OBSERVABLE_EVENT`, not invented.

### Why the fallback?

Gait120 used **two floor-mounted force plates**, which don't measure foot-ground contact everywhere on the walkway. An absent 20 N crossing can mean that a foot landed outside the plates, that the provided recording began while the plate was already loaded, or that an exceptionally small force did not reach threshold. Lowering the threshold cannot reconstruct a contact that was **never measured**. The marker estimate is clearly identified for this reason.

The paper's 20 N rule and right-HS segmentation are from Boo et al., *Scientific Data* (2025): https://doi.org/10.1038/s41597-025-05391-0 . Bilateral plate-to-foot labeling, 20 ms deglitching, subthreshold check and marker fallback are **our extensions**.

## Usage (PowerShell)

```powershell
# Recommended: use the references from the clean, original-file converter.
python extract_gait120_bilateral_heel_strikes.py `
  --reference-root gait120_from_raw_clean `
  --output-root gait120_hs_ready

# Inspect first 10 trials without copying NPZ files:
python extract_gait120_bilateral_heel_strikes.py `
  --reference-root gait120_from_raw_clean `
  --output-root gait120_hs_pilot `
  --limit 10 --csv-only

# Strict force-based version: don't fill missing HS from kinematics.
python extract_gait120_bilateral_heel_strikes.py `
  --reference-root gait120_from_raw_clean `
  --output-root gait120_hs_grf_only `
  --no-marker-fallback
```

The standard command creates a **new** copy of the reference tree with **unchanged `qpos`, `qvel`, raw GRFs, markers and stride timing** but recomputed `left_hs_s`, `phase_left_01`, `phase_left_rad` and HS annotations. For each step, the new `step##_events.csv` holds only accepted *measured* left-HS candidate(s); marker-inferred left HS is kept in the manifest and displays **amber**, not as a green measured crossing.

The root outputs include:

- `bilateral_heel_strikes.csv`: per-step right HS start/end, left HS time/phase, evidence source, plate and diagnostic details.
- `calibration_offsets.csv`: measured left-GRF-HS minus heel-height-minimum differences used to calibrate the marker fallback.
- `heel_strike_failures.csv`: trial-level parsing/detection errors.

To visually check a stride, use the adapted replay:

```powershell
python play_gait120_with_events_hs.py `
  --model "$model" `
  --reference 'gait120_hs_ready\S012\Trial01\step01.npz' `
  --speed 0.5 --hold 0.8
```

**Colors:** green = measured 20 N left HS; amber = authors' right-HS boundary or inferred marker-based left HS / weak-force candidate. Pink TOs are not part of this HS-only export.

## Validation and caveats

Tested with the available S011 and S012 reference GRF/marker examples: measured left HS occurs at 5.30665 s (47.9%) and 4.38913 s (50.9%), respectively. Also tested a synthetic below-20 N contact and an artificially missing-GRF contact. The reference-copy test confirmed `qpos`, `qvel`, and `grf_values` are unchanged. **The entire 969-reference dataset has not been tested in this environment.** Inspect the fraction of measured vs marker-estimated events and spot-check different subjects before relying on estimated HS as training supervision.
