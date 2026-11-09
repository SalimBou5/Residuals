# Verify the loopable gait references (Step 1)

This is a **read-only** verification step. It never modifies `*_loopable.npz` files and does not require MuJoCo.

## 1. Process the gait references at your chosen 2.5% seam

```bash
python make_gait_loopable.py --input references_gait120_standardized --output loopable_references --sample-hz 200 --seam 0.025
```

## 2. Test a handful of strides

```bash
python verify_loopable_gait.py --input loopable_references --source-root references_gait120_standardized --output loop_verification --max-files 6
```

To verify **all** strides, omit `--max-files 6`. To verify one stride, supply its actual `*_loopable.npz` file instead of the directory.

## Outputs

Each stride produces a PNG plot and JSON metrics. The output folder also contains `summary.csv` and `summary.json`.

The PNG shows right/left knee and ankle positions over **three repeated cycles** on the left, and joint angular velocities across the **right heel strike** on the right (zoomed ±8% of stride by default). Orange shading marks the first/last 2.5% correction region when built using `--seam 0.025`.

With `--source-root`, velocity plots show three quantities: (1) original supplied `qvel`, (2) velocity of the uncorrected original-position PCHIP interpolant, and (3) the corrected reference's `qvel`. Comparing (2) vs (3) isolates changes due to the seam correction from changes due to interpolation itself.

## Interpret the results

- `angle_jump_at_hs_deg` and `velocity_jump_at_hs_deg_s` should be near zero: the two sides of the loop meet.
- Examine the **shape** of blue velocity curves within the orange boundary region: a zero jump does not rule out fast velocity excursions or acceleration spikes.
- `max_velocity_change_due_to_seam_deg_s` estimates the maximum additional angular-velocity change caused specifically by the boundary correction.
- `peak_velocity_ratio_corrected_to_measured_near_hs` compares the corrected velocities with supplied original `qvel`, but includes interpolation differences. Do **not** attribute this ratio solely to seam correction.
- Large acceleration values may reflect sharp but already-present changes in measured kinematics or the interpolation. They are diagnostics, **not** automatic rejection thresholds.

This checks **kinematic** smoothness only. It does not verify contacts, muscle feasibility, or MuJoCo equality constraints.
