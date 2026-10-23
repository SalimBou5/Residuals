# Gait120 → MyoLeg: technical reference-dataset preparation

**Purpose:** Convert the Gait120 *LevelWalking* recordings into reproducible, phase-annotated **MuJoCo/MyoLeg kinematic reference trajectories** suitable for the next stage of muscle-driven imitation learning.

**Current result (14 October 2026):** The heading-alignment run supplied with this project reports **969/969 strides aligned, 0 failures**, covering the available steps from **S011–S120**. All six formerly failing S060 trajectories were recovered by the quaternion-continuity correction; S060 then required approximately ±180° heading alignment. The final aligned directory in that run was `gait120_references_forward_2/`.

**Scope:** These files provide positions, velocities, timing and gait-event annotations. They are **not** ground-truth muscle activations, validated ground-contact simulations, or dynamically feasible reference rollouts.

---

## 1. The four-file pipeline

The scripts are listed in the order in which they are used. All helper functions can live inside these four scripts; the description below refers to the logic in the supplied versions.

| Script | Role | Modifies existing references? |
|---|---|---|
| `batch_extract_gait120_quaternion_fixed.py` | Read original Gait120 MOT/TRC/GRF files; convert pelvis and joint coordinates; reconstruct dependent knees/patellae; calculate MuJoCo velocities; identify force events; save one reference per stride. | Creates initial `gait120_references_full/` exports. Does not edit Gait120 source files. |
| `fix_two_gait120_steps.py` | Correct exactly two misleading left-heel-strike annotations: S029/Trial05/step01 and S115/Trial03/step02. | **Yes**: edits those two exported NPZ/CSV/manifest triplets **in place**, after making backups. |
| `standardize_gait120_heading.py` | Apply one rigid horizontal-world rotation per stride to make net progression point along MuJoCo **+Y**; recompute velocities. | **No**: creates a **separate output dataset**. |
| `play_gait120_with_events.py` | Replay the saved poses in MuJoCo while highlighting estimated/observed heel strikes and toe-offs. | No. Viewer only. |

**Recommended data lineage:**

```text
Gait120 source recordings: S011 ... S120
    |  joint-angle MOT + force-plate MOT + marker TRC
    v
batch_extract_gait120_quaternion_fixed.py
    |
    v
gait120_references_full/
    |   qpos, qvel, raw force/marker data, phase and event annotations
    v
fix_two_gait120_steps.py
    |   correct S029 and S115 annotations only
    v
gait120_references_full/  [event-corrected, original world heading]
    |
    v
standardize_gait120_heading.py
    |
    v
gait120_references_forward_2/  [consistent +Y walking heading]
    |
    +--> play_gait120_with_events.py  [visual inspection]
    |
    +--> imitation-learning data loader
```

### Why any conversion is needed

The OpenSim joint-angle recordings cannot be copied blindly into a MuJoCo state. The two models have different **world coordinate conventions**, a different representation of the **floating pelvis**, some different **joint-angle signs**, and MyoLeg contains **coupled knee/patella coordinates** that are not independent gait measurements. Furthermore, measured walking trajectories must be accompanied by correctly timed **velocities and gait phases** before they can serve as trajectory-tracking references.

This pipeline transfers **kinematics**; it does **not** solve inverse dynamics, generate muscle excitations, or rerun OpenSim inverse kinematics for each subject.

---

## 2. Original inputs and assumptions

For each available `Sxxx/Trialyy/stepzz`, the exporter matches the following files by subject, trial and step name:

```text
Sxxx/
├── JointAngle/LevelWalking/Trialyy/stepzz.mot
│     OpenSim generalized coordinates and timestamps
└── MotionCapture/LevelWalking/
      ├── MOT/Trialyy/stepzz.mot   force-plate measurements (GRFs and CoP)
      └── TRC/Trialyy/stepzz.trc   experimental marker trajectories
```

The converter requires monotonically increasing timestamps and checks the synchronisation of the joint-angle MOT and TRC. It checks that the GRF interval covers the kinematic interval, that needed columns exist and that numeric values are finite.

- **Joint angles:** OpenSim MOT angles are declared in **degrees**; MuJoCo hinge angles are stored in **radians**.
- **Linear positions:** converted/stored in **metres**. TRC marker files can declare `mm` or `m`; the reader converts millimetres to metres.
- **Kinematic samples:** their **original times and count are preserved**; the exporter does not uniformly resample a stride to 100 or 101 points.
- **Force samples:** stored at their original, generally higher, sampling rate; event detection uses the force time vector rather than rounding force events to the nearest kinematic frame.
- **Model:** the exact MuJoCo `myolegs.xml` (with its included assets and joint definitions) is required. The mapping assumes the MyoLeg model structure studied in this project.

The exported dataset is the set of **available, successfully processed** gait segments, not automatically every possible subject/trial combination advertised by the original publication. The initial selected extraction range is S011–S120; S001–S010 were not part of our selected reference set.

---

## 3. Coordinate transformation: OpenSim world → MuJoCo world

### 3.1 World axes

For these recordings/model definitions, the basis change used throughout the pipeline is

$$
\mathbf T =
\begin{bmatrix}
 1&0&0\\
 0&0&-1\\
 0&1&0
\end{bmatrix}.
$$

It maps a vector expressed in the source OpenSim coordinates into the chosen MyoLeg MuJoCo coordinates:

$$
\mathbf p_{\mathrm{MJ}}=\mathbf T\,\mathbf p_{\mathrm{OS}},
\qquad
\begin{bmatrix}x\\y\\z\end{bmatrix}_{\mathrm{MJ}}
=
\begin{bmatrix}x\\-z\\y\end{bmatrix}_{\mathrm{OS}}.
$$

In particular, **OpenSim vertical +Y becomes MuJoCo vertical +Z**. This is a proper three-dimensional rotation (not a mirror reflection):

$$\mathbf T^\top\mathbf T=\mathbf I,\qquad\det(\mathbf T)=1.$$

It is equivalent to rotating the coordinate basis **+90° around the X axis**. The Gait120 laboratory direction of travel may vary; **this basis conversion is not the later heading-normalization operation**.

### 3.2 Pelvis translation

The Gait120 MOT contains `pelvis_tx`, `pelvis_ty`, `pelvis_tz`. They provide the translation of the source pelvis frame. The exporter writes the MuJoCo `root` free-joint position using

$$
\mathbf p_{\mathrm{root}}^{\mathrm{MJ}}(t)
= \mathbf T
\begin{bmatrix}
\texttt{pelvis\_tx}(t)\\
\texttt{pelvis\_ty}(t)\\
\texttt{pelvis\_tz}(t)
\end{bmatrix}.
$$

This operation changes the **coordinate representation**, not the physical path.

### 3.3 Pelvis orientation and Euler-rotation order

The three source coordinates are:

- `pelvis_tilt`
- `pelvis_list`
- `pelvis_rotation`

A three-dimensional orientation depends on both the angles **and the order of rotation**. The working convention was selected through a marker-rigidity test: for the sample examined, the **intrinsic `ZXY`** convention kept markers attached to the pelvis markedly more stationary in the reconstructed pelvis frame than the alternative Euler orders tested.

The source pelvis rotation matrix is reconstructed as

$$
\mathbf R_{\mathrm{pelvis}}^{\mathrm{OS}}(t)
= R_z\bigl(\mathrm{tilt}(t)\bigr)
  R_x\bigl(\mathrm{list}(t)\bigr)
  R_y\bigl(\mathrm{rotation}(t)\bigr).
$$

In the implementation, SciPy's **uppercase** `Rotation.from_euler('ZXY', ..., degrees=True)` is used; uppercase denotes intrinsic rotations. Using lowercase `'zxy'` would not mean the same thing.

There is an additional model-structure detail. In `myolegs_chain.xml`, the `pelvis` body has a fixed quaternion representing **+90° about X** relative to the free `root` body. Therefore, the correct **root** rotation to save is

$$
\boxed{
\mathbf R_{\mathrm{root}}^{\mathrm{MJ}}(t)
=\mathbf T\,\mathbf R_{\mathrm{pelvis}}^{\mathrm{OS}}(t)\,\mathbf T^\top
}
$$

and it is converted to a **scalar-first** quaternion:

$$q_{\mathrm{root}}=(w,x,y,z).$$

This is intentional: the root free joint and the pelvis body are not identical coordinate frames. The fixed pelvis-body rotation already exists in the compiled MuJoCo model. Applying it again, or simply copying the OpenSim Euler angles to MuJoCo, would orient the model incorrectly. Similarly, the root body's initial XML Euler attributes do not need to be multiplied again into every exported free-joint quaternion.

**Implementation outline:**

```python
T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
R_os = Rotation.from_euler(
    'ZXY', joints[['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']].to_numpy(),
    degrees=True,
).as_matrix()
root_pos = (T @ joints[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].to_numpy().T).T
root_quat_wxyz = Rotation.from_matrix(T @ R_os @ T.T).as_quat(scalar_first=True)
```

---

## 4. Leg-joint mapping and sign corrections

The independent MyoLeg leg coordinates are mapped by **joint name**, not by assuming identical column indices in the two files. For each joint coordinate:

$$q_j^{\mathrm{MJ}}(t) = s_j\,q_j^{\mathrm{Gait120}}(t)\,\frac{\pi}{180},
\quad s_j\in\{-1,+1\}.$$

The established mapping is:

| Joint coordinates | Sign | Rationale |
|---|---:|---|
| `knee_angle_r`, `knee_angle_l` | **−1** | Gait120 reports the tested knee flexion convention with the opposite sign to MyoLeg's knee coordinate. |
| `hip_adduction_l` | **−1** | Empirical left-leg marker/segment-rigidity comparisons supported the opposite convention. |
| `hip_rotation_l` | **−1** | Same comparison supported reversing the left hip-rotation convention. |
| Hip flexion (both), right hip adduction/rotation | +1 | No correction required by the adopted mapping. |
| Ankle, subtalar, and MTP joints (both sides) | +1 | Preserve their source signs in the common, global mapping. |

The actual independent set contains **14 coordinates**, seven per leg: hip flexion, hip adduction, hip rotation, knee angle, ankle angle, subtalar angle and MTP angle.

**Why not just use the same sign for every joint?** A coordinate's positive direction is set by the corresponding model joint axis and coordinate definition. Anatomical names alone do not guarantee that two models use identical positive directions. We tested sign choices using the observed movement of pelvis, femur, tibia and foot markers, rather than choosing signs solely for visual aesthetics.

**Special case S011:** one diagnostic favored the opposite **left subtalar** sign for this subject, unlike the subjects compared elsewhere. S011 was visually plausible, so we **retain it with the same mapping as everyone else** and record the `s011_subtalar_flag`; we do **not** introduce an unvalidated subject-specific sign flip.

---

## 5. Reconstruction of dependent knee and patellar coordinates

MyoLeg is not a set of entirely independent hinge joints. Its knee mechanism includes extra translations/rotations for the tibia and patella that follow **polynomial coupling constraints** encoded in the MuJoCo XML (`<equality><joint ... polycoef="..." /></equality>`).

The converter first writes the 14 independently mapped leg angles. It then computes the **14 dependent knee/patellar coordinates**—seven for each knee—from their driver coordinates. For an equality in the compiled model, with dependent coordinate $q_d$, driving coordinate $q_s$, reference positions $q_{d,0}$ and $q_{s,0}$, and polynomial coefficients $c_0,\dots,c_4$:

$$
\boxed{
q_d = q_{d,0} + \sum_{k=0}^{4} c_k\left(q_s-q_{s,0}\right)^k
}
$$

The implementation uses `model.eq_obj1id`, `model.eq_obj2id`, `model.jnt_qposadr`, `model.qpos0` and `model.eq_data[..., :5]` so it reads **the compiled model's actual equality parameters** rather than hard-coding knee trajectories.

This detail matters because merely setting `knee_angle_r` and `knee_angle_l` then calling `mujoco.mj_forward` does **not** automatically rewrite the dependent elements of `data.qpos` to the required polynomial values. If those elements stay at defaults, the saved pose can be geometrically inconsistent.

The script independently recomputes each predicted dependent coordinate and checks the maximum error:

$$\max_t|q_d(t)-q_d^{\mathrm{predicted}}(t)|\le 10^{-10}.$$

The equality-generated dependent coordinates are **not arbitrarily clamped** to slightly narrower joint-range limits, because doing so would violate their equality constraints. Independent leg-joint ranges are checked separately.

**MyoLeg state size in this project:** `nq = 35` (7 root position/orientation values + 28 scalar joint positions), `nv = 34` (6 root velocities + 28 scalar joint velocities). These are **MuJoCo state dimensions**, not numbers of independent neural control inputs.

---

## 6. Root-quaternion sign continuity (the S060 fix)

A spatial orientation quaternion has a double representation:

$$q\equiv -q.$$

Both quaternions describe exactly the same physical rotation. Conversion from rotation matrices to quaternions can nevertheless alternate between these equivalent signs on successive frames. A naïve continuity check may read this as a discontinuity even when the physical motion is smooth.

For each successive root quaternion, the exporter chooses the sign closest to the preceding quaternion:

$$
q_i\leftarrow
\begin{cases}
-q_i,& q_{i-1}\cdot q_i<0,\\
\phantom{-}q_i,& q_{i-1}\cdot q_i\ge 0.
\end{cases}
$$

The **rotations remain unchanged**. The check also rejects non-unit quaternions and genuine large physical orientation steps. The physical angle between consecutive unit rotations is computed after resolving sign ambiguity, equivalently as

$$\Delta\theta_i=2\arccos\bigl(|q_{i-1}\cdot q_i|\bigr).$$

The implemented guard rejects steps above **15° per frame**; this is a diagnostic limit, not a physiological law.

**Observed S060 issue:** six steps initially failed a raw quaternion-neighbor check. The separate diagnostic found multiple negative neighboring dot products near −1 but very small real rotations—the largest was only **0.935° per frame**. Sign unwrapping fixed the representation discontinuities and allowed those six steps to export. This fix should be applied **before deriving velocities**.

This quaternion issue is **distinct from S060's opposite direction of progression**: changing $q$ to $-q$ never rotates the body by 180°.

---

## 7. Computing MuJoCo reference velocities

MuJoCo free-joint positions contain a quaternion, which is **four numbers for three rotational degrees of freedom**. Therefore:

- `qpos` has 35 components.
- `qvel` has 34 components.
- Simple componentwise subtraction of `qpos` is **not** a valid way to calculate angular velocities.

The exporter uses MuJoCo's quaternion-aware configuration difference:

```python
mujoco.mj_differentiatePos(model, qvel_out, delta_t, qpos_a, qpos_b)
```

For interior frame $i$, the reference derivative is a centered finite difference in configuration space:

$$
\dot{\mathbf q}_i
\approx \mathrm{mj\_differentiatePos}
\left(\mathbf q_{i-1},\mathbf q_{i+1},t_{i+1}-t_{i-1}\right).
$$

At the first and last frames the method uses a one-sided difference. (The displayed mathematical notation denotes the model-aware operation, not division of two raw quaternion vectors.) This procedure handles free-joint angular velocity in MuJoCo's correct velocity coordinates. It also derives velocities for the polynomially coupled joint positions.

**No smoothing or interpolation was added in this batch.** Some recorded joint-angle trajectories have relatively high angular-velocity peaks near gait transitions; these were investigated previously and should not be silently filtered just to obtain smoother curves. If a policy later requires smoothed references, that should be a documented, separately evaluated preprocessing step.

After heading normalization (Section 10), **all `qvel` values are recomputed from the transformed `qpos`**, using the original timestamps, to guarantee consistent root velocities.

---

## 8. Detecting heel strikes and toe-offs from GRF

### 8.1 Force rule: 20 N threshold crossings

The event extractor uses each plate's **vertical GRF** channel, e.g. `ground_force1_vy` and `ground_force2_vy` in the original OpenSim frame:

- **Heel-strike candidate (HS):** force changes from below 20 N to **at least 20 N**.
- **Toe-off candidate (TO):** force changes from at least 20 N to **below 20 N**.

In code, the contact signal is `force >= 20.0`. This follows the 20 N event convention described for the Gait120 segmentation; within the exported step, crossings are still **candidate observations**, not automatically perfect anatomical labels.

The crossing instant is linearly interpolated between consecutive force samples. If the threshold lies between $(t_{k-1},F_{k-1})$ and $(t_k,F_k)$:

$$
t_e = t_{k-1}+
\frac{20-F_{k-1}}{F_k-F_{k-1}}(t_k-t_{k-1}).
$$

This is more precise than simply attaching the event to the nearest 100 Hz kinematic frame.

If a plate is **already loaded at the first force sample**, the code records that state but does not invent a heel strike at the beginning. Likewise, a force plate showing zero load does **not** by itself prove that a particular foot is in swing: it may be supported elsewhere.

### 8.2 Identify which foot caused a force event

A force-plate number cannot safely be interpreted as a permanent left/right label. To assign a side, the script uses experimental foot-marker trajectories and center-of-pressure (CoP) position:

1. Find the nearest force and marker sample at event time $t_e$.
2. Read CoP from `ground_force#_px` and `ground_force#_pz` (horizontal **OpenSim X–Z** plane).
3. Read each foot's experimental heel (`LHEE`/`RHEE`) and toe (`LTOE`/`RTOE`) markers, also in OpenSim X–Z.
4. For an **onset/HS**, compare CoP-to-heel distances. For an **offset/TO**, compare the distance to each heel–toe **line segment**, allowing the CoP to migrate toward the toes during stance.
5. Choose a side only if its CoP-to-foot distance is sufficiently small **and** clearly smaller than for the other foot.

In the supplied code, the conservative geometric criteria are:

- onset distance ≤ **0.18 m**;
- offset distance ≤ **0.14 m**;
- distance advantage over the other foot ≥ **0.07 m**.

A side cannot be established when these conditions fail: the CSV records `foot_side = unassigned`. During a single continuous contact bout, an assigned side is reused for its later toe-off.

**Important:** The numbers above are practical assignment tolerances, **not** anatomical laws. A high-confidence assignment is stronger evidence than an unassigned crossing, but the algorithm still merits spot checks around unusual contacts.

### 8.3 Preserve raw force evidence

The extracted `_events.csv` contains force crossings, timing, plate, side-assignment method and geometric distances. The `.npz` independently retains the **original unresampled force channels** and force timestamps. This allows later reanalysis without returning to the original MOT files.

Unless specifically corrected in the two known cases (Section 9), all observed crossings remain in the exported event CSV; multiple crossings are not automatically merged into one anatomical event.

---

## 9. Gait-cycle phase and the two manual event corrections

### 9.1 A stride retains its own time scale

Gait120's prepared walking steps are treated as nominal **right heel strike → next right heel strike** strides. For each individual exported step:

$$
T=t_1-t_0,
\qquad
u_R(t)=\frac{t-t_0}{T},
\qquad
\phi_R(t)=2\pi\nu_R(t).
$$

Here:

- $t_0,t_1$: actual timestamps of the first and last recorded kinematic samples;
- $T$: **this specific stride's duration**;
- $\nu_R$: normalized right-stride position from **0 to 1**;
- $\phi_R$: right-stride phase from **0 to $2\pi$**.

No stride is time-warped to a common duration. A 0.9 s stride and a 1.2 s stride still contain their original sampled timestamps, joint speeds, and movement rates. The phase is just an extra coordinate. The final stored right-stride phase is 1 (or $2\pi$); a periodic controller can reduce it modulo one cycle.

The boundary labeling comes from the **dataset segmentation**. The code records whether a labeled right-foot force crossing corroborates each boundary within 30 ms; otherwise it is an **inferred segmentation boundary**, not an independently measured GRF crossing.

### 9.2 Left-leg phase

If a left HS is observed and geometrically assigned, the exporter chooses the eligible candidate closest to 50% of the right stride. In the original exporter, eligible candidate phases were restricted to **10–90%**; this permissive window explains the two false early selections described below.

If no usable measured left HS exists, the left HS is explicitly **estimated at half the stride**:

$$t_{\mathrm{LHS}}^{\mathrm{est}}=t_0+\frac{T}{2}.$$

Writing the selected/estimated HS offset as

$$\delta_L=\frac{t_{\mathrm{LHS}}-t_0}{T},$$

the phase for the left leg is

$$
\nu_L(t)=\operatorname{mod}\bigl(\nu_R(t)-\delta_L,1\bigr),
\qquad
\phi_L(t)=2\pi\nu_L(t).
$$

For the fallback, $\delta_L=0.5$. The source label in the manifest distinguishes a **measured, CoP-labeled candidate** from the **assumed half-stride estimate**. Left phase wraps modulo 1 at the left HS; this is an intentional periodic phase discontinuity, not a discontinuity in joint positions.

**Toe-off is not estimated** in this pipeline. When a reliable downward force crossing is absent, no numerical toe-off is fabricated.

### 9.3 Why we manually corrected only two steps

Visual replay exposed two failures of the original permissive event selection:

| Reference | Spurious labeled left HS | Observation |
|---|---:|---|
| `S029/Trial05/step01` | **10.18%** of right stride | Green HS marker coincided with a pink toe-off marker, not an actual left landing. |
| `S115/Trial03/step02` | **11.79%** of right stride | Same pattern; the expected later left HS was missing from the selected labels. |

The script `fix_two_gait120_steps.py` **only** changes those two steps:

1. Locate the specific early **left HS** row in `stepXX_events.csv` and remove it from that **display/annotation CSV**.
2. Preserve all toe-off rows and all original recorded force data in the NPZ.
3. Set `left_hs_s = t0 + T/2`, `phase_left_01 = (phase_right_01 - 0.5) mod 1`, and `phase_left_rad = 2π phase_left_01`.
4. Update the manifest to say the left HS is **estimated**, not measured, and record the rejected original annotation.
5. Save `.before_two_step_fix.bak` backups of the NPZ, CSV and manifest **before changing anything**.

**Why remove the crossing from the CSV if it was really measured?** That CSV doubles as the original replay viewer's set of drawable annotations; otherwise the viewer would keep showing the known-false green heel-strike marker. The raw GRF samples remain unchanged in the NPZ and the removed event row remains recoverable from the backup/manifest.

**Run once, after extraction and before heading alignment.** It is deliberately not a broad automatic relabeling procedure; other suspicious raw force candidates remain candidates pending further investigation.

---

## 10. Standardize all walking directions to MuJoCo +Y

### 10.1 Why

Original laboratory recordings contain a common progression direction for most subjects but S060 walks in the **opposite** direction. For an initial forward-walking imitation policy, asking the agent to track both headings would introduce unnecessary variation in world-frame root motion and heading-dependent objectives. We therefore use a consistent progression axis for the training references.

This operation **does not** alter the gait cycle, joint angles, stride duration, or relative body mechanics. It changes only the **global placement/orientation** of the same moving skeleton.

### 10.2 Estimate each stride's heading

Using the already-converted MuJoCo root translation, compute net horizontal displacement over the recorded step:

$$
\Delta\mathbf p_{xy}=
\begin{bmatrix}x_{\mathrm{end}}-x_{\mathrm{start}}\\
y_{\mathrm{end}}-y_{\mathrm{start}}\end{bmatrix},
\qquad
\theta_{\mathrm{old}}=\operatorname{atan2}(\Delta y,\Delta x).
$$

The target progression vector is **+Y**, i.e. target heading $\theta_{\mathrm{target}}=\pi/2$ measured from +X. The script calculates one yaw angle per entire stride:

$$
\boxed{
\psi=\operatorname{wrap}_{[-\pi,\pi]}
\left(\frac\pi2-\theta_{\mathrm{old}}\right)
}
$$

It refuses trajectories whose start-to-end horizontal displacement is **less than 0.2 m**, because heading would be unreliable. This is why a valid `ALIGNED` result is stronger than merely generating a file.

### 10.3 Rotate root translation and orientation

Define the vertical-axis world rotation:

$$
\mathbf R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\
0&0&1
\end{bmatrix}.
$$

The script subtracts the **initial horizontal** root position and rotates the entire horizontal root trajectory:

$$
\boxed{
\mathbf p'_{xy}(t)=
\mathbf R_{z,xy}(\psi)\bigl[\mathbf p_{xy}(t)-\mathbf p_{xy}(t_0)\bigr]
}
$$

The vertical root position is **unchanged**. The initial root horizontal position becomes (0,0).

The root orientation must rotate consistently with its new world path:

$$
\boxed{
\mathbf R'_{\mathrm{root}}(t)=\mathbf R_z(\psi)\,\mathbf R_{\mathrm{root}}(t)
}
$$

Equivalently, the new root quaternion is a **left-multiplied world-yaw quaternion** times the original root quaternion. **Left multiplication** is important: rotating on the right would apply a local body-frame rotation and would not represent the same rigid movement placed in a rotated world.

All **non-root joint positions** are copied bit-for-bit, and phase arrays/timestamps/stride duration are preserved. The script then sign-unwraps the new quaternion sequence, recomputes MuJoCo `qvel` from the rotated `qpos`, and checks the new net displacement is $(0,\|\Delta\mathbf p_{xy}\|)$ within numerical tolerance.

### 10.4 S060 example and overall result

All six S060 strides originally progressed opposite to the others. Your alignment log shows yaw corrections of approximately **−178.6° to −179.9°**, with one equivalent value near **+179.6°**. That is exactly what one expects for a near-reversed walking direction; it is **not** evidence of a coordinate bug.

The same log ends with:

```text
Discovered 969 stride references
Output: gait120_references_forward_2 | aligned 969, failed 0
```

Nearly all other strides required only small heading corrections, usually a few degrees or less. The reported yaw is the **rotation applied to the original trajectory**, not a physical change in how the subject walked.

### 10.5 Why the raw GRF and marker arrays are not rotated here

The heading script transforms **only `qpos`/`qvel`** and copies the original GRF and marker channels unchanged. Those measurements therefore remain in the **original OpenSim laboratory frame**. For a future task comparing experimental marker positions to the aligned MyoLeg simulation, we must transform them using the OpenSim→MuJoCo basis change **and** the stride's yaw/XY-origin transform. Similarly, force vectors need the corresponding rotational basis transform if used in the aligned simulation frame.

This separation is intentional for traceability, but it means **do not compare stored original marker/GRF vectors directly with aligned MuJoCo world-frame values**.

The script records applied yaw, original heading, original starting XY, displacement, original-data provenance and a SHA-256 hash of the source reference in `heading_alignment` inside the new manifest. It refuses to realign an input whose manifest already indicates heading alignment.

---

## 11. Replay: how to interpret the colored events

`play_gait120_with_events.py` takes a single saved reference and plays it with MuJoCo's kinematic forward computation:

```python
data.qpos[:] = qpos[frame]
data.qvel[:] = qvel[frame]
mujoco.mj_forward(model, data)
```

It **does not call `mj_step`**, train any muscles, or compute physics-generated contact events. You are watching what the recorded joint configurations look like on the MyoLeg skeleton.

The event markers are rendered just above the corresponding model heel (`RCAL`/`LCAL`) or toe (`RTOE`/`LTOE`) site:

| Display | Interpretation |
|---|---|
| **Green** | Observed **upward 20 N GRF crossing**, labeled to a particular foot — still technically a candidate. |
| **Pink/magenta** | Observed **downward 20 N GRF crossing** (toe-off candidate). |
| **Amber** | Right HS inferred from the dataset stride boundary, or a left HS estimated from a half-stride offset. |

Foot-unassigned force crossings are printed in the terminal but are not drawn at an arbitrary foot. The viewer prints times/stride percentages, slows playback as requested, and temporarily holds at event times (`--hold`) for visual inspection. At a right-stride boundary, it does not draw an additional inferred HS if a measured, right-labeled crossing corroborates the boundary.

**Keyboard:** `Space` pause/resume; `R` restart; left/right arrows step through frames; `Q` quit. Mouse controls viewpoint. `--events-only` prints the annotation list without launching MuJoCo.

The two manually rejected HS rows are absent from the corrected `_events.csv` files, so they do **not** reappear as false green markers when using this original viewer on corrected references.

---

## 12. File contents and practical use in imitation learning

Each stride is stored under `Sxxx/Trialyy/stepzz` and produces:

```text
gait120_references_forward_2/
├── heading_alignment_summary.csv
├── S011/
│   └── Trial01/
│       ├── step01.npz
│       ├── step01_events.csv
│       └── step01_manifest.json
├── ...
├── S060/
│   └── Trial01/
│       ├── step01.npz
│       └── ...
└── S120/...
```

### 12.1 Key NPZ arrays

| Key | Content / units |
|---|---|
| `time_s` | Original absolute timestamps (s). |
| `time_since_start_s` | Original time minus first timestamp (s). |
| `stride_duration_s` | This stride's duration $T=t_1-t_0$ (s). |
| `qpos` | MuJoCo model configuration `[frames, 35]`: root XYZ in m, root quaternion **wxyz**, independent and dependent joint coordinates (mostly radians; slide joints in m). |
| `qvel` | Quaternion-aware MuJoCo generalized velocities `[frames, 34]`, using MuJoCo DOF ordering/units. |
| `phase_right_01`, `phase_right_rad` | Right-stride phase in 0–1 and 0–$2\pi$. |
| `phase_left_01`, `phase_left_rad` | Left phase using observed/selected HS when available, otherwise half-stride estimate. |
| `left_hs_s` | Selected or estimated left HS timestamp; **check manifest for provenance**. |
| `walking_speed_est_m_s` | Estimated horizontal pelvis **net-displacement magnitude / duration** (m/s). |
| `pelvis_horizontal_displacement_m` | Net horizontal pelvis displacement magnitude (m). |
| `original_joint_columns`, `original_joint_values` | Unmodified numeric OpenSim joint MOT table for traceability. |
| `grf_time_s`, `grf_channel_names`, `grf_values` | Original force-plate channels and their original times. |
| `marker_time_s`, `marker_names`, `marker_xyz_os_m` | Original observed markers in **OpenSim lab axes**, converted to metres. |

The estimated speed is not a direct treadmill measurement and is **not** an instantaneous speed profile. It is the magnitude of overall pelvis horizontal displacement divided by stride time; it deliberately does not report the direction sign.

### 12.2 Event CSV and manifest

- `_events.csv`: observed threshold transitions, force plate, event time, assigned foot (or `unassigned`), assignment method, geometric margins. In the two corrected steps, the known false displayed HS row has been removed.
- `_manifest.json`: provenance, stride boundaries, measured-vs-assumed left HS, observed TO candidate times, quaternion QC, knee-equality residuals, independent joint-range warnings, S011 flag and input file hashes. Aligned manifests also contain `heading_alignment` metadata.
- `batch_summary.csv`: overview of the extraction, including saved/existing/failed files. On incremental runs, `SKIPPED_EXISTS` means the file was already present; it does not mean an extraction failure.
- `heading_alignment_summary.csv`: every aligned reference and its yaw/displacement/status; your latest full run reports 969 aligned and 0 failed.

### 12.3 Which reference fields should the RL policy track?

For the initial **joint-based** tracking objective, load the **forward-aligned reference folder** and use the desired **independent leg joint positions and velocities**, indexed through the compiled MuJoCo model's `jnt_qposadr` and `jnt_dofadr`. Use stored time/phase to select or interpolate the appropriate reference frame, and retain the stride duration if the agent's clock needs it.

The available root trajectory may be used for initialization, optional global-heading tracking or reward design, but it is **not compulsory** if the experimental objective is joint-only tracking. A policy does **not** need to track recorded upper-body orientation if this is deliberately outside the evaluation scope.

Do not confuse nominal measured event candidates with *actual simulated* foot contacts. An online gait-phase estimator should ultimately use events from the simulated system, while reference phase supplies the tracking target.

---

## 13. Commands (Windows PowerShell)

Assuming your four Python scripts and the output folders are in the same working directory:

```powershell
$model  = 'C:\Users\BoussofaraS\OneDrive - University of Twente\Documents\myosuite\myosuite\simhive\myo_sim\leg\myolegs.xml'
$source = 'C:\Users\BoussofaraS\Downloads\Gait120_001_to_120'

# 1. Initial extraction (normally ALREADY DONE)
python batch_extract_gait120_quaternion_fixed.py `
  --model "$model" `
  --data-root "$source" `
  --output-root gait120_references_full

# 2. Fix only S029 Trial05 step01 and S115 Trial03 step02 (ALREADY DONE; run once)
python fix_two_gait120_steps.py --root gait120_references_full

# 3. Copy into consistently forward-moving references (ALREADY DONE)
python standardize_gait120_heading.py `
  --model "$model" `
  --input-root gait120_references_full `
  --output-root gait120_references_forward_2

# 4. View a step (example: previously reversed S060)
python play_gait120_with_events.py `
  --model "$model" `
  --reference 'gait120_references_forward_2\S060\Trial01\step02.npz' `
  --speed 0.5 --hold 0.8

# 5. Inspect events without a viewer
python play_gait120_with_events.py `
  --reference 'gait120_references_forward_2\S029\Trial05\step01.npz' `
  --events-only
```

**Do not re-run the two-step correction on already corrected data just to create another output copy.** Likewise, the alignment script correctly refuses to align an input twice, and original exports should be retained so the transformation remains reproducible.

### Python requirements

The supplied workflow requires `numpy`, `scipy`, `pandas` and `mujoco`; the viewer also imports `glfw` and `mujoco.viewer` and needs a working graphical environment. Run using the Python environment in which the MyoLeg XML and included assets load correctly.

**About your merged four-file version:** The original supplied exporter *still imported* `LEG_JOINTS`, `read_mot`, `read_trc`, `candidate_qpos` and `differentiate_qpos` from earlier diagnostic scripts. You said you moved these helper functions into your four main scripts. That is fine: just remove any leftover imports of those diagnostic modules. This README describes the original tested **logic**, but I have not inspected your newly merged local copy to verify that every imported symbol was adjusted.

---

## 14. Validation completed, and what this dataset does not prove

### Completed in this project

- Pelvis orientation/basis mapping examined against experimental marker motion.
- Global independent joint sign mapping tested across subjects; S011 subtalar caveat preserved as a flag.
- Dependent knee/patellar polynomial reconstruction and equality residual checks.
- Timestamp, units, missing-value, joint-range and quaternion-continuity checks.
- S060 representation-only quaternion issue fixed; all six affected exports recovered.
- Visual inspection of representative gait and targeted event anomalies.
- Known false left HS annotations in S029 and S115 corrected without modifying measured GRFs.
- **969/969 available reference strides successfully heading-aligned** in the supplied log.

### Not claimed / limitations

- **No dynamic feasibility guarantee:** kinematic replay does not show that MyoLeg muscles can reproduce the motion under the recorded forces or contacts.
- **No subject-specific musculoskeletal rescaling:** an individual's exact body proportions, joint centers and marker offsets may differ from the unscaled MyoLeg geometry; this pipeline intentionally does not edit the model anatomy.
- **Not all events are ground truth:** measured force crossings can be ambiguous, repeated or truncated; some left HS times are explicitly assumed and some TO events are unobserved.
- **No automatic individual-subject joint sign changes:** S011 is retained under the global mapping with a caution flag.
- **No force-frame conversion in the heading-aligned NPZ:** raw GRF/marker data still belong to the OpenSim laboratory frame.
- **No muscle activation reference:** the dataset's joint kinematics and event timing do not imply measured muscle excitations.

**Conclusion:** The folder `gait120_references_forward_2/` is complete for the **selected, successfully exported 969 LevelWalking strides**, and is ready to be used as a **kinematic imitation-learning reference dataset**, subject to the above scientific limitations.

---

## 15. Reference

Gait120 dataset article: [*Scientific Data* (2025), DOI 10.1038/s41597-025-05391-0](https://www.nature.com/articles/s41597-025-05391-0).

The detailed **mapping conventions, numerical tolerances, event fallbacks and correction cases documented above describe this project's scripts**. They should not automatically be attributed to the Gait120 authors.
