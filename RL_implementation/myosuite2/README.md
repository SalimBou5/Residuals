# MyoSuite / MyoLeg — gait-reference RL skeleton

This **wraps MyoSuite** `myoLegWalk-v0`. It is no longer an abstract simulator placeholder.
The RL observation/action/reward/paradigm are intentionally unimplemented because
we haven't designed them yet.

## 1. Environment

Use the conda environment where your MyoSuite is installed. A supported setup is:

```bash
pip install myosuite
python inspect_myosuite.py
```

This **constructs and resets a real MyoSuite MyoLeg model**, prints the actual
80 actuator names (in simulator order), reads the actual control timestep,
and identifies right-foot/ground contact geometries. Add `--step` for one
random **API smoke test**, not meaningful physiological control.

The integration is in `myosuite_adapter.py`:

```python
import myosuite
from myosuite.utils import gym
sim = gym.make('myoLegWalk-v0', normalize_act=False)
```

`normalize_act=False` is **important**: our activation profiles are in [0,1].
Some MyoSuite versions otherwise apply a nonlinear action transformation.
The adapter validates that the simulator action bounds match direct [0,1]
muscle commands. It supports 4- or 5-element Gym step returns and both
`mj_model`/`mj_data` and older `sim.model`/`sim.data` APIs.

## 2. Controller flow

At each RL control step:

1. **Gait phase** indexes the cycle-normalized reference joint trajectories
   and 15 activation profiles. This is a phase lookup, not resampling gait
   data into simulation time.
2. `MuscleMapper` expands the profiles into **the 80 actuator names and order
   obtained from the real MyoLeg model**, using the provisional PDF mapping.
3. When the final RL action is defined, combine it with reference activation:
   `u = clip(a_ref(phase) + delta_a, 0, 1)`.
4. **Real MyoSuite physics:** call `sim.step(u)`.
5. Extract right-foot/ground collision pairs from MuJoCo contact data.
   Contact edges define HS/TO, correcting the phase and stride estimate.
6. The next phase indexes the reference targets for the following step.

`GaitRLEnv.reset()` actually resets MyoSuite and initializes the phase.
`GaitRLEnv.step()` contains actual simulator wiring, but deliberately
stops until we define the incomplete mapping, RL action, observation and reward.
It will **not** falsely train on the built-in MyoSuite walking reward.

## 3. Reference input format (provisional, adapt to YOUR file layout)

An `.npz` file containing `phase` (shape `(N,)` values in `[0,1)`),
`muscle_activations` `(N,15)` and `muscle_names` `(15,)`, with optional
`joint_angles` and `joint_velocities` `(N,D)`, all indexed by the same gait
phase. The 15 accepted names are:

`TA SOL Per VastLat VastMed RFem Sar Add GlutMed TFL GastLat GastMed BFem Semi GlutMax`

The optional interpolation in `at_phase()` only retrieves values at an
estimated phase between the normalized gait samples (e.g. at 37.4%); it
**does not alter the cycle-normalized reference**.

Check that your activation/EMG profiles are suitable as muscle *excitations*
for MyoSuite; their meaning is not automatically identical.

## 4. Provisional mapping is still a blocker

`muscle_mapping.py` uses the right-leg mapping in your
*MYOLEG vs Gonzales-Vargas* PDF. It includes e.g. `bflh/bfsh -> BFem`,
`glmed1..3 -> GlutMed`, `glmin1..3 -> mean(GlutMed, GlutMax)`,
`vasint -> mean(VastLat,VastMed)`, `ehl -> TA`, `tibpost -> SOL`.

The PDF has **unresolved** adductors, toe muscles, hip flexors, and others.
Left-side mappings are not defined. Missing values remain `NaN`, are clearly
reported, and are NOT implicitly set to zero or silently filled with a
physiologically unjustified profile. Thus reference-based stepping remains
blocked until the full 80-muscle mapping is decided.

## 5. Tests and structure

```bash
python test_skeleton.py       # Data, phase, mapping; no MyoSuite needed
python inspect_myosuite.py    # Real environment; MyoSuite required
```

- `myosuite_adapter.py`: real MyoLeg simulator and contact reading.
- `gait_phase.py`: event-corrected phase and confidence.
- `gait_reference.py`: phase-normalized reference lookup.
- `muscle_mapping.py`: provisional 15-to-80 mapping.
- `rl_env.py`: RL wrapper with actual simulator calls and explicit RL TODOs.
- `inspect_myosuite.py`: simulator connection smoke check.
- `test_skeleton.py`: unit tests with synthetic gait data and a fake sim.

**Contact check:** automatic geom selection matches foot segment names such
as `talus_r`, `calcn_r`, `toes_r` and ground geoms such as `terrain`/`floor`.
Inspect `contact_geometry_report()` in your exact MyoSuite version; if the
model names differ, edit `_identify_contact_geoms` before using the phase
estimator. The contact logic does not use the reference gait to fake HS/TO.

**Validation limitation:** MyoSuite isn't installed in the file-generation
environment here. The standalone unit tests run, but the actual physics
smoke test must be run in your MyoSuite environment.
