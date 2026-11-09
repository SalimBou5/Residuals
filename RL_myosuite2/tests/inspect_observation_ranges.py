#!/usr/bin/env python3
"""Inspect the EXACT Step-4 (505-D) observation on recorded MyoLeg references.

This is an OFFLINE kinematic sweep: it sets recorded qpos/qvel and RSI lookup
activations at each selected phase, calls mj_forward, and builds the existing
ObservationBuilder output. No mj_step, PPO, or simulated balance is involved.

Outputs:
  features.csv       One row per observation dimension, with units and quantiles
  blocks.csv         Summary over features in each of the 10 observation blocks
  contact_forces.csv Raw normal-force distribution in NEWTONS, unscaled
  metadata.json      Scope, sampling, caveats, and skip counts

Usage (from the myoleg_rl_starter directory):
  python inspect_observation_ranges.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\\Users\\BoussofaraS\\Documents\\residuals\\RL_myosuite2\\data_refining\\gait_training_clean --mep "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\myoleg_bilateral_mep_80.csv" --lookup "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\activation_profiles\\activation_lookup.npz" --samples-per-reference 25

Use --samples-per-reference 0 to inspect EVERY recorded frame. This is more
expensive, but the script uses a bounded quantile reservoir rather than keeping
all observation vectors in memory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add the root 'RL_myosuite2' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from env  import MyoLegResidualEnv
from RSI.reference import ReferenceAssets
from observations import ObservationConfig, POSITION_SITES, ORIENTATION_SITES
from phase.phase import PhaseReset

class RunningFeatureStatistics:
    """Exact min/max/mean/std; approximate quantiles if count > reservoir_size."""

    def __init__(self, dimensions: int, reservoir_size: int = 40000, seed: int = 11):
        if dimensions < 1 or reservoir_size < 1:
            raise ValueError("dimensions and reservoir_size must be positive")
        self.dimension = dimensions
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.min = np.full(dimensions, np.inf)
        self.max = np.full(dimensions, -np.inf)
        self.sums = np.zeros(dimensions, dtype=np.float64)
        self.squared_sums = np.zeros(dimensions, dtype=np.float64)
        self.reservoir = np.empty((reservoir_size, dimensions), dtype=np.float32)

    def add(self, row: np.ndarray) -> int | None:
        x = np.asarray(row, dtype=np.float64)
        if x.shape != (self.dimension,) or not np.all(np.isfinite(x)):
            raise ValueError(f"Nonfinite / incorrectly shaped sample: {x.shape}")
        self.count += 1
        np.minimum(self.min, x, out=self.min)
        np.maximum(self.max, x, out=self.max)
        self.sums += x
        self.squared_sums += x * x
        if self.count <= self.reservoir_size:
            self.reservoir[self.count - 1] = x
            return self.count - 1
        else:
            pick = int(self.rng.integers(self.count))
            if pick < self.reservoir_size:
                self.reservoir[pick] = x
                return pick
        return None

    def finish(self) -> dict[str, np.ndarray]:
        if not self.count:
            raise RuntimeError("No samples collected; check filters and input paths")
        q = np.percentile(
            self.reservoir[:min(self.count, self.reservoir_size)],
            [1, 5, 50, 95, 99], axis=0,
        )
        mean = self.sums / self.count
        std = np.sqrt(np.maximum(0, self.squared_sums / self.count - mean ** 2))
        return dict(min=self.min, p01=q[0], p05=q[1], median=q[2],
                    p95=q[3], p99=q[4], max=self.max, mean=mean, std=std)


SITE_AXES = ('x', 'y', 'z')
QUAT_AXES = ('w', 'x', 'y', 'z')


def feature_names(model, mj, obs_builder) -> tuple[list[str], list[str]]:
    """Names and physical units in EXACT ObservationBuilder concatenation order."""
    # Joint data are qpos[7:] and qvel[6:] in this MyoLeg model.
    q_names = [f'qpos[{i}]' for i in range(7, model.nq)]
    v_names = [f'qvel[{i}]' for i in range(6, model.nv)]
    for jid in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid) or f'joint_{jid}'
        a = int(model.jnt_qposadr[jid]); b = int(model.jnt_dofadr[jid])
        for i in range(a, min(model.nq, a + (7 if int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE) else 1))):
            if i >= 7: q_names[i - 7] = name
        for i in range(b, min(model.nv, b + (6 if int(model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE) else 1))):
            if i >= 6: v_names[i - 6] = name

    def site_features(prefix):
        result = []
        for type_, sites, axes in (
            ('position', POSITION_SITES, SITE_AXES),
            ('linear_velocity', POSITION_SITES, SITE_AXES),
            ('orientation', ORIENTATION_SITES, QUAT_AXES),
            ('angular_velocity', ORIENTATION_SITES, SITE_AXES),
        ):
            result.extend((f'{prefix}/{type_}/{s}/{axis}',
                           'm' if type_ == 'position' else
                           'm/s' if type_ == 'linear_velocity' else
                           '1' if type_ == 'orientation' else 'rad/s')
                          for s in sites for axis in axes)
        return result

    names_and_units = {
        'joint_proprioception': (
            [(f'joint_position/{x}', 'rad') for x in q_names]
            + [(f'joint_velocity/{x}', 'rad/s') for x in v_names]),
        'root_state': [('pelvis_height', 'm')]
            + [(f'pelvis_quat/{x}', '1') for x in QUAT_AXES]
            + [(f'pelvis_local_linear_velocity/{x}', 'm/s') for x in SITE_AXES]
            + [(f'pelvis_local_angular_velocity/{x}', 'rad/s') for x in SITE_AXES],
        'foot_contacts': [(f'{n}/scaled', 'kN') for n in
                          ('right_heel','right_toe','left_heel','left_toe')],
        'muscle_activations': [(f'muscle_activation/{n}', '1') for n in obs_builder.observed_muscle_names],
        'actual_sites': site_features('actual'),
        'reference_sites': site_features('reference'),
        'reference_joints': [(f'reference_joint_position/{x}', 'rad') for x in q_names],
        'future_joints': [(f'future_{s:g}s_joint_position/{x}', 'rad')
                          for s in obs_builder.config.lookahead_seconds for x in q_names],
        'mep_baseline': [(f'mep_excitation/{n}', '1') for n in obs_builder.observed_muscle_names],
        'gait_phase': [(n, unit) for n, unit in (
            ('desired_forward_speed', 'm/s'), ('sin_phase', '1'), ('cos_phase', '1'),
            ('wrapped_phase_drift', 'cycles'), ('remaining_event_correction', 'cycles'),
            ('event_correction_active', 'bool'))],
    }
    flat = []
    for block, section in obs_builder.slices.items():
        items = names_and_units[block]
        if len(items) != section.stop-section.start:
            raise RuntimeError(f'Feature naming error for {block}: {len(items)} vs {section}')
        flat.extend(items)
    if len(flat) != obs_builder.dimension:
        raise RuntimeError('Feature names do not match observation dimension')
    return [n for n, _ in flat], [u for _, u in flat]


def phase_indices(length: int, target: int) -> np.ndarray:
    """Uniformly cover the CLOSED recorded cycle (including the last sample)."""
    if target < 0:
        raise ValueError('samples-per-reference must be nonnegative')
    if target == 0 or target >= length:
        return np.arange(length)
    if target == 1:
        return np.array([0], dtype=int)
    return np.unique(np.rint(np.linspace(0, length - 1, target)).astype(int))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--references', required=True, type=Path)
    p.add_argument('--mep', required=True, type=Path)
    p.add_argument('--lookup', required=True, type=Path)
    p.add_argument('--output', default=Path('observation_range_report'), type=Path)
    p.add_argument('--samples-per-reference', default=25, type=int,
                   help='Evenly spaced recorded samples per cycle; 0 means ALL frames (default 25)')
    p.add_argument('--max-references', default=0, type=int,
                   help='Number of references to inspect; 0 means all (default 0)')
    p.add_argument('--max-penetration-mm', default=None, type=float,
                   help='Optional filter: skip replay poses penetrating more than this many mm')
    p.add_argument('--reservoir-size', default=40000, type=int,
                   help='Max sample count for estimated percentiles (exact min/max/mean/std)')
    p.add_argument('--progress-every', default=100, type=int)
    p.add_argument('--enable-unmapped', action='store_true',
                   help='Inspect full 505-D observation; default mapped-only 485-D')
    p.add_argument('--plots', action='store_true',
                   help='Save sampled observation matrix, outlier provenance, and distribution plots')
    p.add_argument('--plot-top', default=24, type=int,
                   help='Number of most suspicious non-duplicate kinematic features to plot (default 24)')
    args = p.parse_args()
    if args.samples_per_reference < 0 or args.max_references < 0 or args.reservoir_size < 1:
        p.error('samples-per-reference/max-references must be >= 0; reservoir-size >= 1')
    if args.max_penetration_mm is not None and args.max_penetration_mm < 0:
        p.error('max-penetration-mm must be >= 0')
    if args.progress_every < 1:
        p.error('progress-every must be >= 1')

    import mujoco as mj
    assets = ReferenceAssets(args.references, args.mep, args.lookup)
    from control import ActionConfig
    env = MyoLegResidualEnv(args.model, assets,
                           action_config=ActionConfig(enable_unmapped=args.enable_unmapped),
                           observation_config=ObservationConfig(contact_force_scale_n=1000.0),
                           event_sync=False)
    try:
        model, data, builder, clock = env.model, env.rsi.data, env.observations, env.core.clock
        names, units = feature_names(model, mj, builder)
        obs_stats = RunningFeatureStatistics(builder.dimension, args.reservoir_size)
        force_stats = RunningFeatureStatistics(4, args.reservoir_size, seed=13)
        sample_refs = np.empty(args.reservoir_size, dtype=np.int32) if args.plots else None
        sample_phases = np.empty(args.reservoir_size, dtype=np.float32) if args.plots else None
        sample_penetrations = np.empty(args.reservoir_size, dtype=np.float32) if args.plots else None
        min_ref_idx = np.full(builder.dimension, -1, dtype=np.int32)
        max_ref_idx = np.full(builder.dimension, -1, dtype=np.int32)
        min_ref_phase = np.full(builder.dimension, np.nan, dtype=float)
        max_ref_phase = np.full(builder.dimension, np.nan, dtype=float)
        min_ref_pen = np.full(builder.dimension, np.nan, dtype=float)
        max_ref_pen = np.full(builder.dimension, np.nan, dtype=float)
        all_paths = assets.library.paths
        paths = all_paths[:args.max_references] if args.max_references else all_paths
        skipped_penetration = 0
        skipped_invalid = 0
        used_references = 0
        total_candidates = 0
        max_pen_seen_m = 0.0
        floor_id = env.rsi.floor_geom_id
        for ref_num, path in enumerate(paths, 1):
            ref = assets.library.get(path)
            builder.set_reference(ref)
            clock.reset(PhaseReset(0., 0.), ref.duration, ref.events)
            lookup_profile, matched_duration = assets.lookup.for_duration(ref.duration)
            selected = phase_indices(ref.n_samples, args.samples_per_reference)
            used_here = 0
            # Each sample is a standalone imposed posture, NOT an integrated rollout.
            for index in selected:
                total_candidates += 1
                theta = float(ref.phase_grid[index])
                # The saved endpoint phi=1 is equivalent to the next cycle's phi=0.
                sample = ref.sample(theta)
                mj.mj_resetData(model, data)
                data.qpos[:] = sample.qpos
                data.qvel[:] = sample.qvel
                env.rsi.binding.init_activations(data, lookup_profile.sample(theta),
                                                  enable_unmapped=args.enable_unmapped)
                data.ctrl[:] = env.rsi.binding.baseline(theta)
                clock.theta = theta
                clock.theta_nominal = theta
                clock.phase_rate_hz = 1.0 / ref.duration
                clock.remaining_error = 0.0
                clock.active = False
                mj.mj_forward(model, data)

                max_pen = 0.0
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    if floor_id in (int(c.geom1), int(c.geom2)):
                        max_pen = max(max_pen, -float(c.dist))
                max_pen_seen_m = max(max_pen_seen_m, max_pen)
                if (args.max_penetration_mm is not None and
                        max_pen * 1000. > args.max_penetration_mm):
                    skipped_penetration += 1
                    continue
                forces = env.reader(model, data)
                try:
                    obs = builder.build(data, clock, forces)
                    new_min = obs < obs_stats.min
                    new_max = obs > obs_stats.max
                    ridx = ref_num - 1
                    min_ref_idx[new_min] = ridx
                    max_ref_idx[new_max] = ridx
                    min_ref_phase[new_min] = theta
                    max_ref_phase[new_max] = theta
                    min_ref_pen[new_min] = max_pen * 1000
                    max_ref_pen[new_max] = max_pen * 1000
                    slot = obs_stats.add(obs)
                    force_stats.add(forces)
                    if args.plots and slot is not None:
                        sample_refs[slot] = ridx
                        sample_phases[slot] = theta
                        sample_penetrations[slot] = max_pen * 1000
                except (ValueError, FloatingPointError):
                    skipped_invalid += 1
                    continue
                used_here += 1
            if used_here:
                used_references += 1
            if ref_num % args.progress_every == 0 or ref_num == len(paths):
                print(f'[{ref_num}/{len(paths)} references] scanned={total_candidates}, '
                      f'accepted={obs_stats.count}, penetration_skipped={skipped_penetration}, '
                      f'invalid_skipped={skipped_invalid}', flush=True)
        s = obs_stats.finish()
        f = force_stats.finish()
        rows = []
        for block, sel in builder.slices.items():
            for i in range(sel.start, sel.stop):
                # This scale is a *diagnostic candidate*, not applied to the policy.
                half_width = max((s['p99'][i] - s['p01'][i]) / 2, 0.)
                constant = bool(s['max'][i] - s['min'][i] < 1e-7)
                rows.append({
                    'index': i, 'block': block, 'feature': names[i], 'unit': units[i],
                    **{key: float(s[key][i]) for key in
                       ('min','p01','p05','median','p95','p99','max','mean','std')},
                    'is_constant_in_replay': constant,
                    'candidate_center_median': float(s['median'][i]) if not constant else '',
                    'candidate_half_p01_p99_range': float(half_width) if not constant else '',
                    'min_reference': str(paths[min_ref_idx[i]].relative_to(args.references)),
                    'min_phase': float(min_ref_phase[i]),
                    'min_penetration_mm': float(min_ref_pen[i]),
                    'max_reference': str(paths[max_ref_idx[i]].relative_to(args.references)),
                    'max_phase': float(max_ref_phase[i]),
                    'max_penetration_mm': float(max_ref_pen[i]),
                })
        blocks = []
        for block, sl in builder.slices.items():
            blocks.append({
                'block':block, 'dimensions':sl.stop-sl.start,
                'smallest_feature_min': float(np.min(s['min'][sl])),
                'largest_feature_max': float(np.max(s['max'][sl])),
                'largest_feature_std': float(np.max(s['std'][sl])),
                'constant_features_in_replay': int(sum(r['is_constant_in_replay']
                                                       for r in rows[sl.start:sl.stop])),
            })
        forces = []
        for i, name in enumerate(('r_foot','r_toes','l_foot','l_toes')):
            forces.append({'sensor': name, 'unit': 'N',
                           **{key: float(f[key][i]) for key in
                              ('min','p01','p05','median','p95','p99','max','mean','std')}})
        args.output.mkdir(parents=True, exist_ok=True)
        write_csv(args.output/'features.csv', rows)
        write_csv(args.output/'blocks.csv', blocks)
        write_csv(args.output/'contact_forces.csv', forces)
        metadata = {
            'model':str(args.model), 'reference_library':str(args.references),
            'references_indexed':len(all_paths), 'references_inspected':len(paths),
            'references_with_accepted_samples':used_references,
            'samples_per_reference_requested':args.samples_per_reference,
            'candidate_poses':total_candidates, 'accepted_poses':obs_stats.count,
            'penetration_skipped':skipped_penetration,'invalid_skipped':skipped_invalid,
            'maximum_penetration_encountered_mm':max_pen_seen_m*1000.,
            'max_penetration_filter_mm':args.max_penetration_mm,
            'observation_dimensions':builder.dimension,
            'enable_unmapped': bool(args.enable_unmapped),
            'observed_muscles': list(builder.observed_muscle_names),
            'normalization_already_applied': {'foot_contacts':'divided by 1000 N in Step 4'},
            'percentile_sample_size': min(obs_stats.count,args.reservoir_size),
            'percentiles_exact': obs_stats.count <= args.reservoir_size,
            'inspection_method':'Imposed reference qpos/qvel + periodic RSI activation lookup, mj_forward, no mj_step',
            'limitations':[
                'Touch sensor forces at imposed reference poses are NOT dynamic ground reaction forces; penetration can generate huge spikes. Do not choose final contact normalization from this replay alone.',
                'Actual proprioception and root-relative landmarks are evaluated from MyoLeg FK/Jacobians.',
                'Reference landmarks/joints, MEP, nominal phase and lookup activations are representative of the input library.',
                'Phase drift, pending event correction and active correction flag are identically zero in replay. Their real ranges require a dynamic rollout.',
                'The phase-rate correction is zero; target velocities use nominal rate 1/T, not the later learned adaptive rate.',
                'No policy perturbations, falls, real activation dynamics or tracking errors are represented.',
                'Median/half-percentile-range are inspection candidates ONLY; NO observation normalization has been changed.',
            ],
        }
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        if args.plots:
            n = min(obs_stats.count, obs_stats.reservoir_size)
            np.savez_compressed(
                args.output / 'observation_samples.npz',
                observations=obs_stats.reservoir[:n],
                feature_names=np.asarray(names),
                units=np.asarray(units),
                reference_ids=sample_refs[:n],
                reference_paths=np.asarray([str(p.relative_to(args.references)) for p in paths]),
                phases=sample_phases[:n],
                penetration_mm=sample_penetrations[:n],
            )
            from RL_myosuite2.tests.plot_observation_distributions import plot_distributions
            plot_distributions(args.output / 'observation_samples.npz', args.output / 'distributions',
                               top_n=args.plot_top)
        print(f'\nSaved {args.output.resolve()}')
        print('  features.csv: per-dimension min/p01/p05/median/p95/p99/max/mean/std')
        print('  blocks.csv: per-block summary')
        print('  contact_forces.csv: raw Newton readings (not validated GRFs)')
        print('  metadata.json: replay limitations, sample and filter counts')
        if args.plots:
            print('  observation_samples.npz: retained 485/505-D sampled observations and provenance')
            print('  distributions/: selected histograms and ranked outlier list')
        print('\nBlock summary:')
        for row in blocks:
            print(f"  {row['block']:<23} n={row['dimensions']:>3} "
                  f"range=[{row['smallest_feature_min']:+.3g}, "
                  f"{row['largest_feature_max']:+.3g}] "
                  f"constant_features={row['constant_features_in_replay']}")
        print('\nNote: no normalization has been applied; actual dynamic ranges must be checked later.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
