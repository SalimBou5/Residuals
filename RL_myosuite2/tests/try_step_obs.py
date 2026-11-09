#!/usr/bin/env python3
"""Step-4 smoke test: 485-D mapped-only or 505-D full MyoLeg observations.

Runs one random RSI and a few scripted 200-Hz steps; prints named observation
ranges and reset-state site parity checks. No learned controller, no rewards.

Example (PowerShell; one line):
python try_step4.py --model "C:\\path\\myolegs.xml" --references "C:\\path\\gait_with_landmarks" --mep "C:\\path\\myoleg_bilateral_mep_80.csv" --lookup "C:\\path\\activation_lookup.npz" --steps 5 --no-event-sync
"""
from pathlib import Path
import argparse
import numpy as np

import sys
# Add the root 'RL_myosuite2' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now use absolute imports from the root directory
from RSI.reference import ReferenceAssets
from control import ActionConfig
from observations import ObservationConfig, POSITION_SITES, ORIENTATION_SITES
from env import MyoLegResidualEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--mep', type=Path, required=True)
    p.add_argument('--lookup', type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--reference', type=Path, default=None)
    p.add_argument('--phase', type=float, default=None)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--enable-unmapped', action='store_true')
    p.add_argument('--no-event-sync', action='store_true')
    p.add_argument('--normalization', type=Path, default=None,
                   help='Fixed normalization JSON fitted from refined reference features')
    p.add_argument('--cache-size', type=int, default=None,
                   help='Optional limited LRU; default preloads all 969 references')
    args = p.parse_args()
    if args.steps < 0:
        p.error('--steps must be nonnegative')
    assets = ReferenceAssets(args.references, args.mep, args.lookup,
                             max_cached_references=args.cache_size)
    env = MyoLegResidualEnv(
        args.model, assets,
        action_config=ActionConfig(enable_unmapped=args.enable_unmapped),
        observation_config=ObservationConfig(
            normalization_file=str(args.normalization) if args.normalization else None),
        event_sync=not args.no_event_sync, seed=args.seed,
    )
    try:
        options={}
        if args.reference is not None:
            options['reference'] = args.reference
        if args.phase is not None:
            options['phase'] = args.phase
        obs,info=env.reset(seed=args.seed, options=options)
        print(f"Model physics dt={env.model.opt.timestep:.4f}s, "
              f"policy dt={env.core.config.control_dt_s:.4f}s")
        print(f"Loaded={assets.library.n_cached}/{len(assets.library.paths)} references")
        print(f"RSI reference={info['reference']}")
        print(f"RSI phase={info['reset_phase']:.4f}, penetration="
              f"{info['max_reset_penetration_m']*1000:.2f} mm")
        print(f"Normalization: {args.normalization if args.normalization else 'OFF (raw force observations)'}")
        print(f"Action dimension={env.action_space.shape[0]}, "
              f"observation dimension={env.observation_space.shape[0]}")
        for name,s in env.observation_slices.items():
            x=obs[s]
            print(f"  {name:<23s} [{s.start:3d}:{s.stop:3d}] "
                  f"n={len(x):3d} min={x.min():+9.4f} max={x.max():+9.4f}")
        print(f"Position sites (six): {', '.join(POSITION_SITES)}")
        print(f"Orientation/angvel sites (eight): {', '.join(ORIENTATION_SITES)}")

        # The initial RSI pose is from the same reference/model as our online
        # FK. These errors should be small; prints diagnostics, does NOT assert
        # an unrealistically zero velocity mismatch from retargeted finite differences.
        # Use RAW blocks for parity checks, even if the returned observation is normalized.
        a=env.observations.last_blocks['actual_sites'].astype(float)
        b=env.observations.last_blocks['reference_sites'].astype(float)
        for name,start,stop in (
                ('site position [m]',0,18),('site linear vel [m/s]',18,36),
                ('site quaternion components',36,68),('site angular vel [rad/s]',68,92)):
            delta=a[start:stop]-b[start:stop]
            print(f"  RSI parity {name}: RMS={np.sqrt(np.mean(delta**2)):.6f}, "
                  f"max_abs={np.max(np.abs(delta)):.6f}")
        print('Note: site orientations are sign-aligned to the reference quaternion.')

        # In full-action mode -1 means zero excitation for unmapped muscles;
        # default 71-D mode already enforces this without policy coordinates.
        action=np.zeros(env.action_space.shape[0],dtype=np.float32)
        if args.enable_unmapped:
            ids=env.core.actions.policy_to_actuator_ids
            action[:-1][env.rsi.binding.unmapped_mask[ids]]=-1.0
        for k in range(args.steps):
            obs,reward,terminated,truncated,info=env.step(action)
            print(f"step={k+1:02d} t={info['time_s']:.3f}s "
                  f"phi={info['phase']:.5f} drift={info['phase_drift']:+.5f} "
                  f"contacts={np.round(obs[env.observation_slices['foot_contacts']],3).tolist()} "
                  f"obs_finite={bool(np.all(np.isfinite(obs)))}")
            if terminated or truncated:
                break
        if env.observations.normalizer is not None:
            n=env.observations.normalizer
            print(f'Normalization saturation: {n.total_clipped}/{n.calls*n.dimension} '
                  f'features ({n.saturation_fraction:.5%}), last clipped={n.last_clipped_count}')
        print('Step 4 smoke test complete. Reward intentionally zero until Step 5.')
    finally:
        env.close()


if __name__=='__main__':
    main()
