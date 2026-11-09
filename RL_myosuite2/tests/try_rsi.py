#!/usr/bin/env python3
"""Headless smoke check of loader + RSI using the user's real MyoLeg model.

This is intentionally NOT a policy rollout. Optional MEP-only integration
validates the reset-to-excitation handoff; no RL policy is involved.
"""
import argparse
from pathlib import Path
import numpy as np

from RSI.reference import ReferenceAssets
from RSI.rsi import RSIConfig, RSIInitializer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path,
                        help="Directory containing S###/Trial##/*_loopable.npz or one NPZ")
    parser.add_argument("--mep", required=True, type=Path)
    parser.add_argument("--lookup", required=True, type=Path)
    parser.add_argument("--resets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--phase", type=float, default=None)
    parser.add_argument("--enable-unmapped", action="store_true", default=False)
    parser.add_argument("--initialize-ctrl", choices=["zero", "mep"], default="zero")
    parser.add_argument("--baseline-seconds", type=float, default=0.0,
                        help="Optional MEP-only physics rollout after each RSI (no RL)")
    parser.add_argument("--cache-mode", choices=("all", "lru"), default="all",
                        help="Preload all references once (default), or bound per-process RAM")
    parser.add_argument("--cache-size", type=int, default=8,
                        help="Number of references in LRU mode (only for --cache-mode lru)")
    parser.add_argument("--max-penetration", type=float, default=0.015,
                        help="RSI acceptance limit, metres")
    args = parser.parse_args()
    if args.baseline_seconds < 0:
        parser.error("--baseline-seconds must be nonnegative")
    if args.cache_size < 1:
        parser.error("--cache-size must be >= 1")

    import mujoco as mj
    model = (mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix == ".mjb"
             else mj.MjModel.from_xml_path(str(args.model)))
    assets = ReferenceAssets(
        args.references, args.mep, args.lookup,
        max_cached_references=(None if args.cache_mode == "all" else args.cache_size),
    )
    rsi = RSIInitializer(model, mj, assets,
                         config=RSIConfig(max_floor_penetration_m=args.max_penetration,
                                          enable_unmapped=args.enable_unmapped,
                                          initialize_ctrl=args.initialize_ctrl),
                         seed=args.seed)
    print(f"Physics timestep from model: {model.opt.timestep:.6f}s; "
          "no model parameters have been changed")
    print(f"Indexed {len(assets.library.paths)} references, "
          f"{len(assets.library.groups)} subjects; model nq={model.nq}, "
          f"nv={model.nv}, nu={model.nu}, na={model.na}")
    print(f"Reference RAM cache: {assets.library.n_cached}/{len(assets.library.paths)} loaded; "
          f"numeric arrays={assets.library.cached_array_bytes / 1024**2:.1f} MiB "
          f"(excluding metadata/Python overhead); mode={args.cache_mode}")
    for i in range(args.resets):
        result = rsi.reset(phase=args.phase)
        u = rsi.apply_baseline(result.phase_state.theta)  # Initial command only; no step.
        print(f"{i + 1:3d}: {result.reference.path.name} | "
              f"T={result.reference.duration:.3f}s N={result.reference.n_samples} "
              f"phase={result.phase_state.phase:.4f} "
              f"lookupT={result.matched_activation_duration_s:.3f}s "
              f"penetration={1000 * result.max_floor_penetration_m:.2f}mm "
              f"attempts={result.phase_attempts} "
              f"u=[{np.min(u):.3f}, {np.max(u):.3f}] "
              f"act=[{np.min(rsi.data.act):.3f}, {np.max(rsi.data.act):.3f}] "
              f"events={result.reference.events.phases}")
        if not args.enable_unmapped:
            assert np.all(u[rsi.binding.unmapped_mask] == 0)
        if args.baseline_seconds > 0:
            # Matches the validated standalone test: MEP evaluated before EACH
            # physics integration step at phi0 + simulated_time / T_ref.
            # It is only a diagnostic; the later RL loop uses a phase clock.
            initial_phase = result.phase_state.theta
            duration = result.reference.duration
            while rsi.data.time < args.baseline_seconds - 1e-10:
                theta = initial_phase + float(rsi.data.time) / duration
                rsi.apply_baseline(theta)
                mj.mj_step(model, rsi.data)
                if not np.all(np.isfinite(rsi.data.qpos)) or not np.all(np.isfinite(rsi.data.qvel)):
                    raise FloatingPointError("Physics produced nonfinite kinematic state")
            print(f"      MEP-only rollout ended at t={rsi.data.time:.3f}s, "
                  f"final root xyz={np.asarray(rsi.data.qpos[:3])}")


if __name__ == "__main__":
    main()
