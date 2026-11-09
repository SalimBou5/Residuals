#!/usr/bin/env python3
"""Step-3 headless smoke test: 81-D scripted actions, actual MyoLeg physics.

No PPO, no reward, no full observation yet. Use --list-sensors to find the
four real touch sensor names before enabling event-driven phase correction.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

# Add the root 'RL_myosuite2' directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now use absolute imports from the root directory
from RSI.reference import ReferenceAssets
from RSI.rsi import RSIConfig, RSIInitializer

from phase.phase import PhaseConfig
from phase.contacts import FourTouchSensorReader, ContactConfig
from control import ActionConfig
from rollout import RolloutConfig
from env import PhysicsControlLoop

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--references', type=Path)
    p.add_argument('--mep', type=Path)
    p.add_argument('--lookup', type=Path)
    p.add_argument('--list-sensors', action='store_true')
    p.add_argument('--contact-sensors', nargs=4, metavar=('RH', 'RT', 'LH', 'LT'),
                   help='Four VERIFIED touch sensor names: right heel/toe, left heel/toe')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--print-every', type=int, default=20)
    p.add_argument('--phase', type=float)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--residual-limit', type=float, default=0.25,
                   help='Experimental calibration setting, NOT a validated optimum')
    p.add_argument('--muscle-action', type=float, default=0.0,
                   help='Constant mapped-muscle action in [-1,1] for smoke testing')
    p.add_argument('--phase-action', type=float, default=0.0)
    p.add_argument('--enable-unmapped', action='store_true')
    p.add_argument('--use-assumed-toe-off', action='store_true')
    p.add_argument('--physics-dt', type=float, default=0.001)
    p.add_argument('--frame-skip', type=int, default=5)
    p.add_argument('--contact-on-n', type=float, default=20.0)
    p.add_argument('--contact-off-n', type=float, default=10.0)
    p.add_argument('--cache-size', type=int, default=None,
                   help='Optional LRU capacity; omit for all references preloaded')
    args = p.parse_args()

    import mujoco as mj
    model = (mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower() == '.mjb'
             else mj.MjModel.from_xml_path(str(args.model)))
    if args.list_sensors:
        print(f'Model has {model.nsensor} sensors:')
        for sid in range(model.nsensor):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, sid)
            print(f'  {name}: type={int(model.sensor_type[sid])} '
                  f'dim={int(model.sensor_dim[sid])}')
        return
    if not all([args.references, args.mep, args.lookup]):
        p.error('Provide --references, --mep and --lookup unless using --list-sensors')
    if args.steps < 1 or args.print_every < 1 or abs(args.phase_action) > 1 or abs(args.muscle_action) > 1:
        p.error('Invalid number of steps or scripted action')
    assets = ReferenceAssets(args.references, args.mep, args.lookup,
                             max_cached_references=args.cache_size)
    reader = FourTouchSensorReader(model, mj, args.contact_sensors) if args.contact_sensors else None
    action_conf = ActionConfig(residual_limit=args.residual_limit,
                               enable_unmapped=args.enable_unmapped)
    rsi = RSIInitializer(model, mj, assets, seed=args.seed, contact_reader=reader,
                         config=RSIConfig(initialize_ctrl='mep',
                                          enable_unmapped=args.enable_unmapped))
    core = PhysicsControlLoop(
        rsi, contact_reader=reader, action_config=action_conf,
        rollout_config=RolloutConfig(physics_dt_s=args.physics_dt,
                                     frame_skip=args.frame_skip),
        phase_config=PhaseConfig(use_assumed_toe_off=args.use_assumed_toe_off),
        contact_config=ContactConfig(on_threshold=args.contact_on_n,
                                     off_threshold=args.contact_off_n))
    reset = core.reset(phase=args.phase)
    print(f'Model dt={model.opt.timestep:.4f}s, policy dt={core.config.control_dt_s:.4f}s; '
          f'preloaded={assets.library.n_cached}/{len(assets.library.paths)}')
    print(f'RSI: {reset.reference.path}, phi={core.clock.phase:.4f}, '
          f'T={reset.reference.duration:.3f}s, penetration={reset.max_floor_penetration_m*1000:.2f}mm')
    if reader is None:
        print('WARNING: no verified touch sensors supplied -> event synchronization DISABLED')
    action = np.zeros(core.actions.action_dim)
    action[:-1] = args.muscle_action
    action[-1] = args.phase_action
    print(f"Policy action dimension={core.actions.action_dim} "
          f"({core.actions.muscle_action_dim} muscles + 1 phase); "
          f"unmapped={'ON' if args.enable_unmapped else 'OFF'}; "
          f"contact thresholds ON={args.contact_on_n:g} N, OFF={args.contact_off_n:g} N")
    for k in range(1, args.steps+1):
        result = core.step(action)
        if result['events']:
            for event in result['events']:
                print('EVENT', event)
        if k % args.print_every == 0 or k == 1:
            print(f'k={k:04d} t={result["time_s"]:.3f}s '
                  f'phi={result["phase"]:.4f} theta={result["theta"]:.4f} '
                  f'drift={result["phase_drift"]:+.4f} '
                  f'pending={result["event_remaining"]:+.4f} '
                  f'ctrl=[{rsi.data.ctrl.min():.3f},{rsi.data.ctrl.max():.3f}] '
                  f'act=[{rsi.data.act.min():.3f},{rsi.data.act.max():.3f}]')
    print('Finished scripted rollout: no learned controller; falling is expected with MEP alone.')


if __name__ == '__main__':
    main()
