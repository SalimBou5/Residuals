"""CPU-only regression tests for feedforward + residual control and kinematic rewards.
Run: python test_skeleton.py
"""
import tempfile
from pathlib import Path
import numpy as np
import torch

from gait_phase import PhaseConfig
from gait_reference import CHANNELS, GaitReference
from gpu_components import GPUReference, GPUMuscleMapper, GPUPhaseEstimator
from rl_env import GaitRLEnv
from imitation_reward import ImitationRewardConfig
from muscle_mapping import RIGHT_MAPPING, MuscleMapper
from rsi import RSIConfig, RSISampler, PhysicalStateBuffer


def run():
    device = 'cpu'
    phase_samples = np.linspace(0.0, 1.0, 100, endpoint=False)
    # Non-constant to detect erroneous same-phase bilateral commands.
    profiles = np.empty((100, 15), dtype=float)
    for i in range(15):
        profiles[:, i] = 0.5 + 0.35 * np.sin(2 * np.pi * phase_samples + i * .09)
    q = np.stack((np.cos(2*np.pi*phase_samples), np.sin(2*np.pi*phase_samples)), axis=1)
    dq = np.stack((-2*np.pi*np.sin(2*np.pi*phase_samples),
                    2*np.pi*np.cos(2*np.pi*phase_samples)), axis=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'test_reference.npz'
        np.savez(path, phase=phase_samples, muscle_activations=profiles,
                 muscle_names=np.asarray(CHANNELS), joint_angles=q,
                 joint_velocities=dq, joint_names=np.asarray(['hip_flexion_r', 'knee_angle_r']))
        ref = GaitReference.from_npz(path)

    gpu_ref = GPUReference(ref, device)
    assert torch.allclose(gpu_ref.at_phase(torch.tensor([.1]))['muscles_15'],
                          torch.tensor(profiles[10:11], dtype=torch.float32), atol=1e-5)

    # Exercise all 80 MyoLeg muscle assignments, including PDF gaps and left leg.
    assert len(RIGHT_MAPPING) == 40
    names = tuple('robot/' + base for base in RIGHT_MAPPING) + tuple(
        'robot/' + base[:-1] + 'l' for base in RIGHT_MAPPING)
    mapper = GPUMuscleMapper(names, device)
    assert len(mapper.unresolved) == 0
    assert len(mapper.provisional) > 0
    r = gpu_ref.at_phase(torch.tensor([0.0, .1]))['muscles_15']
    l = gpu_ref.at_phase(torch.tensor([.5, .6]))['muscles_15']
    mapped = mapper.expand(r, l)
    assert mapped.shape == (2,80)
    assert torch.isfinite(mapped).all()
    assert torch.all((mapped >= 0) & (mapped <= 1))
    idx = {name.rsplit('/',1)[-1]: i for i,name in enumerate(names)}
    assert torch.allclose(mapped[:, idx['tibant_r']], r[:, CHANNELS.index('TA')])
    assert torch.allclose(mapped[:, idx['tibant_l']], l[:, CHANNELS.index('TA')])
    assert not torch.allclose(mapped[:, idx['tibant_r']], mapped[:, idx['tibant_l']])
    assert torch.allclose(mapped[:,idx['vasint_r']],
                          (r[:, CHANNELS.index('VastLat')] + r[:, CHANNELS.index('VastMed')]) / 2)
    assert torch.allclose(mapped[:,idx['addbrev_r']], r[:,CHANNELS.index('Add')])
    assert torch.allclose(mapped[:,idx['semiten_r']], r[:,CHANNELS.index('Semi')])
    assert MuscleMapper(names).expand(
        dict(zip(CHANNELS, profiles[0])), dict(zip(CHANNELS, profiles[50]))).shape == (80,)
    try:
        mapper.expand(r)
    except ValueError:
        pass
    else:
        raise AssertionError('Unshifted bilateral mapping should fail explicitly')
    try:
        GPUMuscleMapper(('robot/unknown_r',), 'cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown muscle must fail rather than quietly zero')

    cfg = PhaseConfig(reference_stride_s=1.0, toe_off_phase=0.62)
    estimator = GPUPhaseEstimator(cfg, 2, device)
    estimator.reset(initial_contact=torch.tensor([True, False]))
    hs, to = estimator.update(.01, torch.tensor([True, False]))
    assert not hs.any() and not to.any()
    for _ in range(4):
        for j in range(100):
            hs, to = estimator.update(.01, torch.tensor([j >= 60, False]))
    assert estimator.count[0].item() >= 2
    estimator.update(.01, torch.tensor([True,False]),done=torch.tensor([True,False]))
    assert estimator.phase[0].item() == 0

    class MockSim:
        num_envs, device, dt = 2, torch.device('cpu'), .01
        actuator_names = names
        def __init__(self):
            self.cmd = None
        def reset(self):
            return {'actor': torch.zeros(2,3)}, {}
        def get_foot_contact(self, side='r'):
            return torch.zeros(2, dtype=torch.bool)
        def get_joint_state(self, joint_names):
            assert joint_names == ('hip_flexion_r','knee_angle_r')
            return torch.ones((2,2)), torch.zeros((2,2))
        def get_muscle_activations(self):
            # Intentionally mismatched from EMG; must not affect imitation reward.
            return torch.full((2,80), 0.99)
        def step(self, cmd):
            self.cmd = cmd.clone()
            zeros = torch.zeros(2, dtype=torch.bool)
            return {'actor':torch.ones(2,3)}, torch.zeros(2), zeros, zeros, {}
        def close(self):
            pass

    sim = MockSim()
    env = GaitRLEnv(ref, cfg, sim=sim,
                    rsi_config=RSIConfig(mode='standard',initial_phase=.1))
    obs, info = env.reset()
    assert obs['actor'].shape == (2,3+2+1+80+2+2)
    before = env.reference_targets()
    base = before['feedforward_muscles_80']
    assert torch.isfinite(base).all()
    zeros = torch.zeros(2,80)
    assert torch.allclose(env.action_to_excitation(zeros,before), base.clamp(0,1))
    actions = torch.full((2,80), .3)
    actual = env.action_to_excitation(actions, before)
    expected = torch.clamp(base + env.residual_scale*torch.tanh(actions), 0, 1)
    assert torch.allclose(actual, expected)
    obs, reward, _term, _trunc, info = env.step(zeros)
    assert torch.allclose(sim.cmd, before['feedforward_muscles_80'].clamp(0,1))
    assert torch.allclose(info['policy_residual'], zeros)
    assert torch.allclose(info['feedforward_excitation'], base)
    assert reward.shape == (2,)
    assert set(info['imitation_terms']) == {
        'q_mse', 'q_reward','dq_mse','dq_reward','total_imitation_reward'}
    assert not any('muscle' in k for k in info['imitation_terms'])
    try:
        ImitationRewardConfig(muscle_weight=1)
    except TypeError:
        pass
    else:
        raise AssertionError('Muscle imitation reward must be removed')
    env.residual_scale = 50.0
    assert torch.all(env.action_to_excitation(torch.full((2,80),-50.0),before) == 0)
    assert torch.all(env.action_to_excitation(torch.full((2,80),50.0),before) == 1)

    # Physical-state RSI buffer and kinematic sampling are future-only.
    future = RSISampler(RSIConfig(mode='kinematic'), env.reference, 'cpu')
    sample = future.sample_kinematic(2)
    assert sample['q'].shape == (2,2)
    buffer = PhysicalStateBuffer(capacity=2)
    buffer.add_batch(torch.ones(3,5),torch.zeros(3,4),torch.zeros(3,80),
                     torch.zeros(3),torch.tensor([True,False,True]))
    assert len(buffer)==2
    env.rsi_config = RSIConfig(mode='physical')
    try:
        env.reset()
    except NotImplementedError:
        pass
    else:
        raise AssertionError('RSI restoration must not silently run')
    env.close()
    print('PASS: 80/80 feedforward mapping, phase-shifted left leg, residual-only actions, kinematic-only imitation reward, RSI staging')


if __name__ == '__main__':
    run()
