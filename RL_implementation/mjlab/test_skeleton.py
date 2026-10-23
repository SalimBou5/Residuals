"""CPU/PyTorch tests, no MyoSuite install or NVIDIA GPU needed.

Run: python test_skeleton.py
"""
from pathlib import Path
import tempfile
import numpy as np
import torch

from gait_phase import PhaseConfig, GaitPhaseEstimator, FootContactEvents
from gait_reference import CHANNELS, GaitReference
from muscle_mapping import RIGHT_MAPPING, MuscleMapper
from rsi import RSIConfig, RSISampler, PhysicalStateBuffer
from imitation_reward import ImitationRewardConfig
from rl_env import GaitRLEnv


def run():
    device = 'cpu'
    p = np.linspace(0, 1, 100, endpoint=False)
    profiles = np.stack([0.5 + 0.35*np.sin(2*np.pi*p + j*0.09)
                         for j in range(15)], axis=1)
    q = np.stack([np.cos(2*np.pi*p), np.sin(2*np.pi*p)], axis=1)
    dq = np.stack([-2*np.pi*np.sin(2*np.pi*p),
                    2*np.pi*np.cos(2*np.pi*p)], axis=1)
    with tempfile.TemporaryDirectory() as folder:
        filename = Path(folder)/'reference.npz'
        np.savez(filename, phase=p, muscle_activations=profiles,
                 muscle_names=np.array(CHANNELS), joint_angles=q,
                 joint_velocities=dq,
                 joint_names=np.array(['hip_flexion_r','knee_angle_r']))
        ref = GaitReference.from_npz(filename)

    # User's scalar APIs must remain available, numerically consistent.
    assert abs(ref.at_phase(.1)['muscles_15']['TA'] - profiles[10,0]) < 1e-6
    assert ref.joint_names == ('hip_flexion_r', 'knee_angle_r')
    ref.to(device)
    torch_ref = ref.at_phase(torch.tensor([0.1, 0.99, -0.01]))
    assert torch_ref['muscles_15'].shape == (3,15)
    assert torch.allclose(torch_ref['muscles_15'][0],
                          torch.tensor(profiles[10],dtype=torch.float32),atol=1e-5)
    assert torch.allclose(torch_ref['muscles_15'][1], torch_ref['muscles_15'][2])
    # Scalar and batched reference queries use one interpolation calculation.
    phases = torch.tensor([0.0, .005, .37, .995, 1.01, -.01], dtype=torch.float32)
    vector = ref.at_phase(phases)
    for i, phi in enumerate(phases.tolist()):
        single = ref.at_phase(phi)
        assert np.allclose(vector['muscles_15'][i].numpy(),
                           [single['muscles_15'][ch] for ch in CHANNELS], atol=2e-6)
        assert np.allclose(vector['joint_angles'][i].numpy(), single['joint_angles'], atol=2e-6)
        assert np.allclose(vector['joint_velocities'][i].numpy(), single['joint_velocities'], atol=2e-6)

    assert len(RIGHT_MAPPING) == 40
    names = tuple('robot/' + muscle for muscle in RIGHT_MAPPING)
    names += tuple('robot/' + muscle[:-1] + 'l' for muscle in RIGHT_MAPPING)
    mapper = MuscleMapper(names, device=device)
    assert len(mapper.provisional) > 0 and not mapper.unresolved
    right = ref.at_phase(torch.tensor([0.0,.1]))['muscles_15']
    left = ref.at_phase(torch.tensor([0.5,.6]))['muscles_15']
    mapped = mapper.expand(right, left)
    assert mapped.shape == (2,80)
    assert (mapped >= 0).all() and (mapped <= 1).all()
    indices = {n.rsplit('/',1)[-1]:i for i,n in enumerate(names)}
    assert torch.allclose(mapped[:,indices['tibant_r']],right[:,CHANNELS.index('TA')])
    assert torch.allclose(mapped[:,indices['tibant_l']],left[:,CHANNELS.index('TA')])
    assert torch.allclose(mapped[:,indices['vasint_r']],
        (right[:,CHANNELS.index('VastLat')]+right[:,CHANNELS.index('VastMed')])/2)
    assert np.allclose(mapper.expand(dict(zip(CHANNELS,profiles[0])),
                dict(zip(CHANNELS,profiles[50]))), mapped[0])

    # The exact Gait120 threshold: upward 19.9 -> 20.0 = HS;
    # downward 20.0 -> 19.9 = TO, and the first sample is never an event.
    event = FootContactEvents(threshold_N=20.0)
    assert event.update(25.0) == (False,False)
    assert event.update(19.9) == (False,True)
    assert event.update(20.0) == (True,False)
    assert event.update(20.0) == (False,False)
    assert event.update(19.9) == (False,True)

    batched_event = FootContactEvents(threshold_N=20.0, num_envs=2, device=device)
    hs, to = batched_event.update(torch.tensor([10.0,30.0]))
    assert not hs.any() and not to.any()
    hs, to = batched_event.update(torch.tensor([20.0,19.0]))
    assert hs.tolist() == [True,False] and to.tolist() == [False,True]
    hs, to = batched_event.update(torch.tensor([22.0,20.0]))
    assert hs.tolist() == [False,True] and not to.any()
    batched_event.reset(mask=torch.tensor([True,False]),
                        initial_grf_N=torch.tensor([22.0,20.0]))
    hs, to = batched_event.update(torch.tensor([22.0,19.9]))
    assert not hs.any() and to.tolist() == [False,True]

    # Verify the MyoSuite adapter sums the *measured* heel/toe sensor data.
    # This is an API-level mock; physical correctness on real GPU is unverified.
    from types import SimpleNamespace
    from myosuite3_adapter import MyoSuite3LegAdapter
    adapter = MyoSuite3LegAdapter.__new__(MyoSuite3LegAdapter)
    adapter.num_envs = 2
    adapter._r_sensor_cols = (1, 3)
    adapter._l_sensor_cols = (0, 2)
    adapter.env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(
        sensordata=torch.tensor([[5., 13., 9., 7.], [1., 30., 2., 4.]]))))
    assert torch.equal(adapter.get_vertical_grf('r'), torch.tensor([20., 34.]))
    assert torch.equal(adapter.get_vertical_grf('l'), torch.tensor([14., 3.]))

    # User's exact phase gains, bounds, and confidence schedule remain intact.
    cfg = PhaseConfig(reference_stride_s=1.0, toe_off_phase=0.62)
    assert cfg.alpha == .3 and cfg.k_hs == 1 and cfg.k_to == .5
    assert cfg.min_stride_s == .5 and cfg.max_stride_s == 2.0
    scalar = GaitPhaseEstimator(cfg)
    batched = GaitPhaseEstimator(cfg, num_envs=2, device=device)
    batched.initial_phase = .0
    event.reset(initial_grf_N=0.0)
    batched_event.reset(initial_grf_N=torch.tensor([0.0,0.0]))
    # Simulate two valid strides. The first HS only starts the clock.
    for i in range(250):
        grf = 100.0 if (i % 100) < 60 else 0.0
        hs_s, to_s = event.update(grf)
        hs_b, to_b = batched_event.update(torch.tensor([grf,0.0]))
        scalar.update(.01,hs_s,to_s)
        batched.update(.01,hs_b,to_b)
    assert abs(float(batched.phase[0]) - scalar.phase) < 2e-5
    assert abs(float(batched.smoothed_stride_s[0])-scalar.smoothed_stride_s) < 2e-5
    assert abs(float(batched.confidence[0])-scalar.confidence) < 2e-5
    assert batched.valid_strides[0] >= 2
    assert batched.confidence[1] == 0
    assert isinstance(scalar.phase, float) and isinstance(scalar.confidence, float)
    assert isinstance(batched.phase, torch.Tensor) and batched.phase.shape == (2,)
    assert batched.frequency_hz.shape == (2,)
    batched.update(.01, torch.tensor([False,False]), torch.tensor([False,False]),
                   done=torch.tensor([True,False]))
    assert batched.phase[0] == 0 and batched.valid_strides[0] == 0

    class MockSim:
        num_envs, device, dt = 2, torch.device('cpu'), .01
        actuator_names = names
        def __init__(self):
            self.cmd = None
            self.grf = torch.tensor([0.0,25.0])
        def reset(self):
            return {'actor':torch.zeros(2,3)}, {}
        def get_vertical_grf(self, side='r'):
            return self.grf
        def get_joint_state(self, joint_names):
            assert joint_names == ('hip_flexion_r','knee_angle_r')
            return torch.ones((2,2)), torch.zeros((2,2))
        def step(self, cmd):
            self.cmd = cmd.clone()
            self.grf = torch.tensor([20.0,19.9])
            z = torch.zeros(2,dtype=torch.bool)
            return {'actor':torch.ones(2,3)},torch.zeros(2),z,z,{}
        def close(self):
            pass

    sim=MockSim()
    env=GaitRLEnv(ref,cfg,sim=sim,
                  rsi_config=RSIConfig(mode='standard', initial_phase=.1))
    obs,_=env.reset()
    assert obs['actor'].shape==(2,3+2+1+80+2+2)
    targets = env.reference_targets()
    base = targets['feedforward_muscles_80']
    action = torch.zeros(2,80)
    assert torch.allclose(env.action_to_excitation(action,targets),base)
    obs,reward,term,trunc,info=env.step(action)
    assert torch.allclose(sim.cmd,base)
    assert info['heel_strike'].tolist()==[True,False]
    assert info['toe_off'].tolist()==[False,True]
    assert torch.allclose(info['vertical_grf_N'], torch.tensor([20.0,19.9]))
    assert reward.shape==(2,)
    assert set(info['imitation_terms'])=={'q_mse','q_reward','dq_mse',
                                          'dq_reward','total_imitation_reward'}
    assert not any('muscle' in k for k in info['imitation_terms'])
    env.rsi_config=RSIConfig(mode='physical')
    try: env.reset()
    except NotImplementedError: pass
    else: raise AssertionError('Physical RSI must not activate yet')
    env.close()

    # Prepared RSI samplers still work with the canonical reference.
    sample=RSISampler(RSIConfig(mode='kinematic'),ref,device).sample_kinematic(2)
    assert sample['q'].shape==(2,2)
    buf=PhysicalStateBuffer(capacity=2)
    buf.add_batch(torch.ones(3,5),torch.zeros(3,4),torch.zeros(3,80),
                  torch.zeros(3),torch.tensor([True,False,True]))
    assert len(buf)==2
    try: ImitationRewardConfig(muscle_weight=1)
    except TypeError: pass
    else: raise AssertionError('Muscle tracking must not be a reward')
    print('PASS: single phase estimator, single reference interpolation, single mapping matrix path; '
          '20 N event crossings, scalar/batched parity, residual feedforward, joint-only reward, RSI gating')


if __name__=='__main__': run()
