#!/usr/bin/env python3
"""Evaluate Step-5 reward on *physically coupled, perturbed reference poses*.

Uses genuine MyoLeg FK + Jacobians and saved data; never advances physics and
never claims that perturbed imposed poses are stable walking. This is a better
calibration probe than replay's nearly perfect site/joint tracking alone.

Example:
python inspect_reward_perturbations.py --model "C:\\Users\\BoussofaraS\\Downloads\\myo_sim\\myo_sim\\leg\\myolegs.xml" --references C:\\Users\\BoussofaraS\\Documents\\residuals\\RL_myosuite2\\data_refining\\gait_training_clean --mep "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\myoleg_bilateral_mep_80.csv" --lookup "C:\\Users\\BoussofaraS\\Documents\\residuals\\data_extraction\\GV-2015\\outputs\\data\\activation_profiles\\activation_lookup.npz" --output reward_perturbation_report
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from rewards import RewardConfig, TerminationConfig
from env import MyoLegResidualEnv
from RSI.reference import ReferenceAssets
from control import ActionConfig
from observations import ObservationConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','references','mep','lookup'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('reward_perturbation_report'))
    p.add_argument('--reward-config',type=Path,default=None)
    p.add_argument('--reference-count',type=int,default=100)
    p.add_argument('--phases-per-reference',type=int,default=2)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if a.reference_count<1 or a.phases_per_reference<1:p.error('Counts must be positive')
    cfg=RewardConfig(**json.loads(a.reward_config.read_text(encoding='utf-8'))) if a.reward_config else RewardConfig()
    rng=np.random.default_rng(a.seed)
    assets=ReferenceAssets(a.references,a.mep,a.lookup)
    env=MyoLegResidualEnv(a.model,assets,action_config=ActionConfig(enable_unmapped=False),
        observation_config=ObservationConfig(normalization_file=None),reward_config=cfg,
        termination_config=TerminationConfig(enable_physical_termination=False,
                                             enable_tracking_termination=False),
        event_sync=False,seed=a.seed)
    a.output.mkdir(parents=True,exist_ok=True)
    refs=np.asarray(assets.library.paths,dtype=object)
    ids=rng.choice(len(refs),size=min(a.reference_count,len(refs)),replace=False)
    scenarios=[('original',0.),('joint_angle_sigma',.05),('joint_angle_sigma',.10),
              ('joint_angle_sigma',.20),('joint_velocity_sigma',1.),
              ('joint_velocity_sigma',2.),('joint_velocity_sigma',4.),
              ('forward_speed_delta',-.35),('forward_speed_delta',.35),
              ('forward_speed_delta',-.7),('pelvis_tilt_deg',15.),('pelvis_tilt_deg',30.),
              ('activation_scale',1.5),('activation_scale',2.),('activation_offset',.10)]
    rows=[];failures=[]
    try:
        for idx in ids:
            ref=assets.library.get(refs[idx])
            for k in range(a.phases_per_reference):
                try:
                    env.reset(options={'reference':ref})
                except RuntimeError as err:
                    failures.append({'reference':str(ref.path),'reason':str(err)})
                    continue
                data=env.rsi.data
                base_q=data.qpos.copy();base_v=data.qvel.copy();base_a=data.act.copy()
                forward=np.asarray(ref.root_displacement[:2],float)
                forward=forward/np.linalg.norm(forward)
                for name,level in scenarios:
                    data.qpos[:]=base_q;data.qvel[:]=base_v;data.act[:]=base_a
                    if name=='joint_angle_sigma':
                        data.qpos[7:]+=rng.normal(0,level,size=env.model.nq-7)
                    elif name=='joint_velocity_sigma':
                        data.qvel[6:]+=rng.normal(0,level,size=env.model.nv-6)
                    elif name=='forward_speed_delta':
                        data.qvel[:2]+=level*forward
                    elif name=='activation_scale':
                        data.act[:]=np.clip(base_a*level,0.,1.)
                    elif name=='activation_offset':
                        data.act[:]=np.clip(base_a+level,0.,1.)
                    elif name=='pelvis_tilt_deg':
                        tilt=np.deg2rad(level)
                        q_tilt=np.array([np.cos(tilt/2),np.sin(tilt/2),0,0],float)
                        dest=np.empty(4)
                        env.mj.mju_mulQuat(dest,q_tilt,base_q[3:7]);data.qpos[3:7]=dest
                    env.mj.mj_forward(env.model,data)
                    env._get_obs() # recompute ALL actual and reference landmarks
                    _,parts,_,_=env.rewards.evaluate(data,env.core.clock,env.core.actions,
                                                    env.core.control_steps)
                    rows.append({'reference':str(ref.path.relative_to(a.references.resolve())),
                      'sample':k,'scenario':name,'value':level,
                      **{key:val for key,val in parts.items() if key.startswith((
                         'score_','contribution_','penalty_','error_'))},
                      'imitation':parts['imitation'],'total':parts['total']})
    finally:env.close()
    if not rows:raise RuntimeError('No accepted RSI poses could be evaluated')
    fields=list(rows[0]);
    with (a.output/'perturbed_reward_samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    group={}
    for row in rows:
        key=(row['scenario'],float(row['value']));group.setdefault(key,[]).append(row)
    with (a.output/'perturbed_reward_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['scenario','value','samples','score_joint_position_mean',
          'score_joint_velocity_mean','score_site_position_mean','score_site_orientation_mean',
          'score_site_linear_velocity_mean','score_site_angular_velocity_mean',
          'score_speed_mean','score_upright_mean','penalty_activation_l1_mean',
          'penalty_activation_l2_mean','reward_mean','reward_p05','reward_p95'])
        for key,rs in sorted(group.items()):
            get=lambda field:np.asarray([r[field] for r in rs],float)
            reward=get('total')
            w.writerow([*key,len(rs),*[float(np.mean(get('score_'+k))) for k in (
                'joint_position','joint_velocity','site_position','site_orientation',
                'site_linear_velocity','site_angular_velocity','speed','upright')],
                float(np.mean(get('penalty_activation_l1'))),float(np.mean(get('penalty_activation_l2'))),
                float(np.mean(reward)),float(np.percentile(reward,5)),float(np.percentile(reward,95))])
    (a.output/'inspection_metadata.json').write_text(json.dumps({
      'description':'Static MyoLeg forward kinematics, NO mj_step, NO learned policy',
      'references_requested':int(len(ids)),'phase_trials_per_reference':a.phases_per_reference,
      'evaluated_cases':len(rows),'failed_resets':failures,
      'note':'Imposed joint perturbations are not physically self-consistent trajectories; these are reward sensitivity checks, not performance forecasts.',
    },indent=2),encoding='utf-8')
    print(f'Wrote {len(rows)} static cases. Inspect {a.output / "perturbed_reward_summary.csv"}')
    print(f'Skipped {len(failures)} reset attempts that could not satisfy RSI ground penetration constraints.')

if __name__=='__main__':main()
