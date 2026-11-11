"""MyoLeg-2 vs MyoSuite-3 *compiled* model audit. Does not alter XML.

This is a gate, not a promise of equivalence. Names/order/addresses MUST match
for direct reuse of Gait120 qpos, lookup act addresses, and policy weights.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def _named(model, mj, typ, n):
    return [mj.mj_id2name(model, typ, i) or f'<unnamed:{i}>' for i in range(n)]


def signature(model, mj):
    obj = mj.mjtObj
    res = {k: int(getattr(model, k)) for k in ('nq','nv','nu','na','njnt','nsite','nbody','nsensor','ntendon','ngeom')}
    res['timestep'] = float(model.opt.timestep)
    res['integrator'] = int(model.opt.integrator)
    res['solver'] = int(model.opt.solver)
    res['joint_names'] = _named(model,mj,obj.mjOBJ_JOINT,model.njnt)
    res['actuator_names'] = _named(model,mj,obj.mjOBJ_ACTUATOR,model.nu)
    res['site_names'] = _named(model,mj,obj.mjOBJ_SITE,model.nsite)
    res['body_names'] = _named(model,mj,obj.mjOBJ_BODY,model.nbody)
    res['sensor_names'] = _named(model,mj,obj.mjOBJ_SENSOR,model.nsensor)
    res['joint_qpos_addresses'] = list(map(int, model.jnt_qposadr))
    res['joint_qvel_addresses'] = list(map(int, model.jnt_dofadr))
    res['actuator_act_addresses'] = list(map(int,model.actuator_actadr))
    res['actuator_act_counts'] = list(map(int,model.actuator_actnum))
    res['actuator_transmission_type'] = list(map(int,model.actuator_trntype))
    res['actuator_transmission_ids'] = model.actuator_trnid.astype(int).tolist()
    res['actuator_gainprm'] = model.actuator_gainprm.tolist()
    res['actuator_biasprm'] = model.actuator_biasprm.tolist()
    res['actuator_dynprm'] = model.actuator_dynprm.tolist()
    res['actuator_ctrlrange'] = model.actuator_ctrlrange.tolist()
    res['body_mass'] = model.body_mass.tolist()
    res['body_inertia'] = model.body_inertia.tolist()
    res['body_local_quat'] = model.body_quat.tolist()
    res['body_local_pos'] = model.body_pos.tolist()
    res['site_local_pos'] = model.site_pos.tolist()
    res['site_local_quat'] = model.site_quat.tolist()
    res['sensor_types'] = list(map(int,model.sensor_type))
    res['sensor_addresses'] = list(map(int,model.sensor_adr))
    res['sensor_dimensions'] = list(map(int,model.sensor_dim))
    # Physical model differences beyond key names/addresses are relevant,
    # but this is not an exhaustive all-field model hash.
    return res


def compare(old, new, *, atol=1e-8):
    critical = ('nq','nv','nu','na','joint_names','joint_qpos_addresses',
                'joint_qvel_addresses','actuator_names','actuator_act_addresses',
                'actuator_act_counts','actuator_transmission_type',
                'actuator_transmission_ids','sensor_names','sensor_types',
                'sensor_addresses','sensor_dimensions','site_names')
    physical = ('timestep','integrator','solver','body_local_quat',
                'body_local_pos','site_local_pos','site_local_quat',
                'body_mass','body_inertia','actuator_gainprm',
                'actuator_biasprm','actuator_dynprm','actuator_ctrlrange')
    report = {'critical_mismatches':{},'physical_mismatches':{},'informational_mismatches':{}}
    for key in old:
        target = ('critical_mismatches' if key in critical else
                  'physical_mismatches' if key in physical else 'informational_mismatches')
        a,b=old[key],new[key]
        if isinstance(a,list) and isinstance(b,list) and key not in critical:
            try:
                aa,bb=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
                same=aa.shape==bb.shape and np.allclose(aa,bb,rtol=0,atol=atol)
                if not same:
                    report[target][key]={
                      'old_shape':list(aa.shape),'new_shape':list(bb.shape),
                      'max_absolute_difference':(float(np.max(np.abs(aa-bb))) if aa.shape==bb.shape and aa.size else None)}
            except (TypeError,ValueError):
                if a!=b: report[target][key]={'old':str(a)[:1000],'new':str(b)[:1000]}
        elif a!=b:
            report[target][key]={'old':a if len(str(a))<1500 else str(a)[:1500],
                                 'new':b if len(str(b))<1500 else str(b)[:1500]}
    report['same_reference_vector_contract']=not bool(report['critical_mismatches'])
    report['matching_checked_physics_fields']=not bool(report['physical_mismatches'])
    report['ready_for_gpu_port']=report['same_reference_vector_contract'] and report['matching_checked_physics_fields']
    report['warning']='Matching audited fields does NOT establish CPU-GPU dynamic equivalence; dynamic parity tests remain mandatory.'
    return report


def _extract_model_from_cpu_env(env, mujoco):
    seen = set()
    for root in (env, getattr(env,'unwrapped',None)):
        if root is None: continue
        for candidate in (getattr(root,'sim',None),root):
            if candidate is None or id(candidate) in seen:continue
            seen.add(id(candidate))
            for attr in ('model','mj_model'):
                obj=getattr(candidate,attr,None)
                if isinstance(obj,mujoco.MjModel):return obj
    raise RuntimeError('Could not locate compiled MjModel on MyoSuite 3 CPU task; pass --new-model XML instead.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-model',required=True,type=Path)
    p.add_argument('--new-model',type=Path,help='Explicit 3.x XML/MJB; otherwise query myosuite.make_env')
    p.add_argument('--task-id',default='myoLegWalk-v0')
    p.add_argument('--out',type=Path,default=Path('model_parity.json'))
    a=p.parse_args()
    import mujoco
    def load(path):
        return (mujoco.MjModel.from_binary_path(str(path)) if path.suffix.lower()=='.mjb'
                else mujoco.MjModel.from_xml_path(str(path)))
    old=load(a.old_model.resolve())
    if a.new_model:
        new=load(a.new_model.resolve())
    else:
        from myosuite import make_env
        env=make_env(a.task_id)
        try:new=_extract_model_from_cpu_env(env,mujoco)
        finally:env.close()
    report=compare(signature(old,mujoco),signature(new,mujoco))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='warning'},indent=2))
    print('Saved',a.out)
    if not report['ready_for_gpu_port']:
        raise SystemExit('MODEL MISMATCH. Do not reuse the 944 vectors without checking the mismatches.')

if __name__=='__main__':main()
