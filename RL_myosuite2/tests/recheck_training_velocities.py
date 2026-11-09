#!/usr/bin/env python3
"""Re-audit velocities AFTER root-reference exclusion and MTP angle repairs.

Uses the SAME quality overlay as the RL loader. This reveals any remaining
large velocities that were previously obscured by the worst seam/MTP outliers.
Produces named per-DOF extrema and reference-level issues; no files rewritten.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from gait_rl.reference import ReferenceLibrary
from gait_rl.mtp_repair import periodic_derivative


def write_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf8') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def audit(model, mj, library, output:Path, *, root_limit=3., joint_limit=20., site_limit=30.):
    output.mkdir(parents=True,exist_ok=True)
    names=[]
    for dof in range(model.nv):
        jid=int(model.dof_jntid[dof]);name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,jid)
        names.append(name or f'dof_{dof}')
    rowlist=[];alerts=[];dof_max=np.zeros(model.nv)
    right=int(model.jnt_dofadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'mtp_angle_r')])
    left=int(model.jnt_dofadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,'mtp_angle_l')])
    # Inspect each *retained*, *patched* reference rather than the source file.
    for idx,p in enumerate(library.paths,1):
        ref=library.get(p)
        v=ref.qvel
        vmax=np.max(np.abs(v),axis=0)
        dof_max=np.maximum(dof_max,vmax)
        root=np.linalg.norm(v[:,:3],axis=1)
        sites=max(float(np.max(np.abs(ref.site_angvel))),float(np.max(np.abs(ref.site_linvel))))
        rpeak=float(np.max(np.abs(v[:,right])));lpeak=float(np.max(np.abs(v[:,left])))
        rowlist.append(dict(relative_reference=p.relative_to(library.root).as_posix(),
            root_peak_m_s=float(np.max(root)),mtp_right_peak_rad_s=rpeak,
            mtp_left_peak_rad_s=lpeak,site_velocity_peak_m_or_rad_s=sites,
            corrected_mtp=ref.mtp_patch_applied))
        for i in np.flatnonzero(vmax[6:]>joint_limit)+6:
            alerts.append(dict(relative_reference=p.relative_to(library.root).as_posix(),
                kind='joint_dof_velocity',quantity=names[i],
                peak=float(vmax[i]), threshold=joint_limit))
        if root.max()>root_limit:
            alerts.append(dict(relative_reference=p.relative_to(library.root).as_posix(),
                kind='root_speed',quantity='root translational speed',
                peak=float(root.max()),threshold=root_limit))
        if sites>site_limit:
            alerts.append(dict(relative_reference=p.relative_to(library.root).as_posix(),
                kind='landmark_velocity',quantity='site linear/angular velocity',
                peak=sites,threshold=site_limit))
        if idx%200==0:print(f'{idx}/{len(library.paths)} audited')
    alerts.sort(key=lambda r:r['peak']/r['threshold'],reverse=True)
    write_csv(output/'remaining_reference_velocity_ranges.csv',rowlist,
              ['relative_reference','root_peak_m_s','mtp_right_peak_rad_s',
               'mtp_left_peak_rad_s','site_velocity_peak_m_or_rad_s','corrected_mtp'])
    write_csv(output/'remaining_velocity_outliers.csv',alerts,
              ['relative_reference','kind','quantity','peak','threshold'])
    write_csv(output/'per_dof_velocity_maxima.csv',
              [dict(dof=i,name=names[i],max_abs_velocity=float(x)) for i,x in enumerate(dof_max)],
              ['dof','name','max_abs_velocity'])
    summary=dict(retained_references=len(library.paths),
        mtp_patched_references=int(sum(r['corrected_mtp'] for r in rowlist)),
        max_root_speed_m_s=max(r['root_peak_m_s'] for r in rowlist),
        references_root_speed_over_limit=sum(r['root_peak_m_s']>root_limit for r in rowlist),
        max_mtp_right_rad_s=max(r['mtp_right_peak_rad_s'] for r in rowlist),
        max_mtp_left_rad_s=max(r['mtp_left_peak_rad_s'] for r in rowlist),
        references_mtp_over_limit=sum(max(r['mtp_right_peak_rad_s'],r['mtp_left_peak_rad_s'])>joint_limit for r in rowlist),
        remaining_alerts=len(alerts),
        note='Audits effective post-exclusion post-MTP-overlay reference arrays. NO physics stepping.')
    (output/'remaining_velocity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--references',type=Path,required=True)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('mtp_training_quality/after_repair_audit'))
    p.add_argument('--root-limit',type=float,default=3.)
    p.add_argument('--joint-limit',type=float,default=20.)
    p.add_argument('--site-limit',type=float,default=30.)
    args=p.parse_args()
    import mujoco as mj
    model=(mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower()=='.mjb'
           else mj.MjModel.from_xml_path(str(args.model)))
    library=ReferenceLibrary(args.references,quality_dir=args.quality_dir)
    print(f'Effective library: {len(library.paths)} references '
          f'({library.n_cached} preloaded, {sum(x.mtp_patch_applied for x in library._cache.values())} MTP patched)')
    audit(model,mj,library,args.output,root_limit=args.root_limit,
          joint_limit=args.joint_limit,site_limit=args.site_limit)

if __name__=='__main__':main()
