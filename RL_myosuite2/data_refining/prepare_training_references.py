#!/usr/bin/env python3
"""Prepare non-destructive standalone MTP repair audit for MyoLeg Gait120 references.

- Exclude ONLY the 17 documented root-seam references (from the earlier audit).
- Locally smooth MTP joint-angle jumps with limits on size and duration.
- Lock the ORIGINAL start/end MTP angles to preserve gait-cycle boundaries.
- Reject overlarge proposals (never silently flatten a genuine movement).
- Recompute corresponding MTP velocities from corrected positions.
- Recompute the *change* in pelvis-relative site poses and velocities from FK,
  preserving any existing source-vs-FK offsets and unchanged rows exactly.
- Save hashed NPZ repair artifacts beside detailed CSV/JSON audit.
- NEVER rewrite the source gait_with_landmarks reference files.

Required: numpy, mujoco. No training or simulated physics stepping.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

try:
    from .mtp_repair import MTPRepairConfig, repair_mtp
except ImportError:  # Direct execution: python data_preparation/prepare_training_references.py
    from mtp_repair import MTPRepairConfig, repair_mtp


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_exclusions(path: Path) -> dict[str, dict]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        rel = row['relative_reference'].replace('\\', '/').strip('/')
        if rel in result:
            raise ValueError(f'Duplicate exclusion {rel}')
        if len(Path(rel).parts) != 3 or not rel.endswith('_loopable.npz'):
            raise ValueError(f'Invalid exclusion relative path {rel!r}')
        result[rel] = row
    return result


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _relative_fk(mj, model, data, pelvis_id: int, site_ids: list[int]):
    """Compute pelvis-relative kinematics in the observation-builder convention."""
    mj.mj_fwdPosition(model, data)  # no simulation step or time advancement
    Rp = np.asarray(data.xmat[pelvis_id]).reshape((3,3))
    pp = np.asarray(data.xpos[pelvis_id])
    jp=np.zeros((3,model.nv));jr=np.zeros((3,model.nv))
    mj.mj_jacBody(model,data,jp,jr,pelvis_id)
    vp=jp@data.qvel;wp=jr@data.qvel
    pos=np.empty((len(site_ids),3)); vel=np.empty_like(pos)
    rot=np.empty((len(site_ids),3,3)); omg=np.empty_like(pos)
    for k,site_id in enumerate(site_ids):
        mj.mj_jacSite(model,data,jp,jr,site_id)
        x=np.asarray(data.site_xpos[site_id]);dx=x-pp
        v=jp@data.qvel;w=jr@data.qvel
        pos[k]=Rp.T@dx
        vel[k]=Rp.T@(v-vp-np.cross(wp,dx))
        rot[k]=Rp.T@np.asarray(data.site_xmat[site_id]).reshape((3,3))
        omg[k]=Rp.T@(w-wp)
    return pos,vel,rot,omg


def _correct_site_references(mj, model, original: dict, q_new: np.ndarray,
                             v_new: np.ndarray, affected: np.ndarray):
    """Update only affected frames using FK/Jacobian DELTAS, not absolute replacements.

    This preserves the source's original landmark convention/offsets and leaves
    all unaffected landmark samples byte-for-byte unchanged.
    """
    names = tuple(str(x) for x in original['site_names'].tolist())
    ids=[int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,name)) for name in names]
    if any(i<0 for i in ids):
        raise ValueError(f'Sites absent from model: {[n for n,i in zip(names,ids) if i<0]}')
    pelvis_id=int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'pelvis'))
    if pelvis_id<0:raise ValueError('Missing pelvis body')
    data=mj.MjData(model)
    src_pos=np.asarray(original['site_pos_rel'],float)
    src_quat=np.asarray(original['site_quat_rel'],float)
    key='site_linvel_rel' if 'site_linvel_rel' in original else 'site_vel_rel'
    src_vel=np.asarray(original[key],float)
    src_omega=np.asarray(original['site_angvel_rel'],float)
    new_pos=src_pos.copy();new_vel=src_vel.copy()
    new_quat=src_quat.copy();new_omega=src_omega.copy()
    worst_orig_pos_error=0.0; worst_orig_velocity_error=0.0
    for i in np.flatnonzero(affected):
        data.qpos[:]=original['qpos'][i];data.qvel[:]=original['qvel'][i]
        p0,v0,R0,w0=_relative_fk(mj,model,data,pelvis_id,ids)
        data.qpos[:]=q_new[i];data.qvel[:]=v_new[i]
        p1,v1,R1,w1=_relative_fk(mj,model,data,pelvis_id,ids)
        worst_orig_pos_error=max(worst_orig_pos_error,float(np.max(np.linalg.norm(src_pos[i]-p0,axis=1))))
        worst_orig_velocity_error=max(worst_orig_velocity_error,float(np.max(np.linalg.norm(src_vel[i]-v0,axis=1))))
        new_pos[i]+=p1-p0
        new_vel[i]+=v1-v0
        new_omega[i]+=w1-w0
        for s in range(len(names)):
            # Transfer only the rotation induced by the MTP repair.
            stored_R=np.zeros(9)
            mj.mju_quat2Mat(stored_R,np.asarray(src_quat[i,s]))
            delta_R=R1[s]@R0[s].T
            repaired_R=delta_R@stored_R.reshape(3,3)
            result=np.zeros(4)
            mj.mju_mat2Quat(result,np.ascontiguousarray(repaired_R.reshape(9)))
            result/=np.linalg.norm(result)
            if np.dot(result,src_quat[i,s])<0:result=-result
            new_quat[i,s]=result
    return dict(site_pos_rel=new_pos,site_quat_rel=new_quat,
                site_linvel_rel=new_vel,site_angvel_rel=new_omega), dict(
                    max_original_site_position_vs_fk_m=worst_orig_pos_error,
                    max_original_site_velocity_vs_fk_m_s=worst_orig_velocity_error)


def process_one(mj,model,ref:Path,root:Path,out:Path,config:MTPRepairConfig):
    with np.load(ref,allow_pickle=False) as raw:
        original={name:np.array(raw[name]) for name in raw.files}
    q=np.asarray(original['qpos'],float)
    v=np.asarray(original['qvel'],float)
    t=np.asarray(original['time_since_start_s'],float)
    sides=('r','l')
    names=[f'mtp_angle_{s}' for s in sides]
    jids=[int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)) for n in names]
    if any(i<0 for i in jids):raise ValueError(f'Missing MTP joint(s): {names}')
    qadrs=[int(model.jnt_qposadr[j]) for j in jids]
    vadrs=[int(model.jnt_dofadr[j]) for j in jids]
    if q.shape!=(len(t),model.nq) or v.shape!=(len(t),model.nv):
        raise ValueError(f'Unexpected qpos/qvel shapes: {q.shape}, {v.shape}')
    qnew=q.copy();vnew=v.copy();affected=np.zeros(len(t),bool)
    reports=[]
    for s,qa,va in zip(sides,qadrs,vadrs):
        nq,nv,report,mask=repair_mtp(q[:,qa],v[:,va],t,config)
        qnew[:,qa]=nq;vnew[:,va]=nv
        affected|=mask
        reports.append(dict(side=s, qpos_index=qa,qvel_index=va,**report))
    # Reject the WHOLE reference if either MTP cannot be fixed conservatively.
    # A repair proposed on the other leg is not materialized in that case.
    if any(r['rejected'] or not r['achieved_target'] for r in reports):
        # Both sides remain in the ORIGINAL NPZ if either side is rejected.
        # Differentiate 'candidate passed local checks' from 'actually saved'.
        for r in reports:
            r['patch_materialized'] = False
        return dict(relative_reference=ref.relative_to(root).as_posix(),
                    status='unresolved_spike', reports=reports,
                    modified_rows=0), []
    if not any(r['modified'] for r in reports):
        for r in reports:
            r['patch_materialized'] = False
        return dict(relative_reference=ref.relative_to(root).as_posix(),
                    status='unchanged',reports=reports,modified_rows=0),[]
    for r in reports:
        r['patch_materialized'] = bool(r['modified'])
    # If qpos changed then the 8 anatomical site frames are dependent quantities.
    site_updates,diagnostics=_correct_site_references(mj,model,original,qnew,vnew,affected)
    rel=ref.relative_to(root)
    patch_path=out/'patches'/rel.parent/(ref.stem+'_mtp_patch.npz')
    patch_path.parent.mkdir(parents=True,exist_ok=True)
    sha=file_sha256(ref)
    np.savez_compressed(patch_path,
        format_version=np.array(1),source_npz_sha256=np.array(sha),
        source_n=np.array(len(t)),source_duration_s=np.array(float(original['stride_duration_s'])),
        mtp_joint_names=np.array(names),mtp_qpos_indices=np.array(qadrs),
        mtp_qvel_indices=np.array(vadrs),qpos_mtp=qnew[:,qadrs],
        qvel_mtp=vnew[:,vadrs],site_names=np.array([str(x) for x in original['site_names']]),
        **site_updates)
    return dict(relative_reference=rel.as_posix(),status='patched',
                patch_relative_path=patch_path.relative_to(out).as_posix(),
                original_sha256=sha,modified_rows=int(np.count_nonzero(affected)),
                reports=reports,**diagnostics),[
       dict(relative_reference=rel.as_posix(),side=r['side'],
            qpos_index=r['qpos_index'],qvel_index=r['qvel_index'],
            trigger_frames=r['trigger_frames'],modified_qpos_rows=r['modified_qpos_rows'],
            modified_qvel_rows=r['modified_qvel_rows'],
            max_angle_change_rad=r['max_abs_angle_change_rad'],
            max_velocity_before_rad_s=r['max_abs_velocity_before_rad_s'],
            max_velocity_after_rad_s=r['max_abs_velocity_after_rad_s'],
            max_derivative_before_rad_s=r['max_abs_central_derivative_before_rad_s'],
            max_derivative_after_rad_s=r['max_abs_central_derivative_after_rad_s'],
            smoothing_margin_frames=r['margin_frames'],
            intervals_json=json.dumps(r['intervals']),
            status='patched') for r in reports if r['modified']]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--references',required=True,type=Path)
    parser.add_argument('--exclude-csv',type=Path,default=Path(__file__).parent/'reference_quality'/'excluded_root_seams.csv')
    parser.add_argument('--output',type=Path,default=Path('mtp_training_quality'))
    parser.add_argument('--spike-threshold',type=float,default=14.,
                        help='Trigger local MTP smoothing above this angular speed (rad/s)')
    parser.add_argument('--target-velocity',type=float,default=10.,
                        help='Target maximum MTP angular speed within smoothed windows (rad/s)')
    parser.add_argument('--initial-margin',type=int,default=8,
                        help='Number of neighboring frames smoothed on each side of a trigger')
    parser.add_argument('--max-margin',type=int,default=24,
                        help='Maximum local smoothing margin in frames')
    parser.add_argument('--max-angle-change',type=float,default=0.20,
                        help='Reject a proposed repair if any MTP angle changes by more than this many radians')
    parser.add_argument('--max-interval-fraction',type=float,default=0.15,
                        help='Reject if any contiguous repaired window exceeds this fraction of one cycle')
    parser.add_argument('--max-files',type=int,default=0,help='For testing only; 0 means all references')
    parser.add_argument('--plots',type=int,default=8,help='Write before/after plots for N largest accepted repairs (default 8)')
    parser.add_argument('--rejected-plots',type=int,default=8,help='Plot N rejected candidates with suspect intervals highlighted (default 8)')
    args=parser.parse_args()
    if args.max_files<0 or args.plots<0 or args.rejected_plots<0:
        parser.error('--max-files, --plots, and --rejected-plots must be nonnegative')
    cfg=MTPRepairConfig(spike_threshold_rad_s=args.spike_threshold,
                        target_peak_rad_s=args.target_velocity,
                        initial_margin_frames=args.initial_margin,
                        max_margin_frames=args.max_margin,
                        max_angle_change_rad=args.max_angle_change,
                        max_interval_fraction=args.max_interval_fraction)
    import mujoco as mj
    model=(mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower()=='.mjb'
           else mj.MjModel.from_xml_path(str(args.model)))
    reference_root=args.references.resolve()
    if not reference_root.is_dir():parser.error('Expected a reference library directory')
    output=args.output.resolve()
    if reference_root == output or reference_root in output.parents:
        parser.error('Output must NOT be inside the original reference library')
    exclusions=read_exclusions(args.exclude_csv)
    paths=sorted(reference_root.rglob('*_loopable.npz'))
    found={p.relative_to(reference_root).as_posix() for p in paths}
    unknown=set(exclusions)-found
    if unknown:raise ValueError(f'Exclusions missing from source library: {sorted(unknown)}')
    if len(paths)!=969:
        print(f'WARNING: found {len(paths)} references rather than expected 969')
    kept=[p for p in paths if p.relative_to(reference_root).as_posix() not in exclusions]
    partial_scan=bool(args.max_files and args.max_files < len(kept))
    if args.max_files:kept=kept[:args.max_files]
    # Never allow an earlier audit's patches to leak into a new run.
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be new or EMPTY; use a new --output path')
    output.mkdir(parents=True, exist_ok=True)
    rows=[];interval_rows=[];unresolved=[]
    for i,p in enumerate(kept,1):
        row,details=process_one(mj,model,p,reference_root,output,cfg)
        rows.append(row);interval_rows.extend(details)
        if row['status']=='unresolved_spike':unresolved.append(row['relative_reference'])
        if i%100==0 or i==len(kept):
            print(f'{i}/{len(kept)} evaluated, {sum(x["status"]=="patched" for x in rows)} patched, '
                  f'{len(unresolved)} unresolved')
    # Manifest is the source of truth for loader filtering and patch coverage.
    manifest=dict(format_version=1,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_library=str(reference_root),
                  model_file=str(args.model.resolve()),model_sha256=file_sha256(args.model),
                  exclusions=sorted(exclusions),
                  excluded_count=len(exclusions),
                  scanned_count=len(kept), partial_scan=partial_scan,
                  patched_references=sorted(row['relative_reference'] for row in rows if row['status']=='patched'),
                  unresolved_references=unresolved,
                  repair_version='mtp-local-hermite-v3',
                  repair_config=vars(cfg),
                  note='Source NPZ untouched; cycle-end MTP angles preserved; rejected candidates are not patched and must be reviewed before building a training library. Root velocity patch is NOT used.')
    (output/'quality_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    cols=['relative_reference','side','qpos_index','qvel_index','trigger_frames',
          'modified_qpos_rows','modified_qvel_rows','max_angle_change_rad',
          'max_velocity_before_rad_s','max_velocity_after_rad_s',
          'max_derivative_before_rad_s','max_derivative_after_rad_s',
          'smoothing_margin_frames','intervals_json','status']
    write_csv(output/'mtp_repairs.csv',interval_rows,cols)
    write_csv(output/'excluded_root_references.csv',[
        {**details, "relative_reference": key} for key,details in sorted(exclusions.items())
    ],['relative_reference','reason','original_seam_peak_m_s','root_position_derivative_mismatch_after_velocity_only_patch_m_s'])
    write_csv(output/'remaining_reference_summary.csv',[
        dict(relative_reference=row['relative_reference'],status=row['status'],
             modified_frames=row['modified_rows']) for row in rows
    ],['relative_reference','status','modified_frames'])
    (output/'complete_changes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    interval_rows=[]
    for row in rows:
        for side in row['reports']:
            for region in side.get('intervals',[]):
                interval_rows.append(dict(relative_reference=row['relative_reference'],
                                          side=side['side'],**region))
    write_csv(output/'mtp_changed_intervals.csv',interval_rows,
              ['relative_reference','side','start_frame','end_frame','start_phase',
               'end_phase','wraps_cycle','frame_count'])
    rejected_rows = []
    for row in rows:
        if row['status'] != 'unresolved_spike':
            continue
        for side in row['reports']:
            if not side.get('rejected', False):
                continue
            candidates = side.get('attempted_candidates', [])
            candidate = candidates[0] if candidates else {}
            rejected_rows.append(dict(
                relative_reference=row['relative_reference'], side=side['side'],
                trigger_frames=side['trigger_frames'],
                reasons=';'.join(side['rejection_reasons']),
                attempted_candidates=len(candidates),
                first_candidate_angle_change_rad=candidate.get('max_abs_angle_change_rad', ''),
                first_candidate_interval_fraction=candidate.get('max_interval_fraction', ''),
                first_candidate_max_velocity_rad_s=candidate.get('max_abs_recomputed_velocity_rad_s', ''),
                accepted_patch=False))
    write_csv(output/'mtp_rejected_repairs.csv', rejected_rows,
              ['relative_reference','side','trigger_frames','reasons',
               'attempted_candidates','first_candidate_angle_change_rad',
               'first_candidate_interval_fraction',
               'first_candidate_max_velocity_rad_s','accepted_patch'])
    if args.plots or args.rejected_plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        to_plot=sorted((r for r in rows if r['status']=='patched'),
                       key=lambda r:max(s['max_abs_angle_change_rad'] for s in r['reports']),
                       reverse=True)[:args.plots]
        (output/'plots').mkdir(exist_ok=True)
        jids=[int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'mtp_angle_{side}'))
              for side in ('r','l')]
        qids=[int(model.jnt_qposadr[j]) for j in jids]
        vids=[int(model.jnt_dofadr[j]) for j in jids]
        for rank,r in enumerate(to_plot,1):
            source=reference_root/r['relative_reference']
            with np.load(source,allow_pickle=False) as d:
                qq=np.asarray(d['qpos']);vv=np.asarray(d['qvel']);tt=np.asarray(d['time_since_start_s'])
            with np.load(output/r['patch_relative_path'],allow_pickle=False) as d:
                qpatch=np.asarray(d['qpos_mtp']);vpatch=np.asarray(d['qvel_mtp'])
            fig,ax=plt.subplots(2,2,figsize=(12,6),sharex=True,constrained_layout=True)
            for col,side in enumerate(('r','l')):
                ax[0,col].plot(tt/tt[-1],qq[:,qids[col]],label='Original MTP angle')
                ax[0,col].plot(tt/tt[-1],qpatch[:,col],label='Repaired MTP angle')
                ax[1,col].plot(tt/tt[-1],vv[:,vids[col]],label='Original MTP velocity')
                ax[1,col].plot(tt/tt[-1],vpatch[:,col],label='Repaired MTP velocity')
                ax[1,col].axhline(cfg.spike_threshold_rad_s,color='0.5',ls=':',lw=1)
                ax[1,col].axhline(-cfg.spike_threshold_rad_s,color='0.5',ls=':',lw=1)
                ax[0,col].set_title(f'{side} MTP')
                ax[0,col].set_ylabel('Angle (rad)');ax[1,col].set_ylabel('Angular velocity (rad/s)')
                ax[1,col].set_xlabel('Gait phase')
                ax[0,col].legend(fontsize=8);ax[1,col].legend(fontsize=8)
                for a in ax[:,col]:a.grid(alpha=.2)
            fig.suptitle(r['relative_reference'])
            fig.savefig(output/'plots'/f'mtp_repair_{rank:02d}.png',dpi=130)
            plt.close(fig)
        if args.rejected_plots:
            rejected_for_plots = sorted(
                (r for r in rows if r['status']=='unresolved_spike'),
                key=lambda r:max((a.get('max_abs_angle_change_rad', 0.)
                                  for side in r['reports']
                                  for a in side.get('attempted_candidates', [])),
                                 default=0.), reverse=True)[:args.rejected_plots]
            for rank, r in enumerate(rejected_for_plots, 1):
                source=reference_root/r['relative_reference']
                with np.load(source,allow_pickle=False) as d:
                    qq=np.asarray(d['qpos']);vv=np.asarray(d['qvel']);tt=np.asarray(d['time_since_start_s'])
                fig,ax=plt.subplots(2,2,figsize=(12,6),sharex=True,constrained_layout=True)
                for col,side in enumerate(('r','l')):
                    ax[0,col].plot(tt/tt[-1],qq[:,qids[col]],label='Original angle (unchanged)')
                    ax[1,col].plot(tt/tt[-1],vv[:,vids[col]],label='Original velocity (unchanged)')
                    for candidate in r['reports'][col].get('attempted_candidates',[])[:1]:
                        for interval in candidate.get('intervals',[]):
                            a,b=interval['start_phase'],interval['end_phase']
                            spans=[(a,1.),(0.,b)] if interval['wraps_cycle'] else [(a,b)]
                            for aa,bb in spans:
                                for rowax in ax[:,col]:
                                    rowax.axvspan(aa,bb,color='tab:red',alpha=.13)
                    ax[0,col].set_title(f'{side} MTP — no patch applied')
                    ax[0,col].set_ylabel('Angle (rad)')
                    ax[1,col].set_ylabel('Angular velocity (rad/s)')
                    ax[1,col].set_xlabel('Gait phase')
                    for rowax in ax[:,col]:
                        rowax.grid(alpha=.2)
                        rowax.legend(fontsize=8)
                fig.suptitle('REJECTED — '+r['relative_reference']+' — source preserved')
                fig.savefig(output/'plots'/f'mtp_rejected_{rank:02d}.png',dpi=130)
                plt.close(fig)
    print(f'DONE: {len(exclusions)} excluded, {len(kept)} scanned, '
          f'{len(manifest["patched_references"])} patched, {len(unresolved)} unresolved')
    if unresolved:
        print('WARNING: proposed repairs exceeded conservative limits or could not reach the velocity target.')
        print('Inspect mtp_rejected_repairs.csv and complete_changes.json. '
              'The training-library builder will refuse unresolved references; '
              'nothing was silently dropped or edited.')


if __name__=='__main__':main()
