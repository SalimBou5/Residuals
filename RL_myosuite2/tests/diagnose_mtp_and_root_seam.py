#!/usr/bin/env python3
"""Check MTP velocity spikes vs MyoLeg foot penetration; preview seam root-velocity repair.

Non-destructive: source *_loopable.npz files are NEVER rewritten. Optional
root-velocity patches are written separately and only applied by the RL
reference loader when explicitly enabled via root_velocity_patch_root.

Only the 3 translational root velocities are patched; joint velocities,
positions, foot markers, events, and site-relative references are untouched.

Important: This is a pragmatic velocity-target correction; it does not change
qpos or prove that the resulting qvel is the derivative of qpos. The report
includes a derivative-consistency diagnostic for this reason.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from audit_periodic_velocities import periodic_neighbors

FOOT_BODIES = {
    "right": ("calcn_r", "toes_r", "talus_r", "foot_r"),
    "left": ("calcn_l", "toes_l", "talus_l", "foot_l"),
}


def fix_root_seam(qvel, times, *, trigger_m_s=3.0, blend_phase=0.05,
                  anchor_min_phase=0.04, anchor_max_phase=0.08):
    """Return patched qvel and metadata, or unchanged copy with reason.

    Anchor speeds are componentwise median of velocities at phase 4-8% and
    92-96%. Only the first/last 5% of *root translational velocity* is blended.
    Only references with an excessive speed inside this seam neighborhood and
    a non-excessive anchor are eligible. No changes to any non-root DoF.
    """
    v = np.asarray(qvel, dtype=float)
    t = np.asarray(times, dtype=float)
    if v.ndim != 2 or v.shape[1] < 3 or t.shape != (len(v),) or len(v) < 12:
        raise ValueError("Expected qvel (N,nv), times (N,), N>=12")
    if not np.isfinite(v).all() or not np.isfinite(t).all() or t[0] != 0 or t[-1] <= 0:
        raise ValueError("Invalid velocities/timing")
    if not 0 < blend_phase < anchor_max_phase < 0.5 or not 0 < anchor_min_phase < anchor_max_phase:
        raise ValueError("Invalid blend/anchor phases")
    phi = t / t[-1]
    dist = np.minimum(phi, 1 - phi)
    seam = dist <= blend_phase + 1e-12
    anchor = ((phi >= anchor_min_phase) & (phi <= anchor_max_phase) |
              (phi >= 1 - anchor_max_phase) & (phi <= 1 - anchor_min_phase))
    vmax = float(np.max(np.linalg.norm(v[seam, :3], axis=1)))
    details = dict(before_peak_seam_m_s=vmax, after_peak_seam_m_s=vmax,
                   anchor_speed_m_s=float('nan'), modified=False,
                   modified_rows=0, reason="", median_root_velocity_xyz=None)
    result = v.copy()
    if vmax <= trigger_m_s:
        details["reason"] = "no_excessive_seam_speed"
        return result, details
    if np.sum(anchor & (phi < 0.5)) < 2 or np.sum(anchor & (phi > 0.5)) < 2:
        details["reason"] = "insufficient_safe_anchor_samples"
        return result, details
    nominal = np.median(v[anchor, :3], axis=0)
    nominal_speed = float(np.linalg.norm(nominal))
    details["anchor_speed_m_s"] = nominal_speed
    details["median_root_velocity_xyz"] = nominal.tolist()
    if nominal_speed >= trigger_m_s:
        details["reason"] = "anchor_itself_too_fast"
        return result, details

    fraction = np.clip(dist[seam] / blend_phase, 0, 1)
    weight = 1 - (3*fraction*fraction - 2*fraction*fraction*fraction)
    result[seam, :3] = (1-weight[:, None])*v[seam, :3] + weight[:, None]*nominal
    # Both endpoint records represent the exact same periodic state/velocity.
    result[-1, :3] = result[0, :3]
    details.update(after_peak_seam_m_s=float(np.max(np.linalg.norm(result[seam, :3], axis=1))),
                   modified=True, modified_rows=int(np.count_nonzero(seam)), reason="patched")
    return result, details


def root_fd_mismatch(qpos, times, displacement, original_qvel, patched_qvel, blend_phase):
    """Compare velocity consistency against seam-aware root-position secants."""
    t = np.asarray(times, float)
    ph = t / t[-1]
    mask = np.minimum(ph, 1-ph) <= blend_phase + 1e-12
    before, after = [], []
    for i in np.flatnonzero(mask):
        q0, q1, elapsed = periodic_neighbors(qpos, times, displacement, int(i))
        v_fd = (q1[:3] - q0[:3]) / elapsed
        before.append(np.linalg.norm(original_qvel[i,:3] - v_fd))
        after.append(np.linalg.norm(patched_qvel[i,:3] - v_fd))
    return (float(np.max(before)), float(np.max(after)))


def foot_geom_set(mj, model, side):
    obj = mj.mjtObj.mjOBJ_BODY
    roots = {int(mj.mj_name2id(model, obj, name)) for name in FOOT_BODIES[side]}
    roots.discard(-1)
    if not roots:
        raise ValueError(f"No foot bodies found for {side}: {FOOT_BODIES[side]}")
    result = set()
    for geom in range(model.ngeom):
        b = int(model.geom_bodyid[geom])
        while b > 0:
            if b in roots:
                result.add(geom)
                break
            b = int(model.body_parentid[b])
    if not result:
        raise ValueError(f"No collision geoms found for {side}")
    return result


def sample_foot_penetration(mj, model, data, qpos, floor_id, geom_sets):
    """Penetration per foot (meters) from collision geometry under imposed poses."""
    out = {side: np.zeros(len(qpos), dtype=float) for side in geom_sets}
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mj.mj_fwdPosition(model, data)  # no dynamics/stepping or force estimation
        for j in range(data.ncon):
            contact = data.contact[j]
            a, b = int(contact.geom1), int(contact.geom2)
            if a != floor_id and b != floor_id:
                continue
            geom = b if a == floor_id else a
            depth = max(0.0, -float(contact.dist))
            if depth > 0:
                for side, eligible in geom_sets.items():
                    if geom in eligible:
                        out[side][i] = max(out[side][i], depth)
    return out


def rank_correlation(a, b):
    """Simple average-rank Spearman correlation, including tied zero values."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    def rank(x):
        order = np.argsort(x, kind='mergesort')
        r = np.empty(len(x), dtype=float)
        sorted_x = x[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_x) != 0)+1]
        ends = np.r_[starts[1:],len(x)]
        for s,e in zip(starts,ends):
            r[order[s:e]] = 0.5*(s+e-1)
        return r
    return float(np.corrcoef(rank(a),rank(b))[0,1])


def write_csv(path, rows, fields):
    with open(path,'w', newline='', encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def relative_patch_path(root, ref):
    if root.is_file():
        return Path(ref.stem + '_root_qvel_patch.npz')
    return ref.relative_to(root).with_name(ref.stem+'_root_qvel_patch.npz')


def build_plots(output, plots, limit):
    if not plots or limit < 1:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[]
    for kind in ('mtp','root'):
        selected.extend(sorted((x for x in plots if x[1]==kind), key=lambda x:x[0], reverse=True)[:limit])
    counters={'mtp':0,'root':0}
    for item in selected:
        _, kind, label, phi, signals = item
        counters[kind]+=1
        rank=counters[kind]
        if kind=='mtp':
            fig, ax=plt.subplots(3,1,figsize=(10,8),sharex=True)
            ax[0].plot(phi,signals['q'], lw=1.5);ax[0].set_ylabel('MTP angle (rad)')
            ax[1].plot(phi,signals['vel'],label='Stored velocity');ax[1].axhline(20,ls=':',color='gray')
            ax[1].axhline(-20,ls=':',color='gray');ax[1].set_ylabel('rad/s');ax[1].legend()
            ax[2].plot(phi,1000*signals['pen'],label='Same-foot penetration')
            ax[2].axhline(15,ls=':',color='red',label='15 mm');ax[2].set_ylabel('mm');ax[2].legend()
        else:
            fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
            ax[0].plot(phi,signals['root_pos'],label='root progression coordinate')
            ax[0].set_ylabel('root position (m)');ax[0].legend()
            ax[1].plot(phi,signals['raw'],label='Original root speed')
            ax[1].plot(phi,signals['patch'],label='Proposed corrected root speed')
            ax[1].plot(phi,signals['fd'],ls='--',alpha=.6,label='qpos-derived root speed')
            ax[1].axhline(3,ls=':',color='gray'); ax[1].set_ylabel('m/s');ax[1].legend()
        ax[-1].set_xlabel('Gait phase')
        for a in ax:
            a.grid(alpha=.2)
        fig.suptitle(label)
        fig.tight_layout()
        fig.savefig(output/f'{kind}_{rank:02d}.png',dpi=145)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True,help='MyoLeg XML/MJB')
    p.add_argument('--references',type=Path,required=True,help='A reference NPZ or whole gait_with_landmarks folder')
    p.add_argument('--output',type=Path,default=Path('reference_artifact_report'))
    p.add_argument('--max-files',type=int,default=0,help='0 means all 969 references')
    p.add_argument('--mtp-alert-rad-s',type=float,default=20.0)
    p.add_argument('--penetration-mm',type=float,default=15.0,help='Correlation event: penetration > threshold')
    p.add_argument('--nearby-ms',type=float,default=25.,help='Test proximity to penetration within +/- this many milliseconds')
    p.add_argument('--root-alert-m-s',type=float,default=3.)
    p.add_argument('--blend-phase',type=float,default=.05)
    p.add_argument('--plots',type=int,default=8)
    p.add_argument('--write-root-patches',action='store_true',help='Write separate root-only NPZ patches; never edit source')
    args=p.parse_args()
    if args.max_files<0 or args.plots<0 or args.mtp_alert_rad_s<=0 or args.nearby_ms<0 or args.penetration_mm<0:
        p.error('invalid threshold or count')
    import mujoco as mj
    model=(mj.MjModel.from_binary_path(str(args.model)) if args.model.suffix.lower()=='.mjb'
           else mj.MjModel.from_xml_path(str(args.model)))
    obj=mj.mjtObj.mjOBJ_JOINT
    joint_ids={side:int(mj.mj_name2id(model,obj,f'mtp_angle_{side[0]}')) for side in ('right','left')}
    if any(x<0 for x in joint_ids.values()):
        raise ValueError(f'MTP joints missing: {joint_ids}')
    dofs={side:int(model.jnt_dofadr[joint_ids[side]]) for side in ('right','left')}
    qpos_adrs={side:int(model.jnt_qposadr[joint_ids[side]]) for side in ('right','left')}
    floor=int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'floor'))
    if floor<0:
        raise ValueError('Could not find floor geom')
    geoms={side:foot_geom_set(mj,model,side) for side in ('right','left')}
    data=mj.MjData(model)
    root=args.references.resolve()
    paths=[root] if root.is_file() else sorted(root.rglob('*_loopable.npz'))
    if args.max_files: paths=paths[:args.max_files]
    if not paths: p.error('No reference NPZs found')
    args.output.mkdir(parents=True,exist_ok=True)
    spike_rows=[];ref_rows=[];patch_rows=[];plots=[]
    spike_by_side={'right':0,'left':0};spike_pen={'right':0,'left':0}; baseline_pen={'right':0,'left':0};baseline_n={'right':0,'left':0}
    velocity_samples={'right':[], 'left':[]};penetration_samples={'right':[], 'left':[]}
    patched=0; root_fast=0
    for number, ref in enumerate(paths,1):
        with np.load(ref,allow_pickle=False) as d:
            q=np.asarray(d['qpos'],float);v=np.asarray(d['qvel'],float)
            t=np.asarray(d['time_since_start_s'],float)
            disp=np.asarray(d['root_displacement_xyz_m'],float) if 'root_displacement_xyz_m' in d else (q[-1,:3]-q[0,:3])
        if q.shape!=(len(t),model.nq) or v.shape!=(len(t),model.nv):
            raise ValueError(f'{ref}: incompatible shape')
        ph=t/t[-1]
        penetration=sample_foot_penetration(mj,model,data,q,floor,geoms)
        corrected,meta=fix_root_seam(v,t,trigger_m_s=args.root_alert_m_s,blend_phase=args.blend_phase)
        raw_root=np.linalg.norm(v[:,:3],axis=1)
        corr_root=np.linalg.norm(corrected[:,:3],axis=1)
        root_fd=np.zeros((len(t),3))
        for i in range(len(t)):
            bef,aft,dt=periodic_neighbors(q,t,disp,i)
            root_fd[i]=(aft[:3]-bef[:3])/dt
        root_fast+=int(np.max(raw_root)>args.root_alert_m_s)
        if meta['modified']:
            patched+=1
            fd0,fd1=root_fd_mismatch(q,t,disp,v,corrected,args.blend_phase)
            relpatch=relative_patch_path(root,ref)
            if args.write_root_patches:
                target=args.output/'patches'/relpatch
                target.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target,
                    root_qvel_corrected=corrected[:,:3],
                    source_qvel_root_sha256=np.array(hashlib.sha256(np.ascontiguousarray(v[:,:3]).tobytes()).hexdigest()),
                    source_n=np.array(len(t)),source_duration_s=np.array(t[-1]))
            patch_rows.append(dict(reference=str(ref),patch_relative_path=str(relpatch),
                before_seam_peak_m_s=meta['before_peak_seam_m_s'],after_seam_peak_m_s=meta['after_peak_seam_m_s'],
                anchor_speed_m_s=meta['anchor_speed_m_s'],before_fd_mismatch_m_s=fd0,after_fd_mismatch_m_s=fd1))
            if args.plots:
                plots.append((meta['before_peak_seam_m_s'],'root',str(ref),ph.copy(),
                              dict(root_pos=q[:,1].copy(),raw=raw_root.copy(),patch=corr_root.copy(),
                                   fd=np.linalg.norm(root_fd,axis=1))))
        ref_rows.append(dict(reference=str(ref),max_root_speed_m_s=float(np.max(raw_root)),
            max_root_speed_corrected_m_s=float(np.max(corr_root)),
            max_seam_speed_before_m_s=meta['before_peak_seam_m_s'],
            max_seam_speed_after_m_s=meta['after_peak_seam_m_s'],
            patch_applied=meta['modified'],patch_reason=meta['reason'],
            max_right_penetration_mm=1000*float(np.max(penetration['right'])),
            max_left_penetration_mm=1000*float(np.max(penetration['left']))))
        for side in ('right','left'):
            foot_pen=penetration[side]
            vel=np.abs(v[:,dofs[side]])
            above=foot_pen*1000>args.penetration_mm
            baseline_pen[side]+=int(np.sum(above));baseline_n[side]+=len(t)
            velocity_samples[side].append(vel)
            penetration_samples[side].append(1000*foot_pen)
            idx=np.flatnonzero(vel>=args.mtp_alert_rad_s)
            spike_by_side[side]+=len(idx)
            for i in idx:
                mask=np.abs(t-t[i])<=args.nearby_ms/1000+1e-10
                nearby=1000*float(np.max(foot_pen[mask]))
                spike_pen[side]+=int(above[i])
                spike_rows.append(dict(reference=str(ref),side=side,frame=int(i),phase=float(ph[i]),
                    mtp_velocity_rad_s=float(v[i,dofs[side]]),
                    penetration_at_spike_mm=1000*float(foot_pen[i]),
                    max_nearby_penetration_mm=nearby,
                    beyond_penetration_threshold=bool(above[i])))
            if len(idx) and args.plots:
                plots.append((float(np.max(vel)), 'mtp',f'{ref} — {side}',ph.copy(),
                              dict(q=q[:,qpos_adrs[side]].copy(),vel=v[:,dofs[side]].copy(),pen=foot_pen.copy())))
        if number%100==0: print(f'Analyzed {number}/{len(paths)} references')

    write_csv(args.output/'mtp_spikes.csv',spike_rows,
        ['reference','side','frame','phase','mtp_velocity_rad_s','penetration_at_spike_mm',
         'max_nearby_penetration_mm','beyond_penetration_threshold'])
    write_csv(args.output/'reference_summary.csv',ref_rows,
        ['reference','max_root_speed_m_s','max_root_speed_corrected_m_s','max_seam_speed_before_m_s',
         'max_seam_speed_after_m_s','patch_applied','patch_reason','max_right_penetration_mm','max_left_penetration_mm'])
    write_csv(args.output/'proposed_root_patches.csv',patch_rows,
        ['reference','patch_relative_path','before_seam_peak_m_s','after_seam_peak_m_s',
         'anchor_speed_m_s','before_fd_mismatch_m_s','after_fd_mismatch_m_s'])
    build_plots(args.output,plots,args.plots)
    print(f'\nReferences: {len(paths)} | Root fast anywhere: {root_fast} | Root seam patches proposed: {patched}')
    association_rows=[]
    for side in ('right','left'):
        event_n=spike_by_side[side];at_pen=spike_pen[side]
        vv=np.concatenate(velocity_samples[side]);pp=np.concatenate(penetration_samples[side])
        corr=rank_correlation(vv,pp)
        event_frac=at_pen/event_n if event_n else float('nan')
        base_frac=baseline_pen[side]/baseline_n[side]
        association_rows.append(dict(side=side,spike_frames=event_n,
            spike_frames_with_penetration=at_pen,spike_penetration_fraction=event_frac,
            total_frames=baseline_n[side],all_frames_with_penetration=baseline_pen[side],
            overall_penetration_fraction=base_frac,
            spearman_abs_mtp_velocity_vs_penetration=corr))
        print(f'{side}: {event_n} MTP spike frames >= {args.mtp_alert_rad_s:g} rad/s; '
              f'{at_pen}/{event_n} coincide with >{args.penetration_mm:g} mm penetration '
              f'(baseline frames: {baseline_pen[side]}/{baseline_n[side]}); '
              f'Spearman(abs(MTP vel), penetration)={corr:.3f}')
    write_csv(args.output/'mtp_penetration_summary.csv',association_rows,
        ['side','spike_frames','spike_frames_with_penetration','spike_penetration_fraction',
         'total_frames','all_frames_with_penetration','overall_penetration_fraction',
         'spearman_abs_mtp_velocity_vs_penetration'])
    print('See mtp_spikes.csv, mtp_penetration_summary.csv, reference_summary.csv, proposed_root_patches.csv and plots.')
    if args.write_root_patches:
        print('Root patches saved under output/patches/; they are NOT automatically active.')
    else:
        print('Preview only: no velocity patches written; use --write-root-patches to create them.')
    print('NOTE: imposed qpos and collision geometry are diagnostic, NOT dynamic foot-GRF measurements.')


if __name__=='__main__':
    main()
