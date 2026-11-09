#!/usr/bin/env python3
"""Fit fixed OOD-tolerant normalization from a Step-4 per-feature CSV.

Example:
python fit_observation_normalization.py --features "features(2).csv" \
    --metadata "metadata(2).json" --output observation_normalization_refined_944.json

Only healthy-reference medians/percentiles are taken from the CSV. This does
NOT imply simulated states remain within healthy reference quantiles. Kinematic
scale floors, a generous clipping limit, and a physically chosen asinh force
scale specifically accommodate early unstable training rollouts.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SITES = ('knee_r','knee_l','ankle_r','ankle_l','toe_r','toe_l')
ORI_SITES = ('hip_r','hip_l',*SITES)
BLOCKS = ('joint_proprioception','root_state','foot_contacts','muscle_activations',
          'actual_sites','reference_sites','reference_joints','future_joints',
          'mep_baseline','gait_phase')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_scale_floor(name):
    """Conservative scales in SI units, deliberately larger than some gait variations.

    Certain MyoLeg joint names contain 'translation' DOFs, which are in meters
    even though older feature CSVs labeled the entire joint block 'rad'.
    """
    if name.startswith(('joint_position/', 'reference_joint_position/', 'future_')):
        return 0.02 if 'translation' in name else 0.15  # m or rad
    if name.startswith('joint_velocity/'):
        return 0.15 if 'translation' in name else 1.5  # m/s or rad/s
    if name == 'pelvis_height':
        return 0.10   # m; permits early falling states without huge magnification
    if name.startswith('pelvis_local_linear_velocity/'):
        return 0.50  # m/s
    if name.startswith('pelvis_local_angular_velocity/'):
        return 0.75  # rad/s
    if '/position/' in name and name.startswith(('actual/', 'reference/')):
        return 0.10  # m
    if '/linear_velocity/' in name and name.startswith(('actual/', 'reference/')):
        return 0.50  # m/s
    if '/angular_velocity/' in name and name.startswith(('actual/', 'reference/')):
        return 2.0   # rad/s
    if name == 'desired_forward_speed':
        return 0.40  # m/s
    raise ValueError(f'No physical scale floor for {name!r}')


def fit(features_path: str | Path, metadata_path: str | Path, *, clip=10.0,
        force_scale_n=200.0):
    features_path, metadata_path = Path(features_path), Path(metadata_path)
    with features_path.open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    rows.sort(key=lambda r: int(r['index']))
    names = [r['feature'] for r in rows]
    if [int(r['index']) for r in rows] != list(range(len(rows))):
        raise ValueError('Feature indices must be unique, contiguous and ordered')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate observation feature names')
    if len(rows) != int(meta['observation_dimensions']):
        raise ValueError('Feature count does not match inspection metadata')
    if int(meta['accepted_poses']) <= 0:
        raise ValueError('Expected a nonempty inspection')
    if int(meta['references_inspected']) != int(meta['references_indexed']):
        raise ValueError('All references must be inspected before fitting the fixed normalizer')
    if len(meta['observed_muscles']) != sum(r['block']=='muscle_activations' for r in rows):
        raise ValueError('Observed muscles mismatch')
    by_name = {r['feature']: r for r in rows}
    blocks = {}
    for block in BLOCKS:
        inds = [i for i,r in enumerate(rows) if r['block'] == block]
        if not inds or inds != list(range(min(inds), max(inds)+1)):
            raise ValueError(f'Missing or noncontiguous observation block {block}')
        blocks[block] = [min(inds), max(inds)+1]

    modes, centers, scales, floors, fit_source = [], [], [], [], []
    for r in rows:
        name = r['feature']
        block = r['block']
        source = name
        if block in ('joint_proprioception','reference_joints','future_joints'):
            if name.startswith('joint_position/'):
                source = 'reference_joint_position/' + name.split('/',1)[1]
            elif name.startswith('future_'):
                source = 'reference_joint_position/' + name.split('_joint_position/',1)[1]
        elif block == 'actual_sites':
            source = 'reference/' + name.split('/',1)[1]
        if block == 'foot_contacts':
            mode, c, s, floor = 'force_asinh', 0., 1., 0.
        elif (block in ('muscle_activations','mep_baseline') or
              'quat/' in name or 'quaternion/' in name or '/orientation/' in name or
              name in ('sin_phase','cos_phase','event_correction_active')):
            mode, c, s, floor = 'identity', 0., 1., 0.
        elif name in ('wrapped_phase_drift','remaining_event_correction'):
            mode, c, s, floor = 'affine', 0., (0.5 if name=='wrapped_phase_drift' else 0.3), 0.
        else:
            mode = 'affine'
            if source not in by_name:
                raise ValueError(f'Missing matching reference feature for {name}: {source}')
            src = by_name[source]
            floor = physical_scale_floor(name)
            c = float(src['median'])
            s = max(0.5*(float(src['p99'])-float(src['p01'])),floor)
        modes.append(mode); centers.append(c); scales.append(s)
        floors.append(floor); fit_source.append(source)

    # Consistency guards: actual joints/reference/lookaheads and actual/ref sites
    # must always receive identical centers/scales.
    for name, source in zip(names, fit_source):
        if source != name and source in by_name:
            i, j = names.index(name), names.index(source)
            if modes[i] != modes[j] or not np.isclose(centers[i],centers[j]) or not np.isclose(scales[i],scales[j]):
                raise ValueError(f'Inconsistent normalization between {name} and {source}')
    # Features already contain input distributions; no raw samples needed to fit.
    # Force values in inspection CSV were previously saturated/scaled so they
    # are deliberately NOT used to determine force normalization.
    return {
        'format_version':1,
        'reference_count':int(meta['references_indexed']),
        'sample_count':int(meta['accepted_poses']),
        'source_library':meta.get('reference_library',''),
        'feature_csv_sha256':sha256(features_path),
        'inspection_metadata_sha256':sha256(metadata_path),
        'description':'Fixed reference-based scale with OOD physical floors; NO running statistics',
        'clip':float(clip),
        'force_scale_n':float(force_scale_n),
        'observed_muscles':list(meta['observed_muscles']),
        'lookahead_seconds':[0.1,0.2],
        'block_slices':blocks,
        'feature_names':names,
        'mode':modes,
        'center':centers,
        'scale':scales,
        'minimum_physical_scale':floors,
        'reference_statistic_source':fit_source,
        'notes':[
            'Forces enter observation builder in raw Newtons, then use asinh(max(F,0)/200 N).',
            'Prior F/(F+100N) inspection force percentiles were not used to fit forces.',
            'Reference distributions provide baseline centers/scales, not allowable state bounds.',
            'MuJoCo joint names containing translation are interpreted as meter/meter-per-second DOFs.',
            'Phase drift and correction use controller-specific cycle scales, not zero replay variance.',
            'Clipping at +/-10 applies only to transformed non-identity observations; monitor saturation.',
            'Normalizing 505-D observations requires a separate inspection with enable_unmapped=true.',
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clip',type=float,default=10.0)
    p.add_argument('--force-scale-n',type=float,default=200.0)
    a=p.parse_args()
    if a.clip <= 0 or a.force_scale_n <= 0:
        p.error('clip and force scale must be positive')
    settings=fit(a.features,a.metadata,clip=a.clip,force_scale_n=a.force_scale_n)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    print(f'Saved {a.output}; n={len(settings["feature_names"])}; references={settings["reference_count"]}; '
          f'samples={settings["sample_count"]}; force asinh F/{settings["force_scale_n"]:g}N')

if __name__=='__main__':main()
