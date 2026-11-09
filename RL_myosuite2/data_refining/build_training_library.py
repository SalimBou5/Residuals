#!/usr/bin/env python3
"""Build a self-contained, cleaned Gait120 reference library WITHOUT touching RL code.

Inputs:
  --references  ORIGINAL gait_with_landmarks directory
  --quality-dir  reports/patches produced by prepare_training_references.py
  --output       NEW directory for the curated reference library

The new directory has the exact standard .../S###/Trial##/step##_loopable.npz
layout and sidecars expected by the original, unmodified gait_rl.reference loader.
No patches are applied at runtime, and the source directory is never written.

Every exclusion and repaired sample interval remains documented in
OUTPUT/_preparation/.  A temporary staging directory prevents half-built
libraries from being mistaken for finished training data.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _inside(directory: Path, parent: Path) -> bool:
    return directory == parent or parent in directory.parents


def _validate_manifest(raw: dict, source: Path, quality: Path, model: Path | None,
                       exclude_unresolved: bool = False) -> tuple[list[Path], set[str], set[str], set[str], set[str]]:
    """Require a complete, internally consistent audit before curating any reference.

    The eight unresolved MTP references can only be omitted with an explicit flag.
    No unresolved data is silently copied or silently excluded.
    """
    if int(raw.get('format_version', -1)) != 1 or raw.get('partial_scan', True):
        raise ValueError('Quality report must be complete and format_version=1')
    if model is not None and raw.get('model_sha256') and sha256(model) != raw['model_sha256']:
        raise ValueError('MuJoCo model differs from model used to create MTP repairs')
    paths = sorted(source.rglob('*_loopable.npz'))
    if not paths:
        raise ValueError(f'No reference NPZ files in {source}')
    available = {p.relative_to(source).as_posix() for p in paths}

    def unique_set(key: str) -> set[str]:
        entries = raw.get(key)
        if not isinstance(entries, list) or any(not isinstance(x, str) or not x for x in entries):
            raise ValueError(f'Invalid or missing {key} in quality report')
        if len(entries) != len(set(entries)):
            raise ValueError(f'Duplicate {key} in quality report')
        return set(entries)

    root_excluded = unique_set('exclusions')
    unresolved = unique_set('unresolved_references')
    patched = unique_set('patched_references')
    if (root_excluded & unresolved or root_excluded & patched or unresolved & patched
            or (root_excluded | unresolved | patched) - available):
        raise ValueError('Reference lists overlap or contain missing files')
    if len(paths) != int(raw['scanned_count']) + len(root_excluded):
        raise ValueError('Quality report is incomplete for this reference library')
    if raw.get('excluded_count') is not None and len(root_excluded) != int(raw['excluded_count']):
        raise ValueError('Excluded-count mismatch')
    if unresolved and not exclude_unresolved:
        raise ValueError(
            f'{len(unresolved)} unresolved MTP references. To deliberately exclude them '
            'from training, pass --exclude-unresolved (source NPZ remain untouched).'
        )

    # A matching complete_changes.json is mandatory: prevents an incomplete or
    # mismatched status list from being accepted merely by editing the manifest.
    audit_path = quality/'complete_changes.json'
    if not audit_path.is_file():
        raise FileNotFoundError(f'Missing source audit: {audit_path}')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not isinstance(audit, list):
        raise ValueError('complete_changes.json must be a list')
    audit_status = {}
    for entry in audit:
        if not isinstance(entry, dict):
            raise ValueError('Invalid complete_changes.json entry')
        rel = entry.get('relative_reference')
        status = entry.get('status')
        if not isinstance(rel, str) or status not in ('unchanged', 'patched', 'unresolved_spike') or rel in audit_status:
            raise ValueError('Duplicate/unknown reference or invalid audit status')
        audit_status[rel] = status
    if set(audit_status) != available - root_excluded:
        raise ValueError('complete_changes.json does not cover every scanned reference exactly once')
    if {k for k,v in audit_status.items() if v == 'patched'} != patched:
        raise ValueError('Patched references disagree between manifest and audit')
    if {k for k,v in audit_status.items() if v == 'unresolved_spike'} != unresolved:
        raise ValueError('Unresolved references disagree between manifest and audit')

    for rel in patched:
        patch_path = quality/'patches'/Path(rel).parent/(Path(rel).stem+'_mtp_patch.npz')
        if not patch_path.is_file():
            raise FileNotFoundError(f'Missing declared MTP repair: {patch_path}')
    excluded = root_excluded | unresolved if exclude_unresolved else root_excluded
    return paths, excluded, patched, root_excluded, unresolved


def _patched_reference(source: Path, patch: Path, destination: Path) -> dict:
    """Apply a verified repair to a complete NPZ, retaining all other original keys."""
    with np.load(source, allow_pickle=False) as d:
        original = {key: np.array(d[key]) for key in d.files}
    with np.load(patch, allow_pickle=False) as d:
        rep = {key: np.array(d[key]) for key in d.files}
    needed = ('format_version', 'source_npz_sha256', 'source_n', 'source_duration_s',
              'mtp_joint_names', 'mtp_qpos_indices', 'mtp_qvel_indices',
              'qpos_mtp', 'qvel_mtp', 'site_names', 'site_pos_rel',
              'site_quat_rel', 'site_linvel_rel', 'site_angvel_rel')
    if any(k not in rep for k in needed):
        raise ValueError(f'Incomplete patch {patch}')
    if int(rep['format_version']) != 1 or str(rep['source_npz_sha256'].item()) != sha256(source):
        raise ValueError(f'Patch does not match source NPZ: {patch}')
    n = len(original['qpos'])
    duration = float(original['stride_duration_s'])
    if int(rep['source_n']) != n or not np.isclose(float(rep['source_duration_s']), duration, atol=1e-7, rtol=0):
        raise ValueError(f'Patch sample count/duration mismatch: {patch}')
    if tuple(str(x) for x in rep['mtp_joint_names'].tolist()) != ('mtp_angle_r', 'mtp_angle_l'):
        raise ValueError('MTP joint name/order mismatch')
    qi = np.asarray(rep['mtp_qpos_indices'], int).reshape(-1)
    vi = np.asarray(rep['mtp_qvel_indices'], int).reshape(-1)
    if (len(qi) != 2 or len(vi) != 2 or len(set(qi)) != 2 or len(set(vi)) != 2
            or np.any(qi < 7) or np.any(qi >= original['qpos'].shape[1])
            or np.any(vi < 6) or np.any(vi >= original['qvel'].shape[1])
            or rep['qpos_mtp'].shape != (n, 2) or rep['qvel_mtp'].shape != (n, 2)):
        raise ValueError('Invalid patch indices or joint array shapes')
    if tuple(str(x) for x in rep['site_names'].tolist()) != tuple(str(x) for x in original['site_names'].tolist()):
        raise ValueError('Patch site order mismatch')
    for key in ('site_pos_rel', 'site_quat_rel', 'site_angvel_rel'):
        if original[key].shape != rep[key].shape or not np.isfinite(rep[key]).all():
            raise ValueError(f'Patch has invalid {key}')
    site_vel_key = 'site_linvel_rel' if 'site_linvel_rel' in original else 'site_vel_rel'
    if original[site_vel_key].shape != rep['site_linvel_rel'].shape:
        raise ValueError('Patch site velocity shape mismatch')
    if not np.isfinite(rep['qpos_mtp']).all() or not np.isfinite(rep['qvel_mtp']).all():
        raise ValueError('Patch contains NaN/inf')
    original['qpos'][:, qi] = rep['qpos_mtp']
    original['qvel'][:, vi] = rep['qvel_mtp']
    original['site_pos_rel'] = rep['site_pos_rel']
    original['site_quat_rel'] = rep['site_quat_rel']
    original[site_vel_key] = rep['site_linvel_rel']
    original['site_angvel_rel'] = rep['site_angvel_rel']
    if not np.allclose(original['qpos'][0, qi], original['qpos'][-1, qi], atol=1e-8):
        raise ValueError('Patched MTP position not loopable')
    if not np.allclose(original['qvel'][0, vi], original['qvel'][-1, vi], atol=1e-8):
        raise ValueError('Patched MTP velocity not loopable')
    np.savez_compressed(destination, **original)
    # Read back the produced file rather than trusting an in-memory object.
    with np.load(destination, allow_pickle=False) as verified:
        if (not np.array_equal(verified['qpos'][:, qi], rep['qpos_mtp'])
                or not np.array_equal(verified['qvel'][:, vi], rep['qvel_mtp'])):
            raise ValueError(f'Written NPZ failed verification: {destination}')
    return dict(source_sha256=sha256(source),output_sha256=sha256(destination),
                changed_joints=['mtp_angle_r', 'mtp_angle_l'])


def build(source: Path, quality: Path, output: Path, model: Path | None = None,
          exclude_unresolved: bool = False) -> dict:
    source, quality, output = source.resolve(), quality.resolve(), output.resolve()
    if not source.is_dir() or not quality.is_dir():
        raise FileNotFoundError('Source library or quality report directory does not exist')
    if _inside(output, source) or _inside(output, quality) or _inside(source, output) or _inside(quality, output):
        raise ValueError('Output must be separate from BOTH the original references and quality artifacts')
    if output.exists():
        raise FileExistsError(f'Output exists; choose an empty NEW destination: {output}')
    manifest_path = quality/'quality_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    paths, excluded, patched, root_excluded, unresolved = _validate_manifest(
        manifest, source, quality, model, exclude_unresolved=exclude_unresolved)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-building-', dir=output.parent))
    changes = []
    try:
        for k, path in enumerate(paths, 1):
            rel = path.relative_to(source)
            rel_str = rel.as_posix()
            if rel_str in excluded:
                continue
            destination = stage/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if rel_str in patched:
                patch = quality/'patches'/rel.parent/(path.stem+'_mtp_patch.npz')
                audit = _patched_reference(path, patch, destination)
                changes.append({'reference':rel_str, **audit})
            else:
                # Unmodified references retain byte-for-byte identical source data.
                shutil.copy2(path, destination)
            # Keep all per-stride event/landmark sidecars under their original names.
            # Most importantly, LoopableReference requires *_loopable_manifest.json.
            for companion in path.parent.glob(path.stem + '_*'):
                if companion.is_file() and companion != path:
                    shutil.copy2(companion, destination.parent/companion.name)
            if not (destination.parent/(path.stem+'_manifest.json')).is_file():
                raise FileNotFoundError(f'Missing companion manifest for {rel_str}')
            if k % 200 == 0:
                print(f'Processed {k}/{len(paths)} original references')
        evidence = stage/'_preparation'
        evidence.mkdir(parents=True, exist_ok=True)
        for report in ('quality_manifest.json', 'mtp_repairs.csv', 'mtp_changed_intervals.csv',
                       'excluded_root_references.csv', 'mtp_rejected_repairs.csv',
                       'remaining_reference_summary.csv', 'complete_changes.json'):
            if (quality/report).is_file():
                shutil.copy2(quality/report, evidence/report)
        # Store the final, explicit curation decision separately from the unchanged
        # original quality report so it is clear why each source file was included.
        curation = dict(
            format_version=1,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            source_quality_manifest_sha256=sha256(manifest_path),
            source_complete_changes_sha256=sha256(quality/'complete_changes.json'),
            exclude_unresolved_requested=bool(exclude_unresolved),
            root_seam_exclusions=sorted(root_excluded),
            approved_unresolved_mtp_exclusions=sorted(unresolved),
            original_reference_count=len(paths),
            retained_reference_count=len(paths)-len(excluded),
            root_seam_excluded_count=len(root_excluded),
            unresolved_mtp_excluded_count=len(unresolved),
            patched_count=len(patched),
            unchanged_count=len(paths)-len(excluded)-len(patched),
        )
        (evidence/'curation_manifest.json').write_text(
            json.dumps(curation, indent=2), encoding='utf-8')
        with (evidence/'file_decisions.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['relative_reference', 'decision', 'reason'])
            writer.writeheader()
            for path in paths:
                rel_str = path.relative_to(source).as_posix()
                if rel_str in root_excluded:
                    decision, reason = 'excluded_root_seam', 'root_seam_speed_and_position_discontinuity'
                elif rel_str in unresolved:
                    decision, reason = 'excluded_unresolved_mtp', 'repair_failed_conservative_limits'
                elif rel_str in patched:
                    decision, reason = 'included_mtp_repaired', 'accepted_local_mtp_correction'
                else:
                    decision, reason = 'included_unchanged', 'no_mtp_repair_required'
                writer.writerow(dict(relative_reference=rel_str, decision=decision, reason=reason))
        result = dict(
            root_seam_exclusions=len(root_excluded),
            unresolved_mtp_exclusions=len(unresolved),
            output_references=len(paths)-len(excluded),
            excluded_references=len(excluded),
            mtp_repaired_references=len(changes),
            references_source=str(source),
            quality_source=str(quality),
            model_sha256=manifest.get('model_sha256'),
            note='Unmodified references copied verbatim; accepted MTP repairs materialized; explicitly declared exclusions omitted; source untouched.',
            repaired_files=changes,
        )
        (evidence/'build_manifest.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
        written = list(stage.rglob('*_loopable.npz'))
        if len(written) != len(paths)-len(excluded) or len(changes) != len(patched):
            raise ValueError('Output reference count does not match report')
        stage.rename(output)  # Finalize ONLY when all files and audits are complete.
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references', required=True, type=Path, help='Original reference library')
    p.add_argument('--quality-dir', required=True, type=Path, help='MTP repair artifacts/report directory')
    p.add_argument('--output', required=True, type=Path, help='NEW, independent training-library directory')
    p.add_argument('--model', type=Path, help='Same exact model used to generate the repair artifacts')
    p.add_argument('--exclude-unresolved', action='store_true',
                   help='Explicitly approve excluding ALL unresolved MTP references listed by the complete V3 audit')
    args = p.parse_args()
    result = build(args.references, args.quality_dir, args.output, args.model, args.exclude_unresolved)
    print(f'DONE: {result["output_references"]} reference files, '
          f'{result["root_seam_exclusions"]} root exclusions + '
          f'{result["unresolved_mtp_exclusions"]} unresolved MTP exclusions, '
          f'{result["mtp_repaired_references"]} MTP corrected.')
    print(f'Use --references "{args.output}" in the UNMODIFIED RL environment.')


if __name__ == '__main__':
    main()
