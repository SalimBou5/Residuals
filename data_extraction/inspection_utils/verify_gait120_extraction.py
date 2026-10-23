"""Check the integrity of an extracted Gait120 -> MyoLeg reference.

Usage:
    python verify_gait120_extraction.py S012_T01_reference

Reads <prefix>.npz, <prefix>_manifest.json, <prefix>_force_events.csv.
Only checks data integrity; it does NOT validate anatomical fidelity or dynamics.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(prefix):
    prefix = Path(prefix)
    with np.load(str(prefix) + '.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    with open(str(prefix) + '_manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    with open(str(prefix) + '_force_events.csv', newline='', encoding='utf-8') as f:
        events = list(csv.DictReader(f))

    t = a['time_s']
    q = a['qpos']
    v = a['qvel']
    t_g = a['grf_time_s']
    force = a['grf_values']
    t_m = a['marker_time_s']
    markers = a['marker_xyz_os_m']

    assert q.shape == (len(t), 35) and v.shape == (len(t), 34)
    assert force.shape == (len(t_g), len(a['grf_channel_names']))
    assert markers.shape == (len(t_m), len(a['marker_names']), 3)
    assert all(np.isfinite(x).all() for x in [t, q, v, t_g, force, t_m, markers])
    assert all(np.all(np.diff(x) > 0) for x in [t, t_g, t_m])
    assert np.allclose(t, t_m, atol=1e-8)

    quat = q[:, 3:7]
    max_norm_error = float(np.max(np.abs(np.linalg.norm(quat, axis=1) - 1)))
    min_dot = float(np.min(np.einsum('ij,ij->i', quat[:-1], quat[1:])))
    assert max_norm_error < 1e-6 and min_dot > 0

    # In this MyoLeg model, all non-root joints are scalar hinge/slide joints.
    # Check their qvel values and the root's linear velocities against finite differences.
    scalar_diff = float(np.max(np.abs(np.gradient(q[:, 7:], t, axis=0, edge_order=1) - v[:, 6:])))
    linear_diff = float(np.max(np.abs(np.gradient(q[:, :3], t, axis=0, edge_order=1) - v[:, :3])))
    assert max(scalar_diff, linear_diff) < 1e-6

    for event in events:
        name = event['plate_channel']
        i = list(a['grf_channel_names']).index(name)
        vertical = force[:, i]
        threshold = float(event['threshold_N'])
        sign = np.diff((vertical > threshold).astype(int))
        direction = 1 if event['transition'] == 'on' else -1
        possible = np.flatnonzero(sign == direction)
        assert len(possible) > 0, f'No {event["transition"]} transition for {name}'
        times = []
        for j in possible:
            frac = (threshold - vertical[j]) / (vertical[j+1] - vertical[j])
            times.append(t_g[j] + frac * (t_g[j+1] - t_g[j]))
        assert np.min(np.abs(np.asarray(times) - float(event['time_s']))) < 1e-5

    print('PASS — file structure, timestamps, finite data, quaternion continuity')
    print('Frames:', len(t), '| time:', t[0], 'to', t[-1], 's')
    print('GRF samples:', len(t_g), '| marker frames:', len(t_m))
    print(f'Quaternion maximum norm error: {max_norm_error:.2e}, minimum consecutive dot: {min_dot:.9f}')
    print(f'qvel vs finite differences: max scalar {scalar_diff:.2e}, root linear {linear_diff:.2e}')
    print('Root position displacement (m):', np.round(q[-1, :3] - q[0, :3], 4))
    print('Verified force events:', len(events), '(foot-side assignments not validated)')
    for e in events:
        print(' ', e['plate_channel'], e['transition'], 'at', round(float(e['time_s']), 6), 's')
    print('Reported knee equality residual:', manifest['max_equality_residual'], '(manifest; not recomputed)')
    print('Status:', manifest['status'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('prefix', help='Common prefix of the three extracted files')
    main(ap.parse_args().prefix)
