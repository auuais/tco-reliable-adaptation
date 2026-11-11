"""Recover each final camera from its own point map and check the physical focal target."""
import argparse
import hashlib
import json
from statistics import mean
from zipfile import ZipFile

import numpy as np

from evaluation_stability import dump, sha
from pointmap_camera_diagnostic import recover
from run_dense_focal_pilot import RUNS
from score_dense_focal_pilot import PUBLIC
from score_saved_geometry import read


def summarize(rows):
    groups = {}
    for arm in ('retain', 'consensus', 'adaptive'):
        group = [r for r in rows if r['arm'] == arm]
        table = {}
        for case in ('all', 'metadata_fault', 'valid_zoom'):
            subset = [r for r in group if case == 'all' or r['case'] == case]
            entry = dict(n=len(subset), valid_views=sum(v['fit'] is not None for r in subset for v in r['views']))
            if entry['valid_views'] == 5 * len(subset):
                for selection in ('affected', 'neighbors'):
                    views = [v for r in subset for v in r['views'] if v['affected'] == (selection == 'affected')]
                    entry[selection] = {stage:mean(v[stage] for v in views) for stage in ('initial_log_error','final_log_error')}
            table[case] = entry
        groups[arm] = table
    return groups


def run():
    source = PUBLIC/'seal.json'
    rows, maps = [], {}
    for row in read(source)['rows']:
        job, tag = row['job'], row['tag']
        folder = RUNS/tag
        if sha(folder/'run.json') != row['run_sha256']:
            raise ValueError('Changed run record')
        d = read(folder/'run.json')['decision']
        supplied = np.asarray(d['supplied_focal'], dtype=np.float64)
        target = supplied.copy()
        idx = job['view']
        if job['case'] == 'metadata_fault':
            target[idx] = np.median(np.delete(supplied, idx, axis=0), axis=0)
        initial = np.asarray(d['model_focal'])
        views = []
        for stage in ('initial','final'):
            meta = row[stage]
            identity = (stage, meta['tensor_sha256'])
            if stage == 'initial' and identity in maps:
                continue
            path = folder/meta['file']
            if sha(path) != meta['sha256']:
                raise ValueError('Changed point map')
            with np.load(path, allow_pickle=False) as z:
                xyz = z['points']
            if xyz.dtype != np.float32 or hashlib.sha256(np.ascontiguousarray(xyz).tobytes()).hexdigest() != identity[1]:
                raise ValueError('Changed tensor or unexpected export dtype')
            maps[identity] = dict(phase=stage, tensor_sha256=identity[1], coordinate_count=int(xyz.size),
                                  exact_bfloat16_count=int(np.count_nonzero((xyz.view(np.uint32) & 65535) == 0)))
            if stage == 'initial':
                continue
            for i, points in enumerate(xyz):
                v = dict(view=i, affected=i==idx, physical_target=target[i].tolist(), initial_focal=initial[i].tolist(),
                         initial_log_error=float(np.mean(np.abs(np.log(initial[i]/target[i])))))
                try:
                    camera, _ = recover(points)
                except ValueError as exc:
                    v.update(fit=None, reason=str(exc))
                else:
                    f = camera['intrinsics'][[0,1],[0,1]]
                    v.update(fit=f.tolist(), max_reprojection_px=camera['dense_reprojection_max_pixels'],
                             final_log_error=float(np.mean(np.abs(np.log(f/target[i])))))
                views.append(v)
        rows.append(dict(tag=tag, arm=job['arm'], case=job['case'], scene=job['scene'], seed=job['seed'], views=views))
    return dict(seal_sha256=sha(source), rows=rows, results=summarize(rows)), dict(
        seal_sha256=sha(source), definition='Among finite exported float32 coordinates, count those whose low 16 mantissa bits are zero.',
        maps=list(maps.values()))


def replay():
    seal_path = PUBLIC/'seal.json'
    seal = read(seal_path)['rows']
    outcome = read(PUBLIC/'camera_readout.json')
    precision = read(PUBLIC/'point_precision.json')
    if outcome['seal_sha256'] != sha(seal_path) or precision['seal_sha256'] != sha(seal_path):
        raise ValueError('Diagnostic refers to a different prediction seal')
    if [r['tag'] for r in outcome['rows']] != [r['tag'] for r in seal] or outcome['results'] != summarize(outcome['rows']):
        raise ValueError('Camera readout table differs from its observed rows')
    identities = {(stage, r[stage]['tensor_sha256']) for r in seal for stage in ('initial','final')}
    if {(m['phase'],m['tensor_sha256']) for m in precision['maps']} != identities or len(precision['maps']) != len(identities):
        raise ValueError('Coordinate precision inventory differs')
    for m in precision['maps']:
        if m['coordinate_count'] != 5*392*518*3 or not 0 <= m['exact_bfloat16_count'] <= m['coordinate_count']:
            raise ValueError('Unexpected coordinate precision counts')
    with ZipFile(PUBLIC/'gpu_traces.zip') as z:
        for r, entry in zip(outcome['rows'], seal):
            j = entry['job']
            d = json.loads(z.read(r['tag']+'/run.json'))['decision']
            if any(r[k] != j[k] for k in ('arm','case','scene','seed')) or [v['view'] for v in r['views']] != list(range(5)):
                raise ValueError('Camera diagnostic identity changed')
            target = np.asarray(d['supplied_focal'])
            if j['case'] == 'metadata_fault':
                target[j['view']] = np.median(np.delete(target, j['view'], axis=0), axis=0)
            for i,v in enumerate(r['views']):
                np.testing.assert_array_equal(v['physical_target'],target[i])
                np.testing.assert_array_equal(v['initial_focal'],d['model_focal'][i])
                if v['affected'] != (i == j['view']):
                    raise ValueError('Affected view changed')
                for stage, focal in [('initial',v['initial_focal']),('final',v['fit'])]:
                    if focal is not None:
                        error = np.abs(np.log(focal)-np.log(target[i])).mean()
                        np.testing.assert_allclose(v[stage+'_log_error'],error,rtol=1e-10,atol=1e-14)
                if v['fit'] is not None and not 0 <= v['max_reprojection_px'] <= .02:
                    raise ValueError('Camera recovery exceeded its reprojection tolerance')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('write','verify','replay'))
    args = p.parse_args()
    if args.action == 'replay':
        replay()
        print('Replayed the published 360 camera readouts and both coordinate-precision inventories.')
        raise SystemExit(0)
    outcome, precision = run()
    for name, values in [('camera_readout.json', outcome), ('point_precision.json', precision)]:
        path = PUBLIC/name
        if args.action == 'write' and not path.exists():
            dump(path, values)
        elif read(path) != values:
            raise ValueError('Existing diagnostics differ: ' + name)
    print('Verified final camera self-projection and coordinate precision across the dense pilot.')
