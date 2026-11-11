"""Export frozen inputs for replay and check the public sparse-depth experiment."""
import argparse
import numpy as np
from neighbor_sparse_depth import (ARMS, FRAMES, PACKET, PANEL, PARENT, PLAN, PUBLIC, ROOT,
                                  SEAL, REFS, aggregate, corrections, inputs, point_map,
                                  read, sha, tensor_digest, write_arrays, HISTORICAL)
from motion_reliability_surface import verify_map_archive


def export():
    rows = read(PARENT/'predictions.json')['rows']
    values = {}
    for key in read(SEAL)['batches']:
        _, depth, k, poses, _ = inputs(key, rows, read(PLAN))
        values[key+'_depth'] = depth.astype(np.float32)
        np.testing.assert_array_equal(values[key+'_depth'], depth)
        values[key+'_k'] = k
        values[key+'_poses'] = poses
    write_arrays(PUBLIC/'frozen_heads.npz', values)


def verify():
    seal = read(SEAL)
    scores = read(PUBLIC/'scores.json')
    assert scores['seal_sha256'] == sha(SEAL) and scores['panel_sha256'] == sha(PANEL)
    assert seal['design_sha256'] == sha(ROOT/'docs/NEIGHBOR_SPARSE_DEPTH.md')
    assert seal['packet_sha256'] == sha(PACKET)
    assert set(seal['batches']) == {'chess_base','heads_base','fire_base'}
    verify_map_archive(PUBLIC/'archive.json', {k:v['predictions'] for k,v in seal['batches'].items()})
    with np.load(PACKET) as data, np.load(PUBLIC/'frozen_heads.npz') as model:
        for key,batch in seal['batches'].items():
            slug = key.split('_')[0]
            samples = [data[f'{slug}_{f}'] for f in FRAMES]
            for frame, sample, count in zip(FRAMES, samples, batch['counts']):
                assert sample.shape == (count, 7) and np.isfinite(sample).all()
                assert len(np.unique(sample[:, :2], axis=0)) == count
                assert set(sample[:, 3]) == {frame-4, frame+4}
                assert np.all((sample[:, 4] % 80 == 40) & (sample[:, 5] % 60 == 30))
            depth = model[key+'_depth'].astype(float)
            k, poses = model[key+'_k'], model[key+'_poses']
            base = point_map(depth,k,poses,np.ones(len(depth)))
            assert tensor_digest(base) == batch['direct']
            updates, gauge = corrections(depth,samples)
            np.testing.assert_allclose(gauge,batch['gauge'],rtol=1e-12)
            for arm in ARMS:
                points = point_map(np.asarray(updates[arm]),k,poses,np.ones(len(depth)))
                # Float32 point hashes can differ across BLAS and libm implementations.
                # Compare a dense, fixed spatial subsample of every point map numerically.
                with np.load(PUBLIC/'replay_points.npz') as reference:
                    np.testing.assert_allclose(points[:,::9,::9],reference[key+'_'+arm],rtol=2e-6,atol=2e-6)
    axial = read(PUBLIC/'metric_depth.json')
    assert axial['seal_sha256'] == sha(SEAL)
    assert {(r['scene'].split('/')[0],r['frame']) for r in axial['rows']} == {
        (key.split('_')[0],frame) for key in seal['batches'] for frame in FRAMES}
    assert len(axial['rows']) == 15
    for arm in ('direct', *ARMS):
        values = [r['mean_absolute_log_depth'][arm] for r in axial['rows']]
        assert np.all(np.isfinite(values)) and min(values) >= 0
        np.testing.assert_allclose(np.mean(values),axial['mean_absolute_log_depth'][arm],rtol=1e-12)
    results = scores['results']
    measured = read(PUBLIC/'surface_observations.json')
    old = read(PANEL)['results']['references']
    rows = []
    for scene in old[REFS[0]]['scenes']:
        metrics = {}
        for ref in REFS:
            arms = results['references'][ref]['scenes'][scene]
            metrics[ref] = {a:v['raw'] for a,v in arms.items()}
            key = scene.split('/')[0] + '_base'
            for arm,digest in seal['batches'][key]['predictions'].items():
                observation = measured[key][arm][ref]
                assert observation['identity']['prediction'] == digest and observation['identity']['reference'] == ref
                assert observation['metrics'] == metrics[ref][arm]
            for a, v in old[ref]['scenes'][scene].items():
                assert metrics[ref]['previous_profile' if a=='guarded' else a] == v['raw']
        rows.append(dict(scene=scene,case='base',metrics=metrics))
    controls = tuple(a for a in metrics[REFS[0]] if a not in HISTORICAL+('guarded',))
    recomputed = aggregate(rows,controls+('guarded',),controls)
    for key in ('advance','checks','local_comparator','historical_comparator'):
        assert results[key] == recomputed[key]
    for key in ('local_gain','historical_gain'):
        np.testing.assert_allclose(results[key],recomputed[key],atol=1e-12)
    print('Replayed sparse-depth adaptation on all 15 targets and checked the transfer decision.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('export','verify'))
    args = parser.parse_args()
    if args.action == 'export':
        from neighbor_sparse_depth import RAW
        export()
        reference = {}
        for key, b in read(SEAL)['batches'].items():
            for arm,h in b['predictions'].items():
                with np.load(RAW/'points'/(h+'.npz')) as data:
                    assert tensor_digest(data['points']) == h
                    reference[key+'_'+arm] = data['points'][:,::9,::9]
        write_arrays(PUBLIC/'replay_points.npz',reference)
    else:
        verify()
