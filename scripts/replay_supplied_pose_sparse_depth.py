"""Replay the supplied-pose transform and the matched surface comparisons on a CPU."""
import argparse
import numpy as np
from motion_reliability_surface import verify_map_archive
from supplied_pose_sparse_depth import (FIRST, METHODS, PUBLIC, RAW, SEAL, REFS, HISTORICAL, aggregate,
                                       place, point_map, read, sha, tensor_digest, write_arrays, dump)


def run(export=False):
    path = PUBLIC/'predictions.json'
    seal = read(path)
    assert seal['parent']==sha(SEAL) and seal['supplied']==sha(PUBLIC/'supplied_poses.npz')
    verify_map_archive(PUBLIC/'archive.json',seal['batches'])
    evidence = {}
    observations = {}
    with np.load(FIRST/'frozen_heads.npz') as heads, np.load(FIRST/'replay_points.npz') as sparse:
        with np.load(PUBLIC/'supplied_poses.npz') as trajectory:
            for key,hashes in seal['batches'].items():
                model = heads[key+'_poses']
                base = point_map(heads[key+'_depth'].astype(float),heads[key+'_k'],model,np.ones(len(model)))
                supplied = trajectory[key]
                gauge = read(SEAL)['batches'][key]['gauge']
                observations[key] = {}
                for arm in METHODS:
                    tag = key+'_'+arm
                    original = base[:,::9,::9] if arm=='direct' else sparse[tag]
                    prediction = place(original,model,supplied,gauge)
                    # Check projected camera coordinates using independent homogeneous multiplication.
                    for i in range(len(model)):
                        world = original[i].reshape(-1,3).astype(float)
                        expected = (world @ model[i,:3,:3].T + model[i,:3,3]) * gauge
                        actual = prediction[i].reshape(-1,3).astype(float) @ supplied[i,:3,:3].T + supplied[i,:3,3]
                        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=2e-6)
                    if export:
                        with np.load(RAW/'points'/(hashes[arm]+'.npz')) as archive:
                            points = archive['points']
                        assert tensor_digest(points) == hashes[arm]
                        evidence[tag] = points[:,::9,::9]
                        observations[key][arm] = {r:read(RAW/'evaluation'/key/hashes[arm]/r/'checked.json') for r in REFS}
                    else:
                        with np.load(PUBLIC/'replay_points.npz') as archive:
                            reference = archive[tag]
                        np.testing.assert_allclose(prediction,reference,rtol=2e-6,atol=2e-6)
    if export:
        write_arrays(PUBLIC/'replay_points.npz',evidence)
        dump(PUBLIC/'surface_observations.json',observations)
    obs = read(PUBLIC/'surface_observations.json')
    scores = read(PUBLIC/'scores.json')
    old = read(FIRST/'scores.json')['results']['references']
    assert scores['seal_sha256'] == sha(path)
    rows = []
    for scene in old[REFS[0]]['scenes']:
        key = scene.split('/')[0]+'_base'
        metrics = {}
        for ref in REFS:
            cell = {('model_pose_depth_aware' if a=='guarded' else a):v['raw'] for a,v in old[ref]['scenes'][scene].items()}
            for arm in METHODS:
                v = obs[key][arm][ref]
                assert v['identity']['prediction']==seal['batches'][key][arm] and v['identity']['reference']==ref
                cell['guarded' if arm=='guarded' else 'supplied_'+arm] = v['metrics']
            metrics[ref] = cell
        rows.append(dict(scene=scene,case='base',metrics=metrics))
    controls = tuple(a for a in cell if a not in HISTORICAL+('guarded',))
    assert aggregate(rows,controls+('guarded',),controls) == scores['results']
    print('Verified 12 pose transforms, 24 surface evaluations and the full advancement decision.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    run(p.parse_args().export)
