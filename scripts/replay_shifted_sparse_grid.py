"""Replay the disjoint grid with the public original model heads and supplied trajectory."""
import argparse
import numpy as np
from shifted_sparse_grid import FIRST, CENTERED, PUBLIC, RAW, METHODS, FRAMES, read, sha, dump, place, point_map, corrections, write_arrays, tensor_digest
from supplied_pose_sparse_depth import REFS
from replay_supplied_pose_sparse_depth import verify_surface_scores
from motion_reliability_surface import verify_map_archive


def replay(export=False):
    seal = read(PUBLIC/'predictions.json')
    assert seal['packet_sha256']==sha(PUBLIC/'measurements.npz') and seal['centered_seal_sha256']==sha(CENTERED/'predictions.json')
    verify_map_archive(PUBLIC/'archive.json',seal['batches'])
    samples3d = {}
    observations = {}
    with np.load(PUBLIC/'measurements.npz') as packet, np.load(FIRST/'frozen_heads.npz') as model, np.load(CENTERED/'supplied_poses.npz') as poses:
        for key in poses:
            samples = [packet[f'{key.split("_")[0]}_{frame}'] for frame in FRAMES]
            assert [len(v) for v in samples]==seal['observations'][key]['counts']
            for frame,s in zip(FRAMES,samples):
                assert set(s[:,3])=={frame-4,frame+4} and (s[:,4]%80==20).all() and (s[:,5]%60==15).all()
            depth=model[key+'_depth'].astype(float)
            update,gauge=corrections(depth,samples)
            np.testing.assert_allclose(gauge,seal['observations'][key]['gauge'],rtol=1e-12)
            update['direct']=depth
            observations[key]={}
            for arm in METHODS:
                h=seal['batches'][key][arm]
                tag=key+'_'+arm
                point=point_map(np.asarray(update[arm]),model[key+'_k'],model[key+'_poses'],np.ones(len(depth)))
                sample=place(point[:,::9,::9],model[key+'_poses'],poses[key],gauge)
                if export:
                    with np.load(RAW/'points'/(h+'.npz')) as z:
                        original=z['points']
                    assert tensor_digest(original)==h
                    samples3d[tag]=original[:,::9,::9]
                    observations[key][arm]={r:read(RAW/'evaluation'/key/h/r/'checked.json') for r in REFS}
                    compare=samples3d[tag]
                else:
                    with np.load(PUBLIC/'replay_points.npz') as z:
                        compare=z[tag]
                np.testing.assert_allclose(sample,compare,rtol=2e-6,atol=2e-6)
    if export:
        write_arrays(PUBLIC/'replay_points.npz',samples3d)
        dump(PUBLIC/'surface_observations.json',observations)
    verify_surface_scores(PUBLIC, seal)
    print('Verified the shifted measurement grid, all 12 predictions and the replicated decision.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    replay(p.parse_args().export)
