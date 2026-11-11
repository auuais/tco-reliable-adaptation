"""Refit the same supplied-pose correction to a disjoint source-ray grid."""
import argparse
from pathlib import Path

import numpy as np
from neighbor_sparse_depth import FRAMES, corrections, packet
from supplied_pose_sparse_depth import (FIRST, PUBLIC as CENTERED, ROOT, METHODS, dump, place,
                                      point_map, read, score, sha, tensor_digest, write_arrays)

PUBLIC = ROOT/'results/shifted_sparse_grid'
RAW = ROOT/'reports/shifted_sparse_grid'


def predict(assets):
    path = PUBLIC/'measurements.npz'
    packet(assets,path,(20,15))
    maps = RAW/'points'
    maps.mkdir(parents=True,exist_ok=True)
    batches, observations = {}, {}
    with np.load(path) as ranges, np.load(FIRST/'frozen_heads.npz') as heads, np.load(CENTERED/'supplied_poses.npz') as poses:
        for key in poses:
            slug = key.split('_')[0]
            samples = [ranges[f'{slug}_{frame}'] for frame in FRAMES]
            assert all((s[:,4]%80==20).all() and (s[:,5]%60==15).all() and len(s)>=8 for s in samples)
            depth = heads[key+'_depth'].astype(float)
            updates,gauge = corrections(depth,samples)
            updates['direct'] = depth
            k,model = heads[key+'_k'],heads[key+'_poses']
            observations[key] = dict(gauge=gauge,counts=list(map(len,samples)))
            batches[key] = {}
            for arm in METHODS:
                model_points = point_map(np.asarray(updates[arm]),k,model,np.ones(len(FRAMES)))
                points = place(model_points,model,poses[key],gauge)
                h = tensor_digest(points)
                write_arrays(maps/(h+'.npz'),dict(points=points))
                batches[key][arm] = h
    record = dict(packet_sha256=sha(path), centered_seal_sha256=sha(CENTERED/'predictions.json'),
                  observations=observations,batches=batches)
    target = PUBLIC/'predictions.json'
    if target.exists() and read(target)!=record:
        raise ValueError('Changed prediction seal')
    dump(target,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','score'))
    p.add_argument('--assets',type=Path)
    args=p.parse_args()
    if args.action=='predict':
        if args.assets is None:
            p.error('--assets required')
        predict(args.assets)
    else:
        score(PUBLIC,RAW)
