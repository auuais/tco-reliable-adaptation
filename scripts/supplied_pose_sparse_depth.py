"""Hold the supplied target trajectory fixed for all four sparse depth methods."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neighbor_sparse_depth import (ARMS, FRAMES, HISTORICAL, PUBLIC as FIRST, RAW as FIRST_RAW,
                                  REFS, ROOT, SCENES, SEAL, aggregate, dump, evaluate_points,
                                  point_map, read, sha, tensor_digest, write_arrays)

PUBLIC = ROOT/'results/supplied_pose_sparse_depth'
RAW = ROOT/'reports/supplied_pose_sparse_depth'
METHODS = ('direct', *ARMS)
DESIGN = ROOT/'docs/SUPPLIED_POSE_SPARSE_DEPTH.md'


def place(points, pose, world_to_registered, gauge):
    camera = (np.einsum('nhwj,nij->nhwi', points, pose[:,:3,:3]) + pose[:,:3,3,None,None].transpose(0,2,3,1))
    inverse = np.linalg.inv(world_to_registered)
    return (np.einsum('nhwj,nij->nhwi', gauge*camera, inverse[:,:3,:3]) + inverse[:,None,None,:3,3]).astype(np.float32)


def predict(assets):
    original = read(SEAL)['batches']
    identity = read(ROOT/'results/source_frame_spacing/inputs.json')['frames']
    calibration = np.asarray(read(ROOT/'manifests/sensor_frame_v14.json')['calibration'])
    maps = RAW/'points'
    maps.mkdir(parents=True, exist_ok=True)
    batches = {}
    poses = {}
    with np.load(FIRST/'frozen_heads.npz') as heads:
        for scene in SCENES:
            key = scene.split('/')[0]+'_base'
            rows = []
            for frame in FRAMES:
                path = assets/scene/f'frame-{frame:06d}.pose.txt'
                if sha(path) != identity[scene][str(frame)]['original']['pose.txt']:
                    raise ValueError('Changed supplied pose')
                rows.append(calibration @ np.linalg.inv(np.loadtxt(path)))
            supplied = np.stack(rows)
            poses[key] = supplied
            b = original[key]
            model = heads[key+'_poses']
            hashes = {}
            for arm in METHODS:
                if arm == 'direct':
                    points = point_map(heads[key+'_depth'].astype(float),heads[key+'_k'],model,np.ones(len(FRAMES)))
                    expected = b['direct']
                else:
                    expected = b['predictions'][arm]
                    with np.load(FIRST_RAW/'points'/(expected+'.npz')) as z:
                        points = z['points']
                assert tensor_digest(points) == expected
                updated = place(points,model,supplied,b['gauge'])
                hashes[arm] = tensor_digest(updated)
                write_arrays(maps/(hashes[arm]+'.npz'),dict(points=updated))
            batches[key] = hashes
    PUBLIC.mkdir(parents=True,exist_ok=True)
    write_arrays(PUBLIC/'supplied_poses.npz',poses)
    record = dict(parent=sha(SEAL),supplied=sha(PUBLIC/'supplied_poses.npz'),batches=batches)
    target = PUBLIC/'predictions.json'
    if target.exists() and read(target)!=record:
        raise ValueError('Seal differs')
    dump(target,record)


def score(public=PUBLIC, raw=RAW):
    seal = read(public/'predictions.json')
    old = read(FIRST/'scores.json')['results']['references']
    rows = []
    with ProcessPoolExecutor(3) as pool:
        jobs = {key: pool.submit(evaluate_points,key,set(hashes.values()),raw) for key,hashes in seal['batches'].items()}
        for scene in SCENES:
            key = scene.split('/')[0]+'_base'
            _, observations = jobs[key].result()
            metrics = {}
            for ref in REFS:
                arms = {('model_pose_depth_aware' if a=='guarded' else a):v['raw']
                        for a,v in old[ref]['scenes'][scene].items()}
                for arm,h in seal['batches'][key].items():
                    arms['guarded' if arm=='guarded' else 'supplied_'+arm] = observations[h][ref]
                metrics[ref] = arms
            rows.append(dict(scene=scene,case='base',metrics=metrics))
    controls = tuple(a for a in arms if a not in HISTORICAL+('guarded',))
    result = aggregate(rows,controls+('guarded',),controls)
    dump(public/'scores.json',dict(seal_sha256=sha(public/'predictions.json'),results=result))
    print(result['checks'],result['local_gain'],flush=True)


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
        score()
