"""Prepare and score local affine updates only on the original development scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from local_affine_sparse_depth import estimate
from neighbor_sparse_depth import (ROOT,FRAMES,SCENES,REFS,HISTORICAL,aggregate,dump,evaluate_points,
                                  point_map,read,sha,tensor_digest,write_arrays)
from supplied_pose_sparse_depth import place

PUBLIC=ROOT/'results/local_affine_sparse_depth'
RAW=ROOT/'reports/local_affine_sparse_depth'
DESIGN=ROOT/'docs/LOCAL_AFFINE_SPARSE_DEPTH.md'
GRIDS={'centered':ROOT/'results/supplied_pose_sparse_depth','shifted':ROOT/'results/shifted_sparse_grid'}
HEADS=ROOT/'results/neighbor_sparse_depth/frozen_heads.npz'
POSES=ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz'


def prepare():
    batches={}
    inputs={'heads':sha(HEADS),'poses':sha(POSES)}
    with np.load(HEADS) as model,np.load(POSES) as poses:
        for grid,parent in GRIDS.items():
            packet=ROOT/'results/neighbor_sparse_depth/measurements.npz' if grid=='centered' else parent/'measurements.npz'
            inputs[grid]=sha(packet)
            batches[grid]={}
            maps=RAW/grid/'points'
            maps.mkdir(parents=True,exist_ok=True)
            with np.load(packet) as observations:
                for scene in SCENES:
                    slug=scene.split('/')[0]
                    key=slug+'_base'
                    depth=model[key+'_depth'].astype(float)
                    samples=[observations[f'{slug}_{frame}'] for frame in FRAMES]
                    outputs,gauge=estimate(depth,samples)
                    hashes={}
                    for arm,adjusted in outputs.items():
                        points=point_map(adjusted,model[key+'_k'],model[key+'_poses'],np.ones(5))
                        points=place(points,model[key+'_poses'],poses[key],gauge)
                        h=tensor_digest(points)
                        write_arrays(maps/(h+'.npz'),dict(points=points))
                        hashes[arm]=h
                    batches[grid][key]=dict(gauge=gauge,hashes=hashes)
                    print(grid,scene,'two affine maps sealed',flush=True)
    result=dict(design_sha256=sha(DESIGN),solver_sha256=sha(ROOT/'scripts/local_affine_sparse_depth.py'),inputs=inputs,batches=batches)
    path=PUBLIC/'predictions.json'
    if path.exists() and read(path)!=result:
        raise ValueError('Changed sealed affine predictions')
    dump(path,result)


def score(public=PUBLIC,raw=RAW,parents=GRIDS,previous_name="previous_depth_aware"):
    seal=read(public/'predictions.json')
    outcomes={}
    observations={}
    for grid,parent in parents.items():
        parent_scores=read(parent/'scores.json')
        old=(parent_scores['grids'][grid] if 'grids' in parent_scores else parent_scores['results'])['references']
        rows=[]
        observations[grid]={}
        with ProcessPoolExecutor(3) as pool:
            jobs={key:pool.submit(evaluate_points,key,set(b['hashes'].values()),raw/grid) for key,b in seal['batches'][grid].items()}
            for scene in SCENES:
                key=scene.split('/')[0]+'_base'
                _,measured=jobs[key].result()
                observations[grid][key]=measured
                metrics={}
                for ref in REFS:
                    cell={(previous_name if a=='guarded' else a):v['raw'] for a,v in old[ref]['scenes'][scene].items()}
                    cell.update({a:measured[h][ref] for a,h in seal['batches'][grid][key]['hashes'].items()})
                    metrics[ref]=cell
                rows.append(dict(scene=scene,case='base',metrics=metrics))
        controls=tuple(a for a in cell if a not in HISTORICAL+('guarded',))
        outcomes[grid]=aggregate(rows,controls+('guarded',),controls)
        print(grid,outcomes[grid]['local_gain'],outcomes[grid]['checks'],flush=True)
    dump(public/'observations.json',observations)
    dump(public/'scores.json',dict(seal_sha256=sha(public/'predictions.json'),grids=outcomes))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','score'))
    args=p.parse_args()
    (prepare if args.action=='prepare' else score)()
