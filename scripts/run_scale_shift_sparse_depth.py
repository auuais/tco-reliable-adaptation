"""Seal the classical scale-and-shift calibration controls and the fixed local residual update."""
import argparse

import numpy as np

from scale_shift_sparse_depth import estimate
from run_local_affine_sparse_depth import ROOT,HEADS,POSES,FRAMES,SCENES,dump,point_map,read,sha,tensor_digest,write_arrays,score
from supplied_pose_sparse_depth import place

PUBLIC=ROOT/'results/scale_shift_sparse_depth'
RAW=ROOT/'reports/scale_shift_sparse_depth'
DESIGN=ROOT/'docs/SCALE_SHIFT_SPARSE_DEPTH.md'
PARENTS={grid:ROOT/'results/local_affine_sparse_depth' for grid in ('centered','shifted')}
PACKETS={'centered':ROOT/'results/neighbor_sparse_depth/measurements.npz','shifted':ROOT/'results/shifted_sparse_grid/measurements.npz'}


def prepare():
    batches={}
    sources=dict(heads=sha(HEADS),poses=sha(POSES),packets={g:sha(p) for g,p in PACKETS.items()},
                 parents={g:sha(p/'scores.json') for g,p in PARENTS.items()})
    with np.load(HEADS) as model,np.load(POSES) as poses:
        for grid,packet in PACKETS.items():
            maps=RAW/grid/'points'
            maps.mkdir(parents=True,exist_ok=True)
            batches[grid]={}
            with np.load(packet) as data:
                for scene in SCENES:
                    slug=scene.split('/')[0]
                    key=slug+'_base'
                    samples=[data[f'{slug}_{frame}'] for frame in FRAMES]
                    outputs,diagnostics=estimate(model[key+'_depth'].astype(float),samples)
                    hashes={}
                    gauges={}
                    for arm,value in outputs.items():
                        points=point_map(value['depth'],model[key+'_k'],model[key+'_poses'],np.ones(5))
                        points=place(points,model[key+'_poses'],poses[key],value['gauge'])
                        h=tensor_digest(points)
                        write_arrays(maps/(h+'.npz'),dict(points=points))
                        hashes[arm]=h
                        gauges[arm]=value['gauge']
                    batches[grid][key]=dict(hashes=hashes,gauges=gauges,diagnostics=diagnostics)
                    print(grid,scene,'three calibrated maps sealed',flush=True)
    result=dict(design_sha256=sha(DESIGN),solver_sha256=sha(ROOT/'scripts/scale_shift_sparse_depth.py'),inputs=sources,batches=batches)
    path=PUBLIC/'predictions.json'
    if path.exists() and read(path)!=result:
        raise ValueError('Changed calibration predictions')
    dump(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','score'))
    args=p.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        score(PUBLIC,RAW,PARENTS,'affine_depth_aware')
