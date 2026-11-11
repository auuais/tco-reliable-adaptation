"""Replay affine residual fitting and validate the two-grid development comparison."""
import argparse

import numpy as np

from local_affine_sparse_depth import estimate
from motion_reliability_surface import verify_map_archive
from run_local_affine_sparse_depth import (PUBLIC,RAW,ROOT,GRIDS,HEADS,POSES,DESIGN,FRAMES,SCENES,REFS,HISTORICAL,
                                          aggregate,dump,place,point_map,read,sha,tensor_digest,write_arrays)


def replay(export=False,predictions_only=False):
    seal=read(PUBLIC/'predictions.json')
    assert seal['design_sha256']==sha(DESIGN)
    assert seal['solver_sha256']==sha(ROOT/'scripts/local_affine_sparse_depth.py')
    assert seal['inputs']['heads']==sha(HEADS) and seal['inputs']['poses']==sha(POSES)
    assert set(seal['batches'])==set(GRIDS)
    all_hashes={}
    samples3d={}
    with np.load(HEADS) as model,np.load(POSES) as poses:
        for grid,parent in GRIDS.items():
            packet=ROOT/'results/neighbor_sparse_depth/measurements.npz' if grid=='centered' else parent/'measurements.npz'
            assert seal['inputs'][grid]==sha(packet)
            assert set(seal['batches'][grid])==set(poses.files)
            with np.load(packet) as measurements:
                for key,b in seal['batches'][grid].items():
                    depth=model[key+'_depth'].astype(float)
                    ranges=[measurements[f'{key.split("_")[0]}_{frame}'] for frame in FRAMES]
                    outputs,gauge=estimate(depth,ranges)
                    np.testing.assert_allclose(gauge,b['gauge'],rtol=1e-12)
                    assert set(outputs)==set(b['hashes'])=={'affine_spatial','guarded'}
                    all_hashes[grid+'_'+key]=b['hashes']
                    for arm,corrected in outputs.items():
                        assert np.max(abs(np.log(corrected/depth)))<=.15+1e-12
                        points=point_map(corrected,model[key+'_k'],model[key+'_poses'],np.ones(5))
                        points=place(points,model[key+'_poses'],poses[key],gauge)
                        tag=grid+'_'+key+'_'+arm
                        if export:
                            h=b['hashes'][arm]
                            with np.load(RAW/grid/'points'/(h+'.npz')) as z:
                                original=z['points']
                            assert tensor_digest(original)==h
                            reference=original[:,::9,::9]
                            samples3d[tag]=reference
                        else:
                            with np.load(PUBLIC/'replay_points.npz') as z:
                                reference=z[tag]
                        np.testing.assert_allclose(points[:,::9,::9],reference,rtol=2e-6,atol=2e-6)
    if export:
        write_arrays(PUBLIC/'replay_points.npz',samples3d)
    verify_map_archive(PUBLIC/'archive.json',all_hashes)
    if not predictions_only:
        saved=read(PUBLIC/'scores.json')
        assert saved['seal_sha256']==sha(PUBLIC/'predictions.json')
        observations=read(PUBLIC/'observations.json')
        for grid,parent in GRIDS.items():
            old=read(parent/'scores.json')['results']['references']
            rows=[]
            for scene in SCENES:
                key=scene.split('/')[0]+'_base'
                measured=observations[grid][key]
                assert set(measured)==set(seal['batches'][grid][key]['hashes'].values())
                metrics={}
                for ref in REFS:
                    cell={('previous_depth_aware' if a=='guarded' else a):v['raw'] for a,v in old[ref]['scenes'][scene].items()}
                    cell.update({a:measured[h][ref] for a,h in seal['batches'][grid][key]['hashes'].items()})
                    metrics[ref]=cell
                rows.append(dict(scene=scene,case='base',metrics=metrics))
            controls=tuple(a for a in cell if a not in HISTORICAL+('guarded',))
            assert aggregate(rows,controls+('guarded',),controls)==saved['grids'][grid]
    print('Replayed twelve affine reconstructions and verified the archived inputs and available decisions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--predictions-only',action='store_true')
    args=p.parse_args()
    replay(args.export,args.predictions_only)
