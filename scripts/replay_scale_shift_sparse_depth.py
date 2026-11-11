"""Regenerate robust sparse scale-shift predictions and verify both development decisions."""
import argparse

import numpy as np

from motion_reliability_surface import verify_map_archive
from run_scale_shift_sparse_depth import PUBLIC,RAW,DESIGN,HEADS,POSES,PACKETS,PARENTS,FRAMES,SCENES,ROOT,read,sha,point_map,place,tensor_digest,write_arrays
from run_local_affine_sparse_depth import aggregate,HISTORICAL,REFS
from scale_shift_sparse_depth import estimate


def replay(export=False,predictions_only=False):
    seal=read(PUBLIC/'predictions.json')
    assert seal['design_sha256']==sha(DESIGN)
    assert seal['solver_sha256']==sha(ROOT/'scripts/scale_shift_sparse_depth.py')
    assert seal['inputs']['heads']==sha(HEADS) and seal['inputs']['poses']==sha(POSES)
    samples3d={}
    all_hashes={}
    with np.load(HEADS) as model,np.load(POSES) as poses:
        for grid,path in PACKETS.items():
            assert seal['inputs']['packets'][grid]==sha(path)
            assert seal['inputs']['parents'][grid]==sha(PARENTS[grid]/'scores.json')
            with np.load(path) as data:
                for scene in SCENES:
                    key=scene.split('/')[0]+'_base'
                    saved=seal['batches'][grid][key]
                    depth=model[key+'_depth'].astype(float)
                    ranges=[data[f'{key.split("_")[0]}_{frame}'] for frame in FRAMES]
                    outputs,diagnostics=estimate(depth,ranges)
                    assert set(outputs)==set(saved['hashes'])=={'scene_affine','per_view_affine','guarded'}
                    np.testing.assert_allclose(diagnostics['initial_gauge'],saved['diagnostics']['initial_gauge'],rtol=1e-10)
                    for arm in ('scene_affine','per_view_affine'):
                        old=saved['diagnostics']['fits'][arm]
                        current=diagnostics['fits'][arm]
                        np.testing.assert_allclose(current['coefficients'],old['coefficients'],rtol=1e-8,atol=1e-10)
                        np.testing.assert_allclose(current['floor_fraction'],old['floor_fraction'],atol=1e-12)
                    all_hashes[grid+'_'+key]=saved['hashes']
                    for arm,output in outputs.items():
                        assert np.isfinite(output['depth']).all() and (output['depth']>0).all()
                        np.testing.assert_allclose(output['gauge'],saved['gauges'][arm],rtol=1e-10)
                        points=point_map(output['depth'],model[key+'_k'],model[key+'_poses'],np.ones(5))
                        points=place(points,model[key+'_poses'],poses[key],output['gauge'])
                        tag=grid+'_'+key+'_'+arm
                        if export:
                            h=saved['hashes'][arm]
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
        scores=read(PUBLIC/'scores.json')
        obs=read(PUBLIC/'observations.json')
        assert scores['seal_sha256']==sha(PUBLIC/'predictions.json')
        for grid,parent in PARENTS.items():
            old=read(parent/'scores.json')['grids'][grid]['references']
            rows=[]
            for scene in SCENES:
                key=scene.split('/')[0]+'_base'
                measured=obs[grid][key]
                assert set(measured)==set(seal['batches'][grid][key]['hashes'].values())
                metrics={}
                for ref in REFS:
                    cell={('affine_depth_aware' if a=='guarded' else a):v['raw'] for a,v in old[ref]['scenes'][scene].items()}
                    cell.update({a:measured[h][ref] for a,h in seal['batches'][grid][key]['hashes'].items()})
                    metrics[ref]=cell
                rows.append(dict(scene=scene,case='base',metrics=metrics))
            controls=tuple(a for a in cell if a not in HISTORICAL+('guarded',))
            assert aggregate(rows,controls+('guarded',),controls)==scores['grids'][grid]
    print('Replayed eighteen scale-shift reconstructions, coefficient diagnostics, and available decisions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--predictions-only',action='store_true')
    args=p.parse_args()
    replay(args.export,args.predictions_only)
