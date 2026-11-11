"""Replay TUM calibration and reconstruction from public frozen VGGT outputs."""
import argparse

import numpy as np

from motion_reliability_surface import verify_map_archive
from run_tum_sparse_depth import (PUBLIC,RAW,SEQUENCES,GRIDS,METHODS,ROOT,calibrate,corrections,dump,
                                 point_map,place,read,sha,summarize,tensor_digest,write_arrays)


def replay(export=False,predictions_only=False):
    seal=read(PUBLIC/'predictions.json')
    assert seal['inputs_sha256']==sha(PUBLIC/'inputs.json')
    assert seal['calibration_solver_sha256']==sha(ROOT/'scripts/scale_shift_sparse_depth.py')
    assert seal['calibration_development_sha256']==sha(ROOT/'results/scale_shift_sparse_depth/scores.json')
    assert set(seal['batches'])==set(SEQUENCES)
    samples3d={}
    inventory={}
    with np.load(PUBLIC/'measurements.npz') as sensors:
        assert sha(PUBLIC/'measurements.npz')==read(PUBLIC/'inputs.json')['packet_sha256']
        for sequence in SEQUENCES:
            meta=PUBLIC/'captures'/(sequence+'.json')
            assert seal['captures'][sequence]==sha(meta)
            path=PUBLIC/'native_outputs'/(sequence+'.npz')
            if export:
                path.parent.mkdir(exist_ok=True)
                source=RAW/(sequence+'.npz')
                assert sha(source)==read(meta)['archive_sha256']
                import shutil
                if not path.exists():
                    shutil.copyfile(source,path)
            assert sha(path)==read(meta)['archive_sha256']
            with np.load(path) as z:
                depth=z['depth'][...,0].astype(float)
                model=z['extrinsics'].astype(float)
                native=z['native']
            k=sensors[sequence+'_intrinsics']
            supplied=sensors[sequence+'_poses']
            maps={'frozen':native,'direct':point_map(depth,k,model,np.ones(5))}
            for grid in GRIDS:
                values=[sensors[f'{sequence}_{grid}_{i}'] for i in range(5)]
                outputs,gauge=corrections(depth,values)
                outputs['direct']=depth
                saved=seal['batches'][sequence]['grids'][grid]
                np.testing.assert_allclose(gauge,saved['gauge'],rtol=1e-10)
                assert list(map(len,values))==saved['counts']
                for arm in METHODS:
                    name='uncalibrated_depth_aware' if arm=='guarded' else arm
                    points=point_map(np.asarray(outputs[arm]),k,model,np.ones(5))
                    maps[grid+'_'+name]=place(points,model,supplied,gauge)
                calibrated,diagnostics=calibrate(depth,values)
                for arm,fit in diagnostics['fits'].items():
                    old=saved['calibration_diagnostics']['fits'][arm]
                    np.testing.assert_allclose(fit['coefficients'],old['coefficients'],rtol=1e-8,atol=1e-10)
                    np.testing.assert_allclose(fit['floor_fraction'],old['floor_fraction'],atol=1e-12)
                for arm,value in calibrated.items():
                    np.testing.assert_allclose(value['gauge'],saved['calibrated_gauges'][arm],rtol=1e-10)
                    points=point_map(value['depth'],k,model,np.ones(5))
                    maps[grid+'_'+arm]=place(points,model,supplied,value['gauge'])
            hashes=seal['batches'][sequence]['hashes']
            assert set(maps)==set(hashes) and len(maps)==16
            inventory[sequence]=hashes
            for arm,points in maps.items():
                h=hashes[arm]
                tag=sequence+'_'+arm
                if export:
                    with np.load(RAW/'points'/(h+'.npz')) as z:
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
    verify_map_archive(PUBLIC/'archive.json',inventory)
    if not predictions_only:
        observations=read(PUBLIC/'observations.json')
        references=read(PUBLIC/'references.json')
        for sequence,b in seal['batches'].items():
            assert set(observations[sequence])==set(b['hashes'].values())
            for h,obs in observations[sequence].items():
                assert obs['identity']['prediction']==h
                assert obs['identity']['reference_sha256']==references['sequences'][sequence]['sha256']
        scores=read(PUBLIC/'scores.json')
        assert scores['seal_sha256']==sha(PUBLIC/'predictions.json')
        assert summarize(seal,observations)==scores['grids']
    print('Replayed all 48 TUM reconstructions and verified available score and calibration records.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--predictions-only',action='store_true')
    args=p.parse_args()
    replay(args.export,args.predictions_only)
