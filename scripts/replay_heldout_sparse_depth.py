"""Replay all held-out range updates from public frozen depth heads."""
import argparse

import numpy as np
from motion_reliability_surface import verify_map_archive
from run_heldout_sparse_depth import (GRIDS, METHODS, PUBLIC, RAW, ROOT, corrections, dump,
                                    place, point_map, read, sha, summarize, tensor_digest, write_arrays)


def replay(export=False, require_scores=True):
    seal=read(PUBLIC/'predictions.json')
    index=read(PUBLIC/'input_index.json')
    assert seal['input_sha256']==sha(PUBLIC/'input_index.json')
    assert seal['packet_sha256']==sha(PUBLIC/'measurements.npz')
    assert len(seal['batches'])==13
    assert set(seal['batches'])==set(index['sequences'])
    assert sum(len(r['targets']) for r in index['sequences'].values())==61
    policy=read(ROOT/'manifests/holdout_policy_v1.json')
    assert {s.split('/')[0] for s in seal['batches']}==set(policy['heldout_scenes'])
    public_heads=PUBLIC/'frozen_heads'
    if export:
        public_heads.mkdir(exist_ok=True)
    samples3d={}
    poses_path=PUBLIC/'supplied_poses.npz'
    with np.load(PUBLIC/'measurements.npz') as packet:
        for scene,entry in index['sequences'].items():
            key=scene.replace('/','_')
            b=seal['batches'][scene]
            capture=PUBLIC/'captures'/(key+'.json')
            assert sha(capture)==b['capture_sha256']
            path=public_heads/(key+'.npz')
            if export:
                source=RAW/(key+'.npz')
                assert sha(source)==read(capture)['archive_sha256']
                with np.load(source) as z:
                    fields={n:z[n] for n in ('depth','intrinsics','extrinsics')}
                write_arrays(path,fields)
            with np.load(path) as z:
                depth=z['depth'][...,0].astype(float)
                k=z['intrinsics'].astype(float)
                model=z['extrinsics'].astype(float)
            with np.load(poses_path) as z:
                supplied=z[key]
            assert len(depth)==len(entry['targets'])
            k[:,[0,1],[0,1]]=np.array([475.9125061035156,480.20001220703125])
            pointmaps={'direct':point_map(depth,k,model,np.ones(len(depth)))[:,::17,::17]}
            for grid,origin in GRIDS.items():
                samples=[packet[f'{key}_{grid}_{f}'] for f in entry['targets']]
                assert list(map(len,samples))==b['grids'][grid]['counts']
                for frame,s in zip(entry['targets'],samples):
                    assert s.shape[1]==7 and len(s)>=8 and np.isfinite(s).all()
                    assert set(s[:,3])=={frame-4,frame+4}
                    assert (s[:,4]%80==origin[0]).all() and (s[:,5]%60==origin[1]).all()
                updates,gauge=corrections(depth,samples)
                np.testing.assert_allclose(gauge,b['grids'][grid]['gauge'],rtol=1e-12)
                updates['direct']=depth
                for arm in METHODS:
                    original=point_map(np.asarray(updates[arm]),k,model,np.ones(len(depth)))[:,::17,::17]
                    pointmaps[grid+'_'+arm]=place(original,model,supplied,gauge)
            for arm,h in b['hashes'].items():
                tag=key+'_'+arm
                if export:
                    with np.load(RAW/'points'/(h+'.npz')) as z:
                        full=z['points']
                    assert tensor_digest(full)==h
                    samples3d[tag]=full[:,::17,::17]
                    expected=samples3d[tag]
                else:
                    with np.load(PUBLIC/'replay_points.npz') as z:
                        expected=z[tag]
                if arm!='frozen':
                    np.testing.assert_allclose(pointmaps[arm],expected,rtol=2e-6,atol=2e-6)
    if export:
        write_arrays(PUBLIC/'replay_points.npz',samples3d)
    verify_map_archive(PUBLIC/'archive.json',{s:b['hashes'] for s,b in seal['batches'].items()})
    if require_scores:
        scores=read(PUBLIC/'scores.json')
        assert scores['seal_sha256']==sha(PUBLIC/'predictions.json')
        obs=read(PUBLIC/'observations.json')
        for scene,b in seal['batches'].items():
            assert set(obs[scene])==set(b['hashes'].values())
            for h,refs in obs[scene].items():
                assert set(refs)=={'registered_rgb','released'}
                for ref,r in refs.items():
                    assert r['prediction']==h and r['reference']==ref
        assert scores['grids']==summarize(seal,obs)
    print('Replayed 13 held-out sequences and verified all frozen decision inputs.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    p.add_argument('--predictions-only',action='store_true')
    args=p.parse_args()
    replay(args.export,not args.predictions_only)
