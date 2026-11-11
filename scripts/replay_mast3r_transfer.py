"""Independently resize native MASt3R depth and replay transfer predictions and decisions."""
import argparse

import numpy as np
from scipy.ndimage import map_coordinates
from motion_reliability_surface import verify_map_archive
from score_mast3r_transfer import (FRAMES, GRIDS, METHODS, PUBLIC, RAW, REFS, ROOT,
                                   corrections, dump, place, point_map, read, sha, summarize, tensor_digest, write_arrays)


def independently_resize(z,k):
    h,w=z.shape[1:]
    y,x=np.indices((392,518),dtype=float)
    coords=[(y+.5)*h/392-.5,(x+.5)*w/518-.5]
    depth=np.stack([map_coordinates(v.astype(float),coords,order=1,mode='nearest',prefilter=False) for v in z])
    scale=np.array([518/w,392/h])
    intrinsics=k.astype(float).copy()
    intrinsics[:,:2,:]*=scale[None,:,None]
    intrinsics[:,:2,2]+=(scale-1)/2
    return depth,intrinsics


def replay(export):
    seal=read(PUBLIC/'predictions.json')
    assert seal['grids']=={g:sha(p) for g,p in GRIDS.items()}
    samples3d,observations,all_hashes={},{},{}
    with np.load(ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz') as supplied:
        for key,b in seal['batches'].items():
            capture=read(PUBLIC/(key+'.json'))
            source=PUBLIC/'native_outputs'/(key+'.npz')
            assert sha(source)==capture['output_sha256'] and capture['pairs']==20
            with np.load(source) as native:
                np.testing.assert_array_equal((native['depth']<=0).sum(axis=(1,2)),capture['nonpositive_depth_pixels'])
                depth,k=independently_resize(native['depth'],native['intrinsics'])
                model=np.linalg.inv(native['c2w'].astype(float))[:,:3,:]
            positive=depth>0
            direct=point_map(depth,k,model,np.ones(len(FRAMES)))
            hashes={'frozen':b['frozen']}
            coords={'frozen':direct[:,::9,::9]}
            for grid,path in GRIDS.items():
                with np.load(path) as z:
                    ranges=[z[f'{key.split("_")[0]}_{f}'] for f in FRAMES]
                good=[s[positive[i,s[:,1].astype(int),s[:,0].astype(int)]] for i,s in enumerate(ranges)]
                updates,gauge=corrections(np.where(positive,depth,1.),good)
                np.testing.assert_allclose(gauge,b[grid]['gauge'],rtol=2e-6)
                updates['direct']=depth
                for arm in METHODS:
                    adjusted=np.where(positive,updates[arm],depth)
                    point=point_map(adjusted,k,model,np.ones(len(FRAMES)))[:,::9,::9]
                    tag=grid+'_'+arm
                    coords[tag]=place(point,model,supplied[key],gauge)
                    hashes[tag]=b[grid]['hashes'][arm]
            all_hashes[key]=hashes
            observations[key]={}
            for arm,digest in hashes.items():
                tag=key+'_'+arm
                if export:
                    with np.load(RAW/'points'/(digest+'.npz')) as z:
                        original=z['points']
                    assert tensor_digest(original)==digest
                    reference=original[:,::9,::9]
                    samples3d[tag]=reference
                    observations[key][digest]={r:read(RAW/'evaluation'/key/digest/r/'checked.json') for r in REFS}
                else:
                    with np.load(PUBLIC/'replay_points.npz') as z:
                        reference=z[tag]
                np.testing.assert_allclose(coords[arm],reference,rtol=1e-5,atol=2e-5)
    verify_map_archive(PUBLIC/'archive.json',all_hashes)
    if export:
        write_arrays(PUBLIC/'replay_points.npz',samples3d)
        dump(PUBLIC/'observations.json',observations)
    obs=read(PUBLIC/'observations.json')
    for key,hashes in all_hashes.items():
        assert set(obs[key])==set(hashes.values())
        for digest,refs in obs[key].items():
            for r,record in refs.items():
                assert record['identity']['prediction']==digest and record['identity']['reference']==r
    metrics={key:{h:{r:v['metrics'] for r,v in refs.items()} for h,refs in d.items()} for key,d in obs.items()}
    scores=read(PUBLIC/'scores.json')
    assert scores['seal_sha256']==sha(PUBLIC/'predictions.json')
    assert summarize(seal,metrics)==scores['grids']
    print('Replayed 27 transfer predictions and all 54 surface measurements in the comparison tables.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true')
    replay(p.parse_args().export)
