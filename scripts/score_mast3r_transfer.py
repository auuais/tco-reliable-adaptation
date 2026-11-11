"""Apply the frozen sparse-depth update to ordinary MASt3R aligned depth and cameras."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import numpy as np
from neighbor_sparse_depth import ROOT, FRAMES, SCENES, REFS, corrections, dump, evaluate_points, point_map, read, sha, tensor_digest, write_arrays
from supplied_pose_sparse_depth import METHODS, place

PUBLIC=ROOT/'results/mast3r_transfer'
RAW=ROOT/'reports/mast3r_transfer'
GRIDS={'centered':ROOT/'results/neighbor_sparse_depth/measurements.npz',
       'shifted':ROOT/'results/shifted_sparse_grid/measurements.npz'}


def resample(depth,k):
    import cv2
    height,width=depth.shape[1:]
    out_h,out_w=392,518
    resize=np.diag([out_w/width,out_h/height,1.])
    resize[:2,2]=(resize[[0,1],[0,1]]-1)/2
    adjusted=resize@k
    z=np.stack([cv2.resize(d.astype(float),(out_w,out_h),interpolation=cv2.INTER_LINEAR) for d in depth])
    return z,adjusted


def predict():
    maps=RAW/'points'
    maps.mkdir(parents=True,exist_ok=True)
    batches={}
    with np.load(ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz') as supplied:
        for scene in SCENES:
            slug=scene.split('/')[0]
            key=slug+'_base'
            capture=read(PUBLIC/(key+'.json'))
            path=RAW/(key+'.npz')
            assert sha(path)==capture['output_sha256']
            with np.load(path) as data:
                depth,k=resample(data['depth'],data['intrinsics'].astype(float))
                model=np.linalg.inv(data['c2w'].astype(float))[:,:3,:]
            direct=point_map(depth,k,model,np.ones(len(FRAMES)))
            h=tensor_digest(direct)
            write_arrays(maps/(h+'.npz'),dict(points=direct))
            batch={'frozen':h}
            for grid,packet in GRIDS.items():
                with np.load(packet) as z:
                    samples=[z[f'{slug}_{f}'] for f in FRAMES]
                positive=depth>0
                # Behind-camera predictions remain in every scored output. Only positive
                # axial depths can define logarithmic range residuals or receive updates.
                samples=[s[positive[i,s[:,1].astype(int),s[:,0].astype(int)]] for i,s in enumerate(samples)]
                assert all(len(s)>=8 for s in samples)
                updates,gauge=corrections(np.where(positive,depth,1.),samples)
                updates={arm:np.where(positive,value,depth) for arm,value in updates.items()}
                updates['direct']=depth
                hashes={}
                for method in METHODS:
                    original=point_map(np.asarray(updates[method]),k,model,np.ones(len(FRAMES)))
                    points=place(original,model,supplied[key],gauge)
                    h=tensor_digest(points)
                    write_arrays(maps/(h+'.npz'),dict(points=points))
                    hashes[method]=h
                batch[grid]=dict(gauge=gauge,hashes=hashes)
            batches[key]=batch
    record=dict(grids={g:sha(p) for g,p in GRIDS.items()},batches=batches)
    path=PUBLIC/'predictions.json'
    if path.exists() and record!=read(path):
        raise ValueError('Changed transfer predictions')
    dump(path,record)


def score():
    record=read(PUBLIC/'predictions.json')
    observations={}
    with ProcessPoolExecutor(3) as pool:
        jobs={}
        for key,b in record['batches'].items():
            hashes={b['frozen']}
            for grid in GRIDS:
                hashes.update(b[grid]['hashes'].values())
            jobs[key]=pool.submit(evaluate_points,key,hashes,RAW)
        for key,job in jobs.items():
            _,observations[key]=job.result()
    summary = summarize(record, observations)
    dump(PUBLIC/'scores.json',dict(seal_sha256=sha(PUBLIC/'predictions.json'),grids=summary))
    print({g:(v['local_gain'],v['checks']) for g,v in summary.items()},flush=True)


def summarize(record, observations):
    summary = {}
    for grid in GRIDS:
        comparison={}
        for ref in REFS:
            scenes={}
            for key,b in record['batches'].items():
                base=observations[key][b['frozen']][ref]
                cell={}
                for arm,h in dict(frozen=b['frozen'],**b[grid]['hashes']).items():
                    scores=observations[key][h][ref]
                    pair=[scores[m]/base[m] for m in ('accuracy_mean_m','completeness_mean_m')]
                    cell[arm]=dict(score=mean(pair),accuracy=pair[0],completeness=pair[1],raw=scores)
                scenes[key]=cell
            overall={a:{m:mean(v[a][m] for v in scenes.values()) for m in ('score','accuracy','completeness')} for a in cell}
            comparison[ref]=dict(overall=overall,scenes=scenes)
        primary=comparison[REFS[0]]['overall']
        best=min((a for a in primary if a!='guarded'),key=lambda a:primary[a]['score'])
        gains={k:1-c['guarded']['score']/c[best]['score'] for k,c in comparison[REFS[0]]['scenes'].items()}
        gain=1-primary['guarded']['score']/primary[best]['score']
        second=comparison[REFS[1]]['overall']
        checks=dict(gain=gain>=.05,all_scenes=min(gains.values())>=0,
                    components=all(primary['guarded'][m]<=1.02*primary[best][m] for m in ('accuracy','completeness')),
                    released=second['guarded']['score']<=second[best]['score'])
        summary[grid]=dict(references=comparison,best_control=best,local_gain=gain,scene_gains=gains,checks=checks,advance=all(checks.values()))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','score'))
    args=p.parse_args()
    (predict if args.action=='predict' else score)()
