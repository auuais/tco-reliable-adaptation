"""Seal and evaluate the complete local MASt3R joint-prior factorial."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
import shutil

import numpy as np
from evaluation_stability import dump,sha
from prepare_dense_focal_pilot import ROOT,write_arrays
from neighbor_sparse_depth import REFS,point_map,tensor_digest,evaluate_points
from score_mast3r_transfer import resample
from score_saved_geometry import read

PUBLIC=ROOT/'results/mast3r_joint_priors'
RAW=ROOT/'reports/mast3r_joint_priors'
OLD=ROOT/'results/mast3r_transfer'
GRIDS=('centered','shifted')
METRICS=('accuracy_mean_m','completeness_mean_m')


def predict():
    plan=read(PUBLIC/'alignment_plan.json')
    (RAW/'points').mkdir(parents=True,exist_ok=True)
    # Check every completion before creating or scoring a single point map.
    for job in plan['jobs']:
        folder=PUBLIC/'full'/job['run_id']
        done=read(folder/'complete.json')
        assert done['job']==job and done['plan_sha256']==sha(PUBLIC/'alignment_plan.json')
        assert done['trace_sha256']==sha(folder/'prior_trace.json')
        assert done['capture_sha256']==sha(folder/(job['scene']+'_base.json'))
    rows=[]
    for job in plan['jobs']:
        case=job['run_id']
        key=job['scene']+'_base'
        folder=PUBLIC/'full'/case
        done=read(folder/'complete.json')
        path=RAW/'full'/case/(key+'.npz')
        meta=read(folder/(key+'.json'))
        assert sha(path)==done['native_sha256']==meta['output_sha256']
        assert meta['input_sha256']==plan['rgb_sha256'][job['scene']]
        with np.load(path) as z:
            depth,k=resample(z['depth'],z['intrinsics'].astype(float))
            e=np.linalg.inv(z['c2w'].astype(float))[:,:3,:]
        points=point_map(depth,k,e,np.ones(len(depth)))
        assert np.isfinite(points).all()
        digest=tensor_digest(points)
        write_arrays(RAW/'points'/(digest+'.npz'),dict(points=points))
        destination=PUBLIC/'native_outputs'/(case+'.npz')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            assert sha(destination)==sha(path)
        else:
            shutil.copyfile(path,destination)
        rows.append(dict(**job,input_key=key,prediction=digest,native_sha256=sha(path)))
    seal=dict(plan_sha256=sha(PUBLIC/'alignment_plan.json'),
              historical_scores_sha256=sha(OLD/'scores.json'),
              historical_observations_sha256=sha(OLD/'observations.json'),rows=rows)
    path=PUBLIC/'predictions.json'
    if path.exists():
        assert seal==read(path)
    dump(path,seal)
    print('Sealed all',len(rows),'reconstructions before dense evaluation.',flush=True)


def summarize(seal,observations,historical):
    grids={}
    for grid in GRIDS:
        references={}
        for ref in REFS:
            scenes={}
            for slug in ('heads','chess','fire'):
                key=slug+'_base'
                rows=[r for r in seal['rows'] if r['scene']==slug and r['grid'] in ('none',grid)]
                raw={r['arm']:observations[key][r['prediction']][ref] for r in rows}
                base=raw['standard']
                # Historical controls use identical RGB, checkpoint, ranges and references.
                old=historical['grids'][grid]['references'][ref]['scenes'][key]
                raw.update({'output_'+a:v['raw'] for a,v in old.items() if a!='frozen'})
                cells={}
                for arm,v in raw.items():
                    pair=[v[m]/base[m] for m in METRICS]
                    cells[arm]=dict(score=mean(pair),accuracy=pair[0],completeness=pair[1],raw=v)
                scenes[key]=cells
            overall={a:{m:mean(c[a][m] for c in scenes.values()) for m in ('score','accuracy','completeness')} for a in cells}
            references[ref]=dict(scenes=scenes,overall=overall)
        primary=references[REFS[0]]['overall']
        best=min((a for a in primary if a!='joint'),key=lambda a:primary[a]['score'])
        gains={k:1-c['joint']['score']/c[best]['score'] for k,c in references[REFS[0]]['scenes'].items()}
        gain=1-primary['joint']['score']/primary[best]['score']
        secondary=references[REFS[1]]['overall']
        checks=dict(gain=gain>=.05,all_scenes=min(gains.values())>=0,
                    components=all(primary['joint'][m]<=1.02*primary[best][m] for m in ('accuracy','completeness')),
                    released=secondary['joint']['score']<=secondary[best]['score'])
        grids[grid]=dict(references=references,best_control=best,local_gain=gain,scene_gains=gains,
                         checks=checks,advance=all(checks.values()))
    return grids


def score():
    seal=read(PUBLIC/'predictions.json')
    assert seal['historical_scores_sha256']==sha(OLD/'scores.json')
    assert seal['historical_observations_sha256']==sha(OLD/'observations.json')
    observations={}
    with ProcessPoolExecutor(3) as pool:
        jobs={key:pool.submit(evaluate_points,key,{r['prediction'] for r in seal['rows'] if r['input_key']==key},RAW)
              for key in sorted({r['input_key'] for r in seal['rows']})}
        for key,job in jobs.items():
            _,observations[key]=job.result()
    dump(PUBLIC/'metrics.json',observations)
    # Preserve evaluator identity records as well as the compact scalar table.
    identities={key:{h:{ref:read(RAW/'evaluation'/key/h/ref/'checked.json') for ref in REFS} for h in v}
                for key,v in observations.items()}
    dump(PUBLIC/'observations.json',identities)
    result=dict(seal_sha256=sha(PUBLIC/'predictions.json'),
                grids=summarize(seal,observations,read(OLD/'scores.json')))
    dump(PUBLIC/'scores.json',result)
    print({g:(v['local_gain'],v['best_control'],v['checks']) for g,v in result['grids'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','score'))
    args=p.parse_args()
    (predict if args.action=='predict' else score)()
