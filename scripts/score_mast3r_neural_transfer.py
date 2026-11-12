"""Seal all neural-transfer outputs before evaluating both reconstruction references."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import shutil
from statistics import mean
import numpy as np
from verify_mast3r_neural_plan import verify,PUBLIC,ROOT
from prepare_dense_focal_pilot import write_arrays
from evaluation_stability import dump,sha
from score_saved_geometry import read
from score_mast3r_transfer import resample
from neighbor_sparse_depth import point_map,tensor_digest,evaluate_points,REFS

RAW=ROOT/'reports/mast3r_neural_transfer'
OLD=ROOT/'results/mast3r_joint_priors'


def predict():
    plan=verify()
    for job in plan['jobs']:
        public=PUBLIC/'cases'/job['case'];raw=RAW/'cases'/job['case']
        done=read(public/'complete.json')
        assert done['job']==job and done['plan_sha256']==sha(PUBLIC/'plan.json')
        for name,h in done['files'].items():assert sha(raw/name)==h
        for name,h in done['metadata_sha256'].items():assert sha(public/name)==h
    (RAW/'points').mkdir(parents=True,exist_ok=True)
    (PUBLIC/'native_outputs').mkdir(exist_ok=True)
    rows=[];baseline={}
    for job in plan['jobs']:
        public=PUBLIC/'cases'/job['case'];raw=RAW/'cases'/job['case']
        for post in job['post']:
            path=raw/(post+'.npz');meta=read(public/(post+'.json'))
            assert sha(path)==meta['output_sha256']
            assert meta['input_sha256']==plan['rgb_sha256'][job['scene']]
            with np.load(path) as z:
                if job['arm']=='zero' and post=='ordinary':
                    with np.load(ROOT/'results/mast3r_transfer/native_outputs'/(job['scene']+'_base.npz')) as old:
                        baseline[job['scene']]={n:bool(np.array_equal(z[n],old[n])) for n in z}
                    assert all(baseline[job['scene']].values()),'Zero-adapter baseline changed'
                depth,k=resample(z['depth'],z['intrinsics'].astype(float))
                e=np.linalg.inv(z['c2w'].astype(float))[:,:3,:]
            points=point_map(depth,k,e,np.ones(len(depth)))
            assert np.isfinite(points).all()
            h=tensor_digest(points)
            write_arrays(RAW/'points'/(h+'.npz'),dict(points=points))
            tag=job['case']+'_'+post
            destination=PUBLIC/'native_outputs'/(tag+'.npz')
            if destination.exists():assert sha(destination)==sha(path)
            else:shutil.copyfile(path,destination)
            rows.append(dict(scene=job['scene'],grid=job['grid'],arm=job['arm'],case=job['case'],post=post,
                             tag=tag,input_key=job['scene']+'_base',prediction=h,native_sha256=sha(path)))
    assert len(rows)==105
    seal=dict(plan_sha256=sha(PUBLIC/'plan.json'),rows=rows,baseline_exact=baseline,
              historical_scores_sha256=sha(OLD/'scores.json'),historical_observations_sha256=sha(OLD/'observations.json'),
              scoring_source_sha256=hashlib.sha256((ROOT/'scripts/score_mast3r_neural_transfer.py').read_text(encoding='utf-8').encode()).hexdigest())
    if (PUBLIC/'predictions.json').exists():assert read(PUBLIC/'predictions.json')==seal
    dump(PUBLIC/'predictions.json',seal)
    print('Sealed all 105 reconstructions before dense-reference evaluation.',flush=True)


def summarize(seal,observations,historical):
    grids={}
    for grid in ('centered','shifted'):
        conditions={}
        for condition in ('clean','noisy'):
            candidate=('J' if condition=='clean' else 'N')+'_KDP_'+condition
            references={}
            for ref in REFS:
                scenes={}
                for slug in ('heads','chess','fire'):
                    key=slug+'_base'
                    rows=[r for r in seal['rows'] if r['scene']==slug and r['grid'] in ('none',grid)]
                    raw={r['arm']+'_'+r['post']:observations[key][r['prediction']][ref] for r in rows}
                    base=raw['zero_ordinary']
                    old=historical['grids'][grid]['references'][ref]['scenes'][key]
                    # Depth-only historical alignment does not supervise the pose channel.
                    for a in ('standard','depth','pose','joint'):
                        if condition=='noisy' and a in ('pose','joint'):continue
                        raw['historical_'+a]=old[a]['raw']
                    cell={}
                    for a,s in raw.items():
                        pair=[s[m]/base[m] for m in ('accuracy_mean_m','completeness_mean_m')]
                        cell[a]=dict(score=mean(pair),accuracy=pair[0],completeness=pair[1],raw=s)
                    scenes[key]=cell
                overall={a:{m:mean(c[a][m] for c in scenes.values()) for m in ('score','accuracy','completeness')} for a in cell}
                references[ref]=dict(scenes=scenes,overall=overall)
            primary=references[REFS[0]]['overall']
            # J/N are candidate families, so neither is selected post hoc as a control.
            eligible=[a for a in primary if not a.startswith(('J_','N_')) and (not a.endswith('KDP_noisy') if condition=='clean' else not a.endswith('KDP_clean'))]
            best=min(eligible,key=lambda a:primary[a]['score'])
            gain=1-primary[candidate]['score']/primary[best]['score']
            scene_gains={k:1-c[candidate]['score']/c[best]['score'] for k,c in references[REFS[0]]['scenes'].items()}
            secondary=references[REFS[1]]['overall']
            checks=dict(gain=gain>=.05,all_scenes=min(scene_gains.values())>=0,
                components=all(primary[candidate][m]<=1.02*primary[best][m] for m in ('accuracy','completeness')),
                released=secondary[candidate]['score']<=secondary[best]['score'])
            conditions[condition]=dict(candidate=candidate,best_control=best,eligible_controls=eligible,local_gain=gain,
                scene_gains=scene_gains,references=references,checks=checks,advance=all(checks.values()))
        grids[grid]=conditions
    return grids


def score():
    seal=read(PUBLIC/'predictions.json')
    assert seal['historical_scores_sha256']==sha(OLD/'scores.json')
    assert seal['historical_observations_sha256']==sha(OLD/'observations.json')
    observations={}
    with ProcessPoolExecutor(3) as pool:
        jobs={k:pool.submit(evaluate_points,k,{r['prediction'] for r in seal['rows'] if r['input_key']==k},RAW)
              for k in sorted({r['input_key'] for r in seal['rows']})}
        for key,job in jobs.items():_,observations[key]=job.result()
    dump(PUBLIC/'metrics.json',observations)
    identities={k:{h:{r:read(RAW/'evaluation'/k/h/r/'checked.json') for r in REFS} for h in v} for k,v in observations.items()}
    dump(PUBLIC/'observations.json',identities)
    summary=summarize(seal,observations,read(OLD/'scores.json'))
    dump(PUBLIC/'scores.json',dict(seal_sha256=sha(PUBLIC/'predictions.json'),grids=summary))
    print({g:{c:(v['local_gain'],v['checks']) for c,v in cells.items()} for g,cells in summary.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('predict','score'))
    args=p.parse_args();(predict if args.action=='predict' else score)()
