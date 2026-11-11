"""Compare temporal and dispersed predictions on the same 15 target surfaces."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from statistics import mean
import zipfile

import numpy as np

from capture_visibility_support import ZIP as OLD_SUPPORT,unpack
from collect_matched_focal_heads import tensor_digest
from dense_rgb_pixel_check import similarity
from evaluation_stability import dump,sha
from predict_temporal_depth import ARMS,SEAL,verify
from prepare_dense_focal_pilot import REF
from prepare_source_spacing import PUBLIC,RAW,REGIMES,SCENES
from run_dense_rgb_depth import PARENT,PLAN,PUBLIC as PHOTO
from run_depth_scale_study import RAW as OLD_POINTS
from run_visibility_model import PUBLIC as VISIBILITY
from score_dense_focal_pilot import REFS,METRICS
from score_dense_rgb_depth import CONTROLS as PHOTO_CONTROLS,build_rows,check_archive
from score_depth_scale_study import HISTORICAL,aggregate,evaluate_points
from score_saved_geometry import read
from visibility_depth_evidence import MODEL_ARMS


OLD = PHOTO_CONTROLS+('rgb_guarded',)+tuple('visibility_'+a for a in MODEL_ARMS)
SIMPLE = OLD+tuple(a for a in ARMS if a!='checked')
SCORES = PUBLIC/'temporal_scores.json'
PIXELS = PUBLIC/'temporal_pixels.json'


def verify_asset_archive():
    index = read(PUBLIC/'inputs.json')
    capture = read(PUBLIC/'frozen_heads.json')
    expected = {}
    for sequence, frames in index['frames'].items():
        scene = sequence.split('/')[0]
        for number, frame in frames.items():
            filename = f"{scene}/{int(number):06d}.npz"
            expected['rgb/'+filename] = frame['rgb_sha256']
            expected['reference/'+filename] = frame['reference_sha256']
    for bundle, record in capture['records'].items():
        expected[f'heads/{bundle}.npz'] = record['archive_sha256']
        payload = (json.dumps(record, indent=2, allow_nan=False)+'\n').encode()
        expected[f'heads/{bundle}.json'] = hashlib.sha256(payload).hexdigest()
    archive = read(PUBLIC/'asset_archive.json')
    entries = archive['files']
    if (archive['verified_physical_drives'] != 2 or len(entries) != len(expected)
            or {entry['file']:entry['sha256'] for entry in entries} != expected
            or any(entry['bytes'] <= 0 for entry in entries)
            or sum(entry['bytes'] for entry in entries) != archive['total_bytes']):
        raise ValueError('Archived spacing inputs or model outputs differ from the capture records')


def controls():
    photo = read(PHOTO/'scores.json')['observed']
    vis = read(VISIBILITY/'scores.json')['observed']
    rows=build_rows([r for r in read(PARENT/'scores.json')['rows'] if r['case']=='metadata_fault'],photo)
    result={}
    for scene in SCENES:
        pair=[r for r in rows if r['scene']==scene]
        if len(pair)!=2 or {r['direction'] for r in pair}!={-1,1}:
            raise ValueError('Missing an original base-image metadata condition')
        key=scene.split('/')[0]+'_base'
        result[scene]={}
        for ref in REFS:
            d={a:{m:mean(r['metrics'][ref][a][m] for r in pair) for m in METRICS} for a in pair[0]['metrics'][ref]}
            d['rgb_guarded']=d.pop('guarded')
            d.update({'visibility_'+a:vis[key][a][ref] for a in MODEL_ARMS})
            result[scene][ref]=d
    return result


def summarize(measured):
    archived=controls()
    result={}
    for regime in REGIMES[1:]:
        rows=[]
        for scene in SCENES:
            key=scene.split('/')[0]+'_base'
            metrics={}
            for ref in REFS:
                vals=archived[scene][ref].copy()
                for a in ARMS:
                    v=measured[key+'_'+regime][a][ref]
                    if (set(v)!=set(METRICS) or not np.isfinite(list(v.values())).all() or min(v[m] for m in METRICS[:2])<=0
                            or not 0<=v[METRICS[2]]<=1):
                        raise ValueError('Invalid measured metric')
                    vals['guarded' if a=='checked' else a]=v
                metrics[ref]=vals
            rows.append(dict(scene=scene,case='base',input_key=key,metrics=metrics))
        result[regime]=aggregate(rows,SIMPLE+('guarded',),SIMPLE)
    return result


def batch(item):
    key,digests=item
    return evaluate_points(key,digests,RAW)


def surface(action,seal,workers):
    provenance=[sha(p) for p in (SEAL,PARENT/'scores.json',PHOTO/'scores.json',VISIBILITY/'scores.json')]
    if action=='evaluate':
        if SCORES.exists():
            raise FileExistsError(SCORES)
        by_key={}
        for b in seal.values():
            by_key.setdefault(b['input_key'],set()).update(b['point_hashes'].values())
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            raw=dict(pool.map(batch,by_key.items()))
        obs={label:{a:raw[b['input_key']][h] for a,h in b['point_hashes'].items()} for label,b in seal.items()}
    else:
        saved=read(SCORES)
        if saved['provenance']!=provenance:
            raise ValueError('Changed temporal comparison inputs')
        obs=saved['observed']
        check_archive(PUBLIC,seal)
    results=summarize(obs)
    if action=='evaluate':
        dump(SCORES,dict(provenance=provenance,observed=obs,results=results,cloud_usd=0))
    elif saved['results']!=results:
        raise ValueError('Temporal surface gate differs')


def load_points(directory,digest):
    with np.load(directory/'points'/(digest+'.npz'),allow_pickle=False) as z:
        p=z['points']
    if tensor_digest(p)!=digest:
        raise ValueError('Changed sealed point map')
    return p


def measure_pixels(seal):
    data={}
    ref_plan=read(PLAN)['references']
    with zipfile.ZipFile(OLD_SUPPORT) as support:
        for label,b in seal.items():
            key=b['input_key']
            spec=ref_plan[key]
            p=REF/spec['file']
            if sha(p)!=spec['sha256']:
                raise ValueError('Changed target depth')
            with np.load(p,allow_pickle=False) as z:
                refs=dict(z)
            valid=refs['valid_mask']
            available=unpack(support.read(key+'.npz'))
            baseline=load_points(OLD_POINTS,b['baseline_sha256'])
            values={}
            for ref in REFS:
                scale,rot,shift=similarity(baseline[valid],refs[ref][valid])
                start=scale*baseline.astype(float)@rot+shift-refs[ref]
                original=np.sum(start**2,axis=-1)
                arms={}
                for a,h in b['point_hashes'].items():
                    pred=load_points(RAW,h)
                    changed=np.any(pred!=baseline,axis=-1)
                    error=np.sum((scale*pred.astype(float)@rot+shift-refs[ref])**2,axis=-1)
                    views=[]
                    for i in range(5):
                        regions={}
                        for name,mask in [('all',valid[i]),('previously_available',valid[i]&available[i]),('previously_unavailable',valid[i]&~available[i])]:
                            moved=mask&changed[i]
                            regions[name]=dict(n=int(mask.sum()),changed=int(moved.sum()),improved=int(np.sum(moved&(error[i]<original[i]))),
                                baseline_sse=float(original[i,mask].sum()),final_sse=float(error[i,mask].sum()))
                        views.append(regions)
                    arms[a]=views
                values[ref]=arms
            data[label]=values
            print(label,'compared aligned 3D displacements at each target pixel',flush=True)
    return data


def summarize_pixels(obs):
    out={}
    for regime in REGIMES[1:]:
        out[regime]={}
        for ref in REFS:
            out[regime][ref]={}
            for a in ARMS:
                out[regime][ref][a]={}
                for region in ('all','previously_available','previously_unavailable'):
                    cells=[v[region] for s in SCENES for v in obs[s.split('/')[0]+'_base_'+regime][ref][a]]
                    if len(cells)!=15 or any(not 0<=c['improved']<=c['changed']<=c['n'] or min(c['baseline_sse'],c['final_sse'])<0 for c in cells):
                        raise ValueError('Invalid temporal pointwise observations')
                    changed=sum(c['changed'] for c in cells)
                    out[regime][ref][a][region]=dict(changed=mean(c['changed']/c['n'] for c in cells),
                        improved_among_changed=sum(c['improved'] for c in cells)/changed if changed else None,
                        sse_reduction=mean(1-c['final_sse']/c['baseline_sse'] for c in cells))
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','pixels','verify'))
    p.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    args=p.parse_args()
    seal=verify()['batches']
    if args.action in ('evaluate','verify'):
        surface(args.action,seal,args.workers)
    if args.action=='pixels':
        if PIXELS.exists():
            raise FileExistsError(PIXELS)
        measured=measure_pixels(seal)
        dump(PIXELS,dict(seal_sha256=sha(SEAL),observed=measured,results=summarize_pixels(measured)))
    if args.action=='verify':
        verify_asset_archive()
        data=read(PIXELS)
        if data['seal_sha256']!=sha(SEAL) or data['results']!=summarize_pixels(data['observed']):
            raise ValueError('Temporal pointwise summary differs')
        print('Verified matched-input controls, 15 target surfaces and both reference scores.')
