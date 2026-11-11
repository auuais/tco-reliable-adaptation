"""Score sealed depth-evidence predictions under both references and the fixed gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

import dense_rgb_pixel_check as pixels
from evaluation_stability import dump,sha
from run_dense_rgb_depth import PARENT,PUBLIC as PHOTO
from run_visibility_model import PUBLIC,RAW,verify
from score_dense_focal_pilot import METRICS,REFS
from score_dense_rgb_depth import CONTROLS as PHOTO_CONTROLS,build_rows,check_archive
from score_depth_scale_study import aggregate,evaluate_points
from score_saved_geometry import read
from visibility_depth_evidence import MODEL_ARMS


CONTROLS = PHOTO_CONTROLS+('photo_guarded',)+tuple(k for k in MODEL_ARMS if k!='guarded')


def batch(entry):
    key,digests = entry
    return evaluate_points(key,set(digests.values()),RAW)


def surface(action,workers,seal):
    target = PUBLIC/'scores.json'
    provenance = {s:sha(p) for s,p in [('seal',PUBLIC/'predictions.json'),('older',PARENT/'scores.json'),('photo',PHOTO/'scores.json')]}
    keys = {k:b['point_hashes'] for k,b in seal.items()}
    if action=='evaluate':
        if target.exists():
            raise FileExistsError(target)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            obs = dict(pool.map(batch,keys.items()))
        measured = {k:{a:obs[k][h] for a,h in arms.items()} for k,arms in keys.items()}
    else:
        saved = read(target)
        if saved['provenance']!=provenance:
            raise ValueError('Surface score source changed')
        measured = saved['observed']
        check_archive(PUBLIC,seal)
    old = read(PARENT/'scores.json')['rows']
    photo = read(PHOTO/'scores.json')['observed']
    rows = build_rows(old,photo)
    if set(measured)!=set(keys):
        raise ValueError('Surface measurement missing')
    for r in rows:
        k = r['input_key']
        if set(measured[k])!=set(MODEL_ARMS):
            raise ValueError('Missing a model arm')
        for ref in REFS:
            d = r['metrics'][ref]
            d['photo_guarded'] = d.pop('guarded')
            for arm in MODEL_ARMS:
                m = measured[k][arm][ref]
                if (set(m)!=set(METRICS) or not np.isfinite(list(m.values())).all()
                        or min(m[x] for x in METRICS[:2])<=0 or not 0<=m[METRICS[2]]<=1):
                    raise ValueError('Invalid measured surface distances')
                d[arm] = m
                if keys[k][arm]==seal[k]['direct_sha256']:
                    np.testing.assert_allclose([m[x] for x in METRICS],[d['direct'][x] for x in METRICS],atol=1e-10,rtol=0)
    result = aggregate(rows,CONTROLS+('guarded',),CONTROLS)
    if action=='evaluate':
        dump(target,dict(provenance=provenance,observed=measured,results=result,cloud_usd=0))
    elif result!=saved['results']:
        raise ValueError('Registered criteria or outcome changed')


def pixel(action,seal):
    methods = tuple(MODEL_ARMS)+('oracle',)
    rows = read(PARENT/'predictions.json')['rows']
    path = PUBLIC/'pixel_check.json'
    if action=='pixels':
        if path.exists():
            raise FileExistsError(path)
        batches = pixels.measure(seal,PUBLIC)
    else:
        saved = read(path)
        batches = saved['batches']
    pixels.check(batches,seal,rows,methods)
    result = pixels.aggregate(batches,rows,methods)
    # The two studies use the identical depth grid and alignment at every valid pixel.
    previous = read(PHOTO/'pixel_check.json')['results']
    for ref in REFS:
        for region,value in result[ref].items():
            for k,x in value['oracle'].items():
                np.testing.assert_allclose(x,previous[ref][region]['oracle'][k],rtol=1e-10)
    digest = sha(PUBLIC/'predictions.json')
    if action=='pixels':
        dump(path,dict(seal_sha256=digest,batches=batches,results=result))
    elif saved['seal_sha256']!=digest or result!=saved['results']:
        raise ValueError('Pixel diagnostic differs')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','pixels','verify'))
    p.add_argument('--workers',type=int,choices=range(1,5),default=3)
    args = p.parse_args()
    seal = verify()['batches']
    if args.action in ('evaluate','verify'):
        surface(args.action,args.workers,seal)
    if args.action in ('pixels','verify'):
        pixel(args.action,seal)
    if args.action=='verify':
        print('Verified five frozen-model depth rules, historical controls and matching oracle under both references.')
