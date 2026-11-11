"""Measure pixel-corresponding 3D errors in the fixed-camera development study."""
import argparse
from statistics import mean
import sys

import numpy as np

from collect_matched_focal_heads import PUBLIC as HEAD_RECORDS, RAW as HEADS, tensor_digest
from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import REF
from run_depth_scale_study import ARMS, PLAN, PUBLIC, RAW, ROOT
from score_dense_focal_pilot import REFS
from score_saved_geometry import read


METHODS = ('native','fp32')+ARMS


def metrics(points, target, mask):
    from evaluation.mv_recon.utils import umeyama
    p, q = points[mask].astype(float), target[mask].astype(float)
    scale, rot, shift = umeyama(p.T,q.T)
    distances = np.linalg.norm(scale*p @ rot.T + shift.ravel() - q,axis=1)
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite corresponding-pixel distance')
    return dict(mean_m=float(distances.mean()),rmse_m=float(np.sqrt(np.mean(distances**2))),
                median_m=float(np.median(distances)),count=int(mask.sum()))


def aggregate(rows):
    def summarize(cells,ref):
        result = {}
        for arm in METHODS:
            observations = [c['metrics'][ref] for c in cells]
            result[arm] = dict(mean_m=mean(d[arm]['mean_m'] for d in observations),
                rmse_m=mean(d[arm]['rmse_m'] for d in observations),
                relative_mean=mean(d[arm]['mean_m']/d['fp32']['mean_m'] for d in observations))
        return result
    return {ref:dict(overall=summarize(rows,ref),
                     by_scene={s:summarize([r for r in rows if r['scene']==s],ref) for s in sorted({r['scene'] for r in rows})})
            for ref in REFS}


def measure(seal):
    sys.path.insert(0,str(ROOT/'third_party/TCO'))
    index = read(PLAN)
    captures = {r['input_key']:r for r in read(HEAD_RECORDS/'capture.json')['rows']}
    result = {}
    for key in sorted({r['input_key'] for r in seal}):
        ref = index['references'][key]
        heads = HEADS/key/'heads_and_points.npz'
        if sha(REF/ref['file'])!=ref['sha256'] or sha(heads)!=captures[key]['archive_sha256']:
            raise ValueError('Reference or frozen head capture changed')
        with np.load(REF/ref['file'],allow_pickle=False) as z:
            targets = dict(z)
        cache = {}
        def score_map(points,h):
            if tensor_digest(points)!=h:
                raise ValueError('Sealed point map changed')
            cache[h] = {label:metrics(points,targets[label],targets['valid_mask']) for label in REFS}
        with np.load(heads,allow_pickle=False) as z:
            for arm in ('native','fp32'):
                score_map(z[arm],captures[key][arm+'_sha256'])
        for r in (r for r in seal if r['input_key']==key):
            hashes = dict(native=captures[key]['native_sha256'],fp32=captures[key]['fp32_sha256'],**r['predictions'])
            for h in hashes.values():
                if h not in cache:
                    with np.load(RAW/'points'/(h+'.npz'),allow_pickle=False) as z:
                        score_map(z['points'],h)
            result[r['condition']] = dict(condition=r['condition'],scene=r['scene'],input_key=key,hashes=hashes,
                reference_sha256=ref['sha256'],metrics={label:{a:cache[h][label] for a,h in hashes.items()} for label in REFS})
        print(key,'pixel-corresponding distances evaluated',flush=True)
    return [result[r['condition']] for r in seal]


def main(action):
    path = PUBLIC/'pixel_diagnostic.json'
    seal = PUBLIC/'predictions.json'
    records = read(seal)['rows']
    if action=='measure':
        if path.exists():
            raise FileExistsError(path)
        rows = measure(records)
        dump(path,dict(predictions_sha256=sha(seal),rows=rows,summary=aggregate(rows)))
    else:
        data = read(path)
        refs = read(PLAN)['references']
        cap = {r['input_key']:r for r in read(HEAD_RECORDS/'capture.json')['rows']}
        if data['predictions_sha256']!=sha(seal) or len(data['rows'])!=len(records):
            raise ValueError('Pixel evaluation inputs differ')
        for p,r in zip(records,data['rows']):
            ref = refs[r['input_key']]
            c = cap[r['input_key']]
            if (r['condition']!=p['condition'] or r['input_key']!=p['input_key'] or r['scene']!=p['scene']
                    or r['reference_sha256']!=ref['sha256'] or
                    r['hashes']!=dict(native=c['native_sha256'],fp32=c['fp32_sha256'],**p['predictions'])):
                raise ValueError('Pixel evaluation identity differs')
            for label in REFS:
                for arm in METHODS:
                    v = r['metrics'][label][arm]
                    if (v['count']!=ref['valid_pixels'] or not 0<v['mean_m']<=v['rmse_m']
                            or not 0<=v['median_m'] or not np.isfinite(list(v.values())).all()):
                        raise ValueError('Invalid pixel error')
        if data['summary']!=aggregate(data['rows']):
            raise ValueError('Pixel-error summary changed')
        print('Verified paired-pixel observations and their two-reference aggregates.')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('measure','verify'))
    main(p.parse_args().action)
