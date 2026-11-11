"""Compare three sealed RGB depth rules with local baselines and archived TCO."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from dense_rgb_depth import ARMS
from evaluation_stability import dump, sha
from run_dense_rgb_depth import PARENT, PUBLIC, RAW, verify_choices
from score_dense_focal_pilot import METRICS, REFS
from score_depth_scale_study import HISTORICAL, aggregate, evaluate_points
from score_saved_geometry import read


CONTROLS = ('frozen','direct','shared','previous_guard','two_source','four_source')


def score_batch(entry):
    key,digests = entry
    return evaluate_points(key,set(digests.values()),RAW)


def build_rows(controls,measured):
    rows = []
    for r in controls:
        row = {k:r[k] for k in ('condition','input_key','scene','direction','case')}
        values = {}
        for ref in REFS:
            existing = r['metrics'][ref]
            values[ref] = {a:existing['guarded' if a=='previous_guard' else a] for a in CONTROLS[:4]+HISTORICAL}
            for arm in ARMS:
                metrics = measured[r['input_key']][arm][ref]
                if (set(metrics)!=set(METRICS) or not np.isfinite(list(metrics.values())).all()
                        or min(metrics[k] for k in METRICS[:2])<=0 or not 0<=metrics[METRICS[2]]<=1):
                    raise ValueError('Invalid observed depth metrics')
                values[ref][arm] = metrics
        rows.append(dict(**row,metrics=values))
    return rows


def main(action,workers):
    seal = verify_choices()['batches']
    target = PUBLIC/'scores.json'
    old = PARENT/'scores.json'
    provenance = dict(seal_sha256=sha(PUBLIC/'predictions.json'),controls_sha256=sha(old))
    if action=='evaluate':
        if target.exists():
            raise FileExistsError(target)
        hashes = {k:b['point_hashes'] for k,b in seal.items()}
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            values = dict(pool.map(score_batch,hashes.items()))
        measured = {key:{arm:values[key][h] for arm,h in arms.items()} for key,arms in hashes.items()}
    else:
        saved = read(target)
        if saved['provenance']!=provenance:
            raise ValueError('Scores refer to a different experiment')
        measured = saved['observed']
    rows = build_rows(read(old)['rows'],measured)
    report = aggregate(rows,CONTROLS+('guarded',),CONTROLS)
    if action=='evaluate':
        dump(target,dict(provenance=provenance,observed=measured,results=report,cloud_usd=0))
    elif report!=saved['results'] or set(measured)!=set(seal):
        raise ValueError('Dense score comparisons differ')
    else:
        print('Verified all discrete predictions, both reference comparisons and the development spending decision.')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','verify'))
    p.add_argument('--workers',type=int,choices=range(1,5),default=3)
    args = p.parse_args()
    main(args.action,args.workers)
