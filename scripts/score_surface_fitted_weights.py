"""Measure whole surfaces for scene-cross-fitted symmetric-distance weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import fit_rgb_surface_weights as fitting
import motion_placement_surface as placement
import motion_reliability_surface as old
import score_temporal_depth as temporal
from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import PUBLIC, RAW, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_depth_scale_study import aggregate, evaluate_points
from score_saved_geometry import read

DATA = RAW/'surface_fitted'
SEAL = PUBLIC/'surface_fitted_predictions.json'
RESULT = PUBLIC/'surface_fitted_scores.json'
RULES = ('quarter','pooled','weighted')


def infer():
    folds = read(fitting.RESULT)['results']['folds']
    records = {}
    (DATA/'points').mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        _,depth,k,e,direct = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        fraction,cell = fitting.cues(scene,depth)
        data = {}
        for arm in RULES:
            weights = folds[scene]['weights'] if arm=='weighted' else dict.fromkeys(old.model.CELLS,.25 if arm=='quarter' else folds[scene]['pooled'])
            lookup = np.array([weights[c] for c in old.model.CELLS]+[0.])
            delta = lookup[cell]*fraction
            points = point_map(depth*(1+delta),k,e,np.ones(5))
            h = tensor_digest(points)
            if not np.any(delta) and direct!=h: raise ValueError('Zero step changed original map')
            write_arrays(DATA/'points'/(h+'.npz'),dict(points=points))
            data[arm] = h
        records[key] = data
    return records


def batch(item):
    key,hashes = item
    return evaluate_points(key,set(hashes.values()),DATA)


def summarize(obs, candidate=None):
    historic = temporal.controls()
    context = read(temporal.SCORES)['observed']
    axial = read(old.RESULT)['observed']
    placements = read(placement.RESULT)['observed']
    rows = []
    controls = (temporal.OLD+tuple('context_'+a for a in temporal.ARMS)+tuple('axial_'+a for a in old.RULES)
                +placement.ARMS+('quarter','pooled'))
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        metrics = {}
        for ref in temporal.REFS:
            v = historic[scene][ref].copy()
            v.update({'context_'+a:context[key+'_short'][a][ref] for a in temporal.ARMS})
            v.update({'axial_'+a:axial[key][a][ref] for a in old.RULES})
            v.update({a:placements[key][a][ref] for a in placement.ARMS})
            v.update({('guarded' if a=='weighted' else a):obs[key][a][ref] for a in RULES})
            if candidate is not None:
                v["previous_surface_fit"] = v["guarded"]
                v["guarded"] = candidate[key][ref]
            metrics[ref] = v
        rows.append(dict(scene=scene,case='base',input_key=key,metrics=metrics))
    if candidate is not None:
        controls += ('previous_surface_fit',)
    return aggregate(rows,controls+('guarded',),controls)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('infer','evaluate','verify'))
    action = p.parse_args().action
    if action=='infer':
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(fit=sha(fitting.RESULT),hashes=infer()))
    else:
        seal = read(SEAL)
        if seal['fit']!=sha(fitting.RESULT): raise ValueError('Training fit changed')
        deps = [sha(v) for v in (SEAL,old.RESULT,placement.RESULT,temporal.SCORES,temporal.PARENT/'scores.json',temporal.PHOTO/'scores.json',temporal.VISIBILITY/'scores.json')]
        if action=='evaluate':
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                raw = dict(pool.map(batch,seal['hashes'].items()))
            obs = {k:{a:raw[k][h] for a,h in arms.items()} for k,arms in seal['hashes'].items()}
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report = read(RESULT)
        if report['inputs']!=deps or report['results']!=summarize(report['observed']):
            raise ValueError('Surface training verification failed')
        if action=='verify':
            old.verify_map_archive(PUBLIC/'surface_fitted_archive.json',seal['hashes'])
        print(report['results']['local_gain'],report['results']['historical_gain'],report['results']['advance'])
