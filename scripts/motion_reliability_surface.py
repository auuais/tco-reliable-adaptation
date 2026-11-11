"""Evaluate cross-fitted source-agreement weights on original TCO scene surfaces."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import numpy as np
import motion_reliability_weights as model
import score_temporal_depth as temporal
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES, FRAMES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_depth_scale_study import aggregate, evaluate_points
from collect_matched_focal_heads import tensor_digest
from score_saved_geometry import read

DESIGN = ROOT/'docs/MOTION_RELIABILITY_SURFACE.md'
DATA = RAW/'weighted_surface'
SEAL = PUBLIC/'weighted_surface_predictions.json'
RESULT = PUBLIC/'weighted_surface_scores.json'
RULES = ('quarter','pooled','weighted')


def infer():
    study = read(model.RESULT)['results']
    archives = read(model.prior.ESTIMATES)['targets']
    expected = {(r['scene'],r['frame']):r['risk'] for r in study['targets']}
    out = {}
    (DATA/'points').mkdir(parents=True,exist_ok=True)
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        fold = study['folds'][scene]
        _,depth,k,e,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        fields = {rule:[] for rule in RULES}
        for i,frame in enumerate(FRAMES):
            label = model.prior.tag(scene,'short',frame)
            path = model.prior.MAPS/(label+'.npz')
            if sha(path)!=archives[label]['archive_sha256']:
                raise ValueError('Motion prediction hash differs')
            with np.load(path) as f:
                u,masks = model.reliability_cells(depth[i],f['frozen_any_source'].astype(float),f['frozen_two_sided'])
            for rule in RULES:
                weights = fold['weights'] if rule=='weighted' else dict.fromkeys(model.CELLS,.25 if rule=='quarter' else fold['pooled'])
                step = np.zeros_like(depth[i])
                for cell,mask in zip(model.CELLS,masks):
                    step[mask] = weights[cell]*u[mask]
                fields[rule].append(depth[i]*np.exp(step))
        hashes = {}
        for rule in RULES:
            z = np.stack(fields[rule])
            points = point_map(z,k,e,np.ones(5))
            h = tensor_digest(points)
            write_arrays(DATA/'points'/(h+'.npz'),dict(points=points))
            hashes[rule] = h
        # Verify depth scores only after all map choices for this scene are fixed.
        scale = read(model.prior.PRIOR/'source_transplant.json')['batches'][key]['scale_to_metres']
        for i,frame in enumerate(FRAMES):
            ref = RAW/'reference'/scene.split('/')[0]/f'{frame:06d}.npz'
            if sha(ref)!=read(model.INDEX)['frames'][scene][str(frame)]['reference_sha256']:
                raise ValueError('Reference changed')
            with np.load(ref) as f:
                valid = f['valid']
                truth = f['depth'][valid]
            base = np.sum(np.log(scale*depth[i][valid]/truth)**2)
            for rule in RULES:
                risk = np.sum(np.log(scale*fields[rule][i][valid]/truth)**2)/base
                np.testing.assert_allclose(risk,expected[scene,frame][rule],rtol=1e-9)
        out[key] = hashes
    dump(SEAL,dict(weights=sha(model.RESULT),protocol=hashlib.sha256(DESIGN.read_text().encode()).hexdigest(),hashes=out))


def batch(item):
    key,h = item
    return evaluate_points(key,set(h.values()),DATA)


def verify_inventory(seal,observed):
    keys = {s.split('/')[0]+'_base' for s in SCENES}
    if set(seal['hashes'])!=keys or set(observed)!=keys:
        raise ValueError('Missing surface comparison scene')
    for key in keys:
        if set(seal['hashes'][key])!=set(RULES) or set(observed[key])!=set(RULES):
            raise ValueError('Missing a registered prediction')
        for arm in RULES:
            if set(observed[key][arm])!=set(temporal.REFS):
                raise ValueError('Both depth references are required')
            for metrics in observed[key][arm].values():
                if set(metrics)!=set(temporal.METRICS) or not np.isfinite(list(metrics.values())).all():
                    raise ValueError('Unexpected surface metrics')
    archive = read(PUBLIC/'weighted_surface_archive.json')
    expected = {h+'.npz' for arms in seal['hashes'].values() for h in arms.values()}
    entries = archive['files']
    if (archive['verified_physical_drives']!=2 or len(entries)!=len(expected)
            or {e['file'] for e in entries}!=expected or archive['total_bytes']!=sum(e['bytes'] for e in entries)
            or any(e['bytes']<=0 or len(e['sha256'])!=64 for e in entries)):
        raise ValueError('Two-disk prediction inventory differs')


def summary(obs):
    historic = temporal.controls()
    local = temporal.OLD+tuple('context_'+a for a in temporal.ARMS)+('quarter','pooled')
    previous = read(temporal.SCORES)['observed']
    rows = []
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        metrics = {}
        for ref in temporal.REFS:
            scores = historic[scene][ref].copy()
            scores.update({'context_'+a:previous[key+'_short'][a][ref] for a in temporal.ARMS})
            scores.update({('guarded' if a=='weighted' else a):obs[key][a][ref] for a in RULES})
            metrics[ref] = scores
        rows.append(dict(scene=scene,case='base',input_key=key,metrics=metrics))
    return aggregate(rows,local+('guarded',),local)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('infer','evaluate','verify'))
    action = parser.parse_args().action
    if action=='infer':
        if SEAL.exists(): raise FileExistsError(SEAL)
        infer()
    else:
        seal = read(SEAL)
        if seal['weights']!=sha(model.RESULT) or seal['protocol']!=hashlib.sha256(DESIGN.read_text().encode()).hexdigest():
            raise ValueError('Surface inputs changed')
        deps = [sha(p) for p in (SEAL,temporal.SCORES,PARENT/'scores.json',temporal.PHOTO/'scores.json',temporal.VISIBILITY/'scores.json')]
        if action=='evaluate':
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                raw = dict(pool.map(batch,seal['hashes'].items()))
            obs = {k:{a:raw[k][h] for a,h in arms.items()} for k,arms in seal['hashes'].items()}
            dump(RESULT,dict(inputs=deps,observed=obs,results=summary(obs)))
        data = read(RESULT)
        if action=='verify':
            verify_inventory(seal,data['observed'])
        if data['inputs']!=deps or data['results']!=summary(data['observed']):
            raise ValueError('Surface comparison failed replay')
        print(data['results']['local_gain'],data['results']['historical_gain'],data['results']['advance'])
