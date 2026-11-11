"""Repeat spatial permutations without changing camera geometry or the eligible edits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from statistics import mean
import numpy as np
import motion_reliability_surface as primary
import motion_placement_surface as placement
import motion_reliability_weights as model
from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import PUBLIC, RAW, ROOT, SCENES, FRAMES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_depth_scale_study import evaluate_points
from score_saved_geometry import read

SEEDS = (101,211,307,401)
DESIGN = ROOT/'docs/MOTION_SHUFFLE_REPEATS.md'
SEAL = PUBLIC/'shuffle_repeat_predictions.json'
RESULT = PUBLIC/'shuffle_repeat_scores.json'
DATA = RAW/'shuffle_repeat'


def infer():
    (DATA/'points').mkdir(parents=True,exist_ok=True)
    records = {}
    folds = read(model.RESULT)['results']['folds']
    heads = read(model.prior.ESTIMATES)['targets']
    refs = read(model.INDEX)['frames']
    scales = read(model.prior.PRIOR/'source_transplant.json')['batches']
    original = read(placement.SEAL)['hashes']
    for scene in SCENES:
        slug = scene.split('/')[0]
        key = slug+'_base'
        _,depth,k,e,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        maps = {seed:[] for seed in (0,*SEEDS)}
        outcomes = {seed:[] for seed in SEEDS}
        for i,frame in enumerate(FRAMES):
            name = model.prior.tag(scene,'short',frame)
            path = model.prior.MAPS/(name+'.npz')
            if sha(path)!=heads[name]['archive_sha256']:
                raise ValueError('Motion identity changed')
            with np.load(path) as f:
                u,masks = model.reliability_cells(depth[i],f['frozen_any_source'].astype(float),f['frozen_two_sided'])
            active = np.isfinite(u)
            correction = np.zeros_like(u)
            for c,m in zip(model.CELLS,masks):
                correction[m] = folds[scene]['weights'][c]*u[m]
            values = correction[active]
            ordered = np.sort(values)
            for seed in maps:
                label = f'{scene}/{frame}' if seed==0 else f'{scene}/{frame}/{seed}'
                generator = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big')
                delta = np.zeros_like(u)
                delta[active] = np.random.default_rng(generator).permutation(values)
                np.testing.assert_array_equal(np.sort(delta[active]),ordered)
                maps[seed].append(depth[i]*np.exp(delta))
        hashes = {}
        for seed,values in maps.items():
            points = point_map(np.stack(values),k,e,np.ones(5))
            h = tensor_digest(points)
            if seed==0:
                if h!=original[key]['permuted']:
                    raise ValueError('Original permutation failed exact map replay')
            else:
                write_arrays(DATA/'points'/(h+'.npz'),dict(points=points))
                hashes[str(seed)] = h
        for i,frame in enumerate(FRAMES):
            path = RAW/'reference'/slug/f'{frame:06d}.npz'
            if sha(path)!=refs[scene][str(frame)]['reference_sha256']:
                raise ValueError('Target identity differs')
            with np.load(path) as f:
                valid = f['valid']
                truth = f['depth'][valid]
            scale = scales[key]['scale_to_metres']
            original_sse = np.sum(np.log(scale*depth[i][valid]/truth)**2)
            for seed in SEEDS:
                outcomes[seed].append(float(1-np.sum(np.log(scale*maps[seed][i][valid]/truth)**2)/original_sse))
        records[key] = dict(hashes=hashes,axial_gain=outcomes,original_hash=original[key]['permuted'])
    return records


def batch(item):
    key,record = item
    return evaluate_points(key,set(record['hashes'].values()),DATA)


def summarize(obs,records):
    historic = primary.temporal.controls()
    old = read(placement.RESULT)['observed']
    base = read(primary.RESULT)['observed']
    results = {}
    for ref in primary.temporal.REFS:
        draws = {}
        for seed in ('weighted','original',*map(str,SEEDS)):
            rows = {}
            for scene in SCENES:
                key = scene.split('/')[0]+'_base'
                if seed=='weighted':
                    metrics = base[key]['weighted'][ref]
                elif seed=='original':
                    metrics = old[key]['permuted'][ref]
                else:
                    metrics = obs[key][seed][ref]
                if set(metrics)!=set(primary.temporal.METRICS) or not np.isfinite(list(metrics.values())).all():
                    raise ValueError('Invalid permutation metrics')
                score = mean(metrics[m]/historic[scene][ref]['frozen'][m] for m in primary.temporal.METRICS[:2])
                rows[scene] = dict(**metrics,score=score)
            draws[seed] = dict(scenes=rows,mean={m:mean(v[m] for v in rows.values()) for m in next(iter(rows.values()))})
        results[ref] = draws
    axial = {s:mean(v for r in records.values() for v in r['axial_gain'][s]) for s in map(str,SEEDS)}
    return dict(references=results,axial_gain=axial)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('infer','evaluate','verify'))
    action = p.parse_args().action
    deps = {v.name:sha(v) for v in (model.RESULT,placement.SEAL,model.prior.ESTIMATES,PLAN,model.INDEX,model.prior.PRIOR/'source_transplant.json')}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if action=='infer':
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(inputs=deps,records=infer()))
    else:
        seal = read(SEAL)
        if seal['inputs']!=deps: raise ValueError('Permutation source identity differs')
        records = seal['records']
        provenance = [sha(v) for v in (SEAL,placement.RESULT,primary.RESULT)]
        if action=='evaluate':
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                raw=dict(pool.map(batch,records.items()))
            obs = {k:{s:raw[k][h] for s,h in r['hashes'].items()} for k,r in records.items()}
            dump(RESULT,dict(inputs=provenance,observed=obs,results=summarize(obs,records)))
        report = read(RESULT)
        if report['inputs']!=provenance or report['results']!=summarize(report['observed'],records):
            raise ValueError('Permutation comparison changed')
        print(report['results']['axial_gain'])
