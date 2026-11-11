"""Score global, support-wide and shuffled controls for the weighted reconstruction."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib

import motion_reliability_surface as primary
import motion_spatial_placement as spatial
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, ROOT, SCENES
from score_depth_scale_study import evaluate_points
from score_saved_geometry import read

SEAL = PUBLIC/'placement_surface_predictions.json'
RESULT = PUBLIC/'placement_surface_scores.json'
DESIGN = ROOT/'docs/MOTION_PLACEMENT_SURFACE.md'
ARMS = ('global_mean','support_mean','permuted')


def batch(item):
    key, hashes = item
    return evaluate_points(key,set(hashes.values()),primary.PLACEMENT_DATA)


def summarize(obs):
    weighted = read(primary.RESULT)['results']['references']
    base = primary.temporal.controls()
    result = {}
    for ref in primary.temporal.REFS:
        scenes = {}
        for scene in SCENES:
            key = scene.split('/')[0]+'_base'
            unit = base[scene][ref]['frozen']
            scores = {arm:sum(obs[key][arm][ref][m]/unit[m] for m in primary.temporal.METRICS[:2])/2 for arm in ARMS}
            scores['weighted'] = weighted[ref]['scenes'][scene]['guarded']['score']
            scenes[scene] = scores
        result[ref] = dict(scenes=scenes,overall={arm:sum(v[arm] for v in scenes.values())/len(scenes) for arm in (*ARMS,'weighted')})
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('infer','evaluate','verify'))
    action = p.parse_args().action
    protocol = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if action=='infer':
        if SEAL.exists(): raise FileExistsError(SEAL)
        dump(SEAL,dict(protocol=protocol,placement=sha(spatial.RESULT),weights=sha(primary.model.RESULT),
                       hashes=primary.infer(placement=True)))
    else:
        seal = read(SEAL)
        if seal['protocol']!=protocol or seal['placement']!=sha(spatial.RESULT) or seal['weights']!=sha(primary.model.RESULT):
            raise ValueError('Placement dependencies changed')
        deps = [sha(p) for p in (SEAL,primary.RESULT,primary.temporal.SCORES)]
        if action=='evaluate':
            if RESULT.exists(): raise FileExistsError(RESULT)
            with ProcessPoolExecutor(max_workers=3) as pool:
                raw = dict(pool.map(batch,seal['hashes'].items()))
            obs = {k:{a:raw[k][h] for a,h in v.items()} for k,v in seal['hashes'].items()}
            dump(RESULT,dict(inputs=deps,observed=obs,results=summarize(obs)))
        report = read(RESULT)
        if report['inputs']!=deps or report['results']!=summarize(report['observed']):
            raise ValueError('Spatial surface comparison failed replay')
        if action=='verify':
            archive = read(PUBLIC/'placement_surface_archive.json')
            hashes = {h+'.npz' for arms in seal['hashes'].values() for h in arms.values()}
            entries = archive['files']
            if (archive['verified_physical_drives']!=2 or len(entries)!=len(hashes)
                    or {e['file'] for e in entries}!=hashes or archive['total_bytes']!=sum(e['bytes'] for e in entries)
                    or any(e['bytes']<=0 or len(e['sha256'])!=64 for e in entries)):
                raise ValueError('Placement map archive differs')
        print({r:g['overall'] for r,g in report['results'].items()})
