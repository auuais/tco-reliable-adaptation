"""Measure paired 3D error using shared and independently fitted scene similarities."""
import argparse
import hashlib
from statistics import mean
import numpy as np
import motion_reliability_surface as surface
import score_temporal_depth as temporal
from dense_rgb_pixel_check import similarity
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, ROOT, SCENES
from score_saved_geometry import read

DESIGN = ROOT/'docs/MOTION_PAIRED_GEOMETRY.md'
RESULT = PUBLIC/'weighted_paired_geometry.json'


def measure():
    hashes = read(surface.SEAL)['hashes']
    direct = read(temporal.SEAL)['batches']
    plan = read(temporal.PLAN)['references']
    records = {}
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        path = temporal.REF/plan[key]['file']
        if sha(path)!=plan[key]['sha256']:
            raise ValueError('Changed target reference')
        with np.load(path) as z:
            refs = dict(z)
        base = temporal.load_points(temporal.OLD_POINTS,direct[key+'_short']['baseline_sha256'])
        valid = refs['valid_mask']
        cameras = np.broadcast_to(np.arange(5)[:,None,None],valid.shape)[valid]
        counts = np.bincount(cameras,minlength=5)
        results = {}
        for ref in temporal.REFS:
            truth = refs[ref][valid].astype(float)
            shared = similarity(base[valid],truth)
            by_arm = {}
            for arm in ('direct',*surface.RULES):
                points = base if arm=='direct' else temporal.load_points(surface.DATA,hashes[key][arm])
                points = points[valid].astype(float)
                own = shared if arm=='direct' else similarity(points,truth)
                by_arm[arm] = {}
                for mode,(scale,rotation,translation) in [('shared',shared),('refit',own)]:
                    difference = scale*points@rotation+translation-truth
                    squared = np.einsum('ij,ij->i',difference,difference)
                    by_arm[arm][mode] = dict(sse=np.bincount(cameras,weights=squared,minlength=5).tolist(),
                                             scale=float(scale),rotation=rotation.tolist(),translation=translation.tolist())
            results[ref] = by_arm
        records[scene] = dict(counts=counts.tolist(),references=results)
    return records


def summarize(rows):
    result = {}
    for ref in temporal.REFS:
        scenes = {}
        for scene in SCENES:
            counts = rows[scene]['counts']
            arms = rows[scene]['references'][ref]
            if len(counts)!=5 or min(counts)<=0: raise ValueError('Incorrect target counts')
            by_arm = {}
            for a,b in arms.items():
                vals = {}
                for m,stats in b.items():
                    sse = np.array(stats['sse'])
                    baseline = np.array(arms['direct'][m]['sse'])
                    if sse.shape!=(5,) or np.any(sse<0) or not np.isfinite(sse).all():
                        raise ValueError('Invalid squared 3D error')
                    vals[m] = dict(mse=float(sse.sum()/sum(counts)),gain=float(1-sse.sum()/baseline.sum()),
                                   target_gains=(1-sse/baseline).tolist())
                if vals['refit']['mse']>vals['shared']['mse']+1e-10:
                    raise ValueError('Refitting similarity increased error')
                by_arm[a] = vals
            scenes[scene] = by_arm
        result[ref] = dict(scenes=scenes,overall={a:{m:mean(g[a][m]['gain'] for g in scenes.values())
                              for m in ('shared','refit')} for a in ('direct',*surface.RULES)})
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('measure','verify'))
    deps = {p.name:sha(p) for p in (surface.SEAL,temporal.SEAL,temporal.PLAN,surface.RESULT)}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if parser.parse_args().action=='measure':
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows = measure()
        dump(RESULT,dict(inputs=deps,rows=rows,results=summarize(rows)))
    data = read(RESULT)
    if data['inputs']!=deps or data['results']!=summarize(data['rows']):
        raise ValueError('3D geometry replay failed')
    print({r:g['overall'] for r,g in data['results'].items()})
