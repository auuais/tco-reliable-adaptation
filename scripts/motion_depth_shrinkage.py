"""Reconstruct the exact squared-error curve for scaled RGB triangulation updates."""
import argparse
from statistics import mean

import numpy as np

from evaluation_stability import dump, sha
from nearby_motion_depth import ESTIMATES, MAPS, PRIOR, RESULT as MOTION_RESULT
from collect_spacing_heads import tag
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read


DESIGN = ROOT/'docs/MOTION_DEPTH_SHRINKAGE.md'
RESULT = PUBLIC/'motion_shrinkage.json'
ARMS = ('frozen_any_source','frozen_two_sided','measured_any_source','measured_two_sided')


def measure():
    records = read(ESTIMATES)['targets']
    frames = read(INDEX)['frames']
    metric_scales = read(PRIOR/'source_transplant.json')['batches']
    rows = []
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        scale = metric_scales[key]['scale_to_metres']
        _, original, _, _, _ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        for i, frame in enumerate(FRAMES):
            name = tag(scene,'short',frame)
            points = MAPS/(name+'.npz')
            reference = RAW/'reference'/scene.split('/')[0]/f'{frame:06d}.npz'
            if sha(points)!=records[name]['archive_sha256'] or sha(reference)!=frames[scene][str(frame)]['reference_sha256']:
                raise ValueError('Sealed target depth or reference changed')
            with np.load(reference,allow_pickle=False) as f:
                valid = f['valid']
                truth = f['depth'][valid]
            start = original[i][valid]*scale
            error = np.log(start/truth)
            cells = {}
            with np.load(points,allow_pickle=False) as maps:
                for arm in ARMS:
                    depth = maps[arm][valid].astype(float)
                    if arm.startswith('frozen_'):
                        depth *= scale
                    ok = np.isfinite(depth)
                    u = np.clip(np.log(depth[ok]/start[ok]),-.08,.08)
                    cells[arm] = dict(eligible=int(ok.sum()), C=float(error@error),
                                      A=float(u@u), B=float(error[ok]@u))
            rows.append(dict(scene=scene,frame=frame,valid=int(valid.sum()),cells=cells))
    return rows


def fit(cells):
    curvature = mean(v['A']/v['C'] for v in cells)
    direction = mean(v['B']/v['C'] for v in cells)
    signed = -direction/curvature if curvature>0 else 0.
    step = min(1.,max(0.,signed))
    return dict(curvature=curvature,slope=2*direction,signed_optimum=signed,
                step=step,best_gain=-(2*direction*step+curvature*step**2),
                unit_gain=-(2*direction+curvature))


def summarize(rows):
    result = {}
    for arm in ARMS:
        pooled = fit([r['cells'][arm] for r in rows])
        scenes = {}
        for scene in SCENES:
            train = [r['cells'][arm] for r in rows if r['scene']!=scene]
            test = [r['cells'][arm] for r in rows if r['scene']==scene]
            step = fit(train)['step']
            own = fit(test)
            held_gain = -(own['slope']*step+own['curvature']*step**2)
            scenes[scene] = dict(oracle=own,held_step=step,held_gain=held_gain)
        bound = mean(fit([r['cells'][arm]])['best_gain'] for r in rows)
        result[arm] = dict(pooled=pooled,scenes=scenes,
            leave_scene_out_gain=mean(v['held_gain'] for v in scenes.values()),target_oracle_gain=bound)
    return result


def provenance():
    return {p.name:sha(p) for p in (DESIGN,ESTIMATES,INDEX,MOTION_RESULT,PRIOR/'source_transplant.json')}


def verify(report):
    scores = read(MOTION_RESULT)
    rows = report['targets']
    expected = [(s,f) for s in SCENES for f in FRAMES]
    if report['provenance']!=provenance() or [(r['scene'],r['frame']) for r in rows]!=expected:
        raise ValueError('Changed shrinkage experiment inputs')
    for row, score in zip(rows,scores['targets']):
        if (row['scene'],row['frame'])!=(score['scene'],score['frame']) or set(row['cells'])!=set(ARMS):
            raise ValueError('Target mismatch')
        for arm, v in row['cells'].items():
            natural = score['cells'][arm]['natural']['all']
            if (row['valid']!=natural['n'] or v['eligible']!=natural['eligible'] or v['A']<0 or v['C']<=0
                    or v['B']**2>v['A']*v['C']+1e-9):
                raise ValueError('Invalid correction moments')
            np.testing.assert_allclose([v['C'],v['C']+2*v['B']+v['A']],
                                       [natural['initial_sse'],natural['final_sse']],rtol=1e-10,atol=1e-8)
    summary = summarize(rows)
    if summary!=report['results']:
        raise ValueError('Recomputed optimal steps differ')
    for arm, v in summary.items():
        unit = scores['results']['overall'][arm]['natural']['all']['sse_reduction']
        np.testing.assert_allclose(v['pooled']['unit_gain'],unit,rtol=1e-10,atol=1e-10)
        if v['target_oracle_gain']+1e-10<v['pooled']['best_gain'] or v['pooled']['best_gain']+1e-10<max(0,unit):
            raise ValueError('Scalar risk bound violated')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('measure','verify'))
    if parser.parse_args().action=='measure':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = measure()
        report = dict(provenance=provenance(),targets=rows,results=summarize(rows))
        verify(report)
        dump(RESULT,report)
    else:
        verify(read(RESULT))
    print('Verified the entire quadratic from the sealed motions and both original score endpoints.')
