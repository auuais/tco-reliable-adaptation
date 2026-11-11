"""Fit observable depth-update strengths using only other development scenes."""
import argparse
import hashlib
from statistics import mean
import numpy as np
import motion_depth_shrinkage as prior
from evaluation_stability import dump, sha
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read

RESULT = PUBLIC/'motion_reliability_weights.json'
DESIGN = ROOT/'docs/MOTION_RELIABILITY_WEIGHTS.md'
CELLS = ('short_single','long_single','short_agree','long_agree')


def reliability_cells(original, estimate, two_sided):
    update = np.clip(np.log(estimate/original),-.08,.08)
    usable = np.isfinite(update)
    agree = np.isfinite(two_sided)
    return update, [usable & (agree==paired) & ((update>=0)==positive)
                    for paired in (False,True) for positive in (False,True)]


def collect():
    records = read(prior.ESTIMATES)['targets']
    refs = read(INDEX)['frames']
    scales = read(prior.PRIOR/'source_transplant.json')['batches']
    rows = []
    for scene in SCENES:
        slug = scene.split('/')[0]
        key = slug+'_base'
        _,depth,_,_,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        scale = scales[key]['scale_to_metres']
        for i,frame in enumerate(FRAMES):
            path = prior.MAPS/(prior.tag(scene,'short',frame)+'.npz')
            ref = RAW/'reference'/slug/f'{frame:06d}.npz'
            if sha(path)!=records[path.stem]['archive_sha256'] or sha(ref)!=refs[scene][str(frame)]['reference_sha256']:
                raise ValueError('Input identity mismatch')
            with np.load(path) as f:
                u,masks = reliability_cells(depth[i],f['frozen_any_source'].astype(float),f['frozen_two_sided'])
            with np.load(ref) as f:
                valid = f['valid']
                e = np.log(scale*depth[i]/f['depth'])
            cells = {}
            for name,mask in zip(CELLS,masks):
                m = valid & mask
                cells[name] = dict(n=int(m.sum()),A=float(u[m]@u[m]),B=float(e[m]@u[m]))
            rows.append(dict(scene=scene,frame=frame,C=float(e[valid]@e[valid]),cells=cells))
    return rows


def strength(train,names,default=0.):
    energy = sum(r['cells'][c]['A']/r['C'] for r in train for c in names)
    cross = sum(r['cells'][c]['B']/r['C'] for r in train for c in names)
    return min(1.,max(0.,-cross/energy)) if energy else default


def risk(row,weights):
    delta = sum((2*v['B']*weights[c]+v['A']*weights[c]**2) for c,v in row['cells'].items())
    return 1+delta/row['C']


def summarize(rows):
    old = read(prior.RESULT)['targets']
    if len(rows)!=15 or len(old)!=len(rows):
        raise ValueError('Missing development targets')
    folds,targets = {},[]
    for r,o in zip(rows,old):
        v = o['cells']['frozen_any_source']
        if (r['scene'],r['frame'])!=(o['scene'],o['frame']) or sum(c['n'] for c in r['cells'].values())!=v['eligible']:
            raise ValueError('Support does not reproduce frozen predictions')
        np.testing.assert_allclose([r['C'],*[sum(c[k] for c in r['cells'].values()) for k in ('A','B')]],
                                   [v['C'],v['A'],v['B']],rtol=1e-9,atol=1e-9)
    for scene in SCENES:
        train = [r for r in rows if r['scene']!=scene]
        pooled = strength(train,CELLS)
        weights = {c:strength(train,[c],pooled) for c in CELLS}
        folds[scene] = dict(pooled=pooled,weights=weights)
        for r in rows:
            if r['scene']==scene:
                outcomes = {k:risk(r,dict.fromkeys(CELLS,w)) for k,w in [('unchanged',0.),('quarter',.25),('pooled',pooled)]}
                outcomes['weighted'] = risk(r,weights)
                outcomes['oracle'] = risk(r,{c:strength([r],[c]) for c in CELLS})
                if outcomes['oracle']>min(outcomes.values())+1e-10:
                    raise ValueError('Target oracle bound violated')
                targets.append(dict(scene=scene,frame=r['frame'],risk=outcomes))
    def average(group):
        return {k:mean(t['risk'][k] for t in group) for k in targets[0]['risk']}
    overall = average(targets)
    comparator = min(('unchanged','quarter','pooled'),key=overall.get)
    scenes = {s:average([t for t in targets if t['scene']==s]) for s in SCENES}
    gain = 1-overall['weighted']/overall[comparator]
    return dict(folds=folds,targets=targets,overall=overall,comparator=comparator,gain=gain,scenes=scenes,
                consider_surface=gain>=.02 and all(v['weighted']<v[comparator] for v in scenes.values()))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('measure','verify'))
    deps = {p.name:sha(p) for p in (prior.RESULT,prior.ESTIMATES,INDEX,PLAN,prior.PRIOR/'source_transplant.json')}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text(encoding='utf-8').encode()).hexdigest()
    if parser.parse_args().action=='measure':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = collect()
        dump(RESULT,dict(inputs=deps,targets=rows,results=summarize(rows)))
    report = read(RESULT)
    if report['inputs']!=deps or report['results']!=summarize(report['targets']):
        raise ValueError('Reliability calibration replay differs')
    print(report['results']['overall'],report['results']['gain'],report['results']['consider_surface'])
