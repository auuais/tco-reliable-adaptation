"""Compare spatial depth corrections with matched global and support controls."""
import argparse
import hashlib
from statistics import mean
import numpy as np
import motion_reliability_weights as model
from evaluation_stability import dump, sha
from prepare_source_spacing import PUBLIC, ROOT, RAW, SCENES, FRAMES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read

DESIGN = ROOT/'docs/MOTION_SPATIAL_PLACEMENT.md'
RESULT = PUBLIC/'motion_spatial_placement.json'


def measure():
    results = []
    records = read(model.prior.ESTIMATES)['targets']
    folds = read(model.RESULT)['results']['folds']
    index = read(model.INDEX)['frames']
    scales = read(model.prior.PRIOR/'source_transplant.json')['batches']
    for scene in SCENES:
        slug = scene.split('/')[0]
        key = slug+'_base'
        scale = scales[key]['scale_to_metres']
        _,original,_,_,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        for i,frame in enumerate(FRAMES):
            name = model.prior.tag(scene,'short',frame)
            path = model.prior.MAPS/(name+'.npz')
            if sha(path)!=records[name]['archive_sha256']:
                raise ValueError('Motion input changed')
            with np.load(path) as f:
                u,groups = model.reliability_cells(original[i],f['frozen_any_source'].astype(float),f['frozen_two_sided'])
            active = np.isfinite(u)
            correction = np.zeros_like(u)
            for cell,mask in zip(model.CELLS,groups):
                correction[mask] = folds[scene]['weights'][cell]*u[mask]
            shuffle = np.zeros_like(correction)
            seed = int.from_bytes(hashlib.sha256(f'{scene}/{frame}'.encode()).digest()[:8],'big')
            shuffle[active] = np.random.default_rng(seed).permutation(correction[active])
            avg = correction[active].mean() if active.any() else 0.
            maps = dict(weighted=correction,global_mean=np.full_like(u,correction.mean()),
                        support_mean=np.where(active,avg,0.),permuted=shuffle)
            np.testing.assert_array_equal(np.sort(shuffle[active]),np.sort(correction[active]))
            for m in maps.values():
                np.testing.assert_allclose(m.mean(),correction.mean(),atol=1e-12)
            ref = RAW/'reference'/slug/f'{frame:06d}.npz'
            if sha(ref)!=index[scene][str(frame)]['reference_sha256']:
                raise ValueError('Reference changed')
            with np.load(ref) as f:
                valid = f['valid']
                error = np.log(scale*original[i][valid]/f['depth'][valid])
            base = float(error@error)
            residual = error-error.mean()
            scores = {a:float(np.sum((error+m[valid])**2)) for a,m in maps.items()}
            results.append(dict(scene=scene,frame=frame,valid=len(error),base=base,scores=scores,
                                global_oracle=float(residual@residual),error_sum=float(error.sum())))
    return results


def summarize(rows):
    targets = read(model.RESULT)['results']['targets']
    if len(rows)!=15 or len(targets)!=15:
        raise ValueError('Incomplete target set')
    for row,t in zip(rows,targets):
        if (row['scene'],row['frame'])!=(t['scene'],t['frame']):
            raise ValueError('Wrong target')
        np.testing.assert_allclose(row['scores']['weighted']/row['base'],t['risk']['weighted'],rtol=1e-9)
        np.testing.assert_allclose(row['base']-row['error_sum']**2/row['valid'],row['global_oracle'])
        if row['global_oracle']>row['scores']['global_mean']+1e-8:
            raise ValueError('Invalid constant-shift bound')
    def group(items):
        names = rows[0]['scores'].keys()
        return {a:mean(1-r['scores'][a]/r['base'] for r in items) for a in names}
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r['scene']==s]) for s in SCENES},
                mean_global_oracle_gain=mean(1-r['global_oracle']/r['base'] for r in rows))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('measure','verify'))
    deps = {p.name:sha(p) for p in (model.RESULT,model.INDEX,model.prior.ESTIMATES,PLAN,model.prior.PRIOR/'source_transplant.json')}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if parser.parse_args().action=='measure':
        if RESULT.exists(): raise FileExistsError(RESULT)
        rows = measure()
        dump(RESULT,dict(inputs=deps,targets=rows,results=summarize(rows)))
    report = read(RESULT)
    if report['inputs']!=deps or summarize(report['targets'])!=report['results']:
        raise ValueError('Placement comparison failed replay')
    print(report['results'])
