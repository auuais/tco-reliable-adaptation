"""Fit RGB correction strengths to fixed-association surface distances in other scenes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from statistics import mean
import numpy as np
from scipy.spatial import cKDTree

import motion_icp_effect as icp
import motion_reliability_surface as surface
import motion_reliability_weights as model
import score_temporal_depth as temporal
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from prepare_source_spacing import PUBLIC, ROOT, SCENES, FRAMES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read

DESIGN = ROOT/'docs/SURFACE_FITTED_RGB_WEIGHTS.md'
RESULT = PUBLIC/'surface_weight_fit.json'
GRID = (0.,.125,.25,.5,1.)


def cues(scene, depth):
    key = scene.split('/')[0]+'_base'
    records = read(model.prior.ESTIMATES)['targets']
    changes,labels = [],[]
    for i,frame in enumerate(FRAMES):
        name = model.prior.tag(scene,'short',frame)
        path = model.prior.MAPS/(name+'.npz')
        if sha(path)!=records[name]['archive_sha256']:
            raise ValueError('RGB motion estimate differs')
        with np.load(path) as f:
            u,masks = model.reliability_cells(depth[i],f['frozen_any_source'].astype(float),f['frozen_two_sided'])
        cell = np.full(u.shape,len(model.CELLS),np.uint8)
        for j,mask in enumerate(masks):
            cell[mask] = j
        changes.append(np.where(np.isfinite(u),np.expm1(u),0.))
        labels.append(cell)
    return np.stack(changes),np.stack(labels)


def tabulate(residual, shift, labels):
    n = len(residual)
    table = []
    for j in range(len(model.CELLS)+1):
        use = labels==j
        r,d = residual[use],shift[use]
        table.append([float(np.linalg.norm(r+a*d,axis=1).sum()/n) for a in GRID])
    return table


def measure(scene):
    from run_repeat_calibration import umeyama
    key = scene.split('/')[0]+'_base'
    _,depth,_,e,direct_hash = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
    points = temporal.load_points(temporal.OLD_POINTS,direct_hash)
    fraction,cell = cues(scene,depth)
    spec = read(PLAN)['references'][key]
    ref = temporal.REF/spec['file']
    if sha(ref)!=spec['sha256']: raise ValueError('Changed reference geometry')
    with np.load(ref) as f:
        mask = f['valid_mask']
        target = f['registered_rgb'][mask]
    scale,r,t = umeyama(points[mask].T,target.T)
    matrix = np.array(read(icp.RESULT)['observed'][scene]['registered_rgb']['direct']['transform'])
    rot = matrix[:3,:3]
    aligned = (scale*np.einsum('nhwj,ij->nhwi',points,r)+t.T)[mask]@rot.T+matrix[:3,3]
    ray = (points-centers(e)[:,None,None,:])[mask]
    displacement = scale*((fraction[mask,None]*ray)@r.T)@rot.T
    forward = cKDTree(target).query(aligned,workers=2)[1]
    backward = cKDTree(aligned).query(target,workers=2)[1]
    labels = cell[mask]
    tables = dict(accuracy_mean_m=tabulate(aligned-target[forward],displacement,labels),
                  completeness_mean_m=tabulate(aligned[backward]-target,displacement[backward],labels[backward]))
    print(scene,'computed surface-distance tables',flush=True)
    return scene,tables


def fit(tables):
    archive = temporal.controls()
    objectives = {}
    for scene in SCENES:
        data = archive[scene]['registered_rgb']
        parts = []
        for metric in temporal.METRICS[:2]:
            distances = np.asarray(tables[scene][metric])
            if distances.shape!=(5,5) or not np.isfinite(distances).all() or np.any(distances<0):
                raise ValueError('Invalid distance table')
            np.testing.assert_allclose(distances[:,0].sum(),data['direct'][metric],atol=.00002,rtol=0)
            np.testing.assert_allclose(distances[-1],distances[-1,0],rtol=1e-10)
            parts.append(distances/data['frozen'][metric])
        objectives[scene] = np.mean(parts,axis=0)
    folds = {}
    for scene,held in objectives.items():
        train = np.mean([v for s,v in objectives.items() if s!=scene],axis=0)
        cell_steps = train[:-1].argmin(axis=1)
        pooled = int(train.sum(axis=0).argmin())
        def score(table,steps):
            return float(table[-1,0]+sum(table[j,a] for j,a in enumerate(steps)))
        train_weighted = score(train,cell_steps)
        if train_weighted>train[:,0].sum()+1e-10: raise ValueError('Zero step should bound training risk')
        folds[scene] = dict(weights={c:GRID[a] for c,a in zip(model.CELLS,cell_steps)},pooled=GRID[pooled],
                            training=dict(weighted=train_weighted,direct=float(train[:,0].sum())),
                            held=dict(weighted=score(held,cell_steps),pooled=float(held[:,pooled].sum()),
                                      quarter=float(held[:,2].sum()),direct=float(held[:,0].sum())))
    return dict(folds=folds,overall={a:mean(f['held'][a] for f in folds.values()) for a in ('weighted','pooled','quarter','direct')})


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('measure','verify'))
    deps = {v.name:sha(v) for v in (icp.RESULT,model.RESULT,model.prior.ESTIMATES,PLAN,temporal.SEAL,surface.RESULT)}
    deps[DESIGN.name] = hashlib.sha256(DESIGN.read_text().encode()).hexdigest()
    if p.parse_args().action=='measure':
        if RESULT.exists(): raise FileExistsError(RESULT)
        with ProcessPoolExecutor(max_workers=3) as pool:
            tables = dict(pool.map(measure,SCENES))
        dump(RESULT,dict(inputs=deps,grid=GRID,tables=tables,results=fit(tables)))
    report = read(RESULT)
    if report['inputs']!=deps or report['grid']!=list(GRID) or report['results']!=fit(report['tables']):
        raise ValueError('Surface objective fit replay failed')
    print(report['results'])
