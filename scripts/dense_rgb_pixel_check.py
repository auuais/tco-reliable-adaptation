"""Evaluate dense RGB choices and reference-assisted headroom on identical rays."""
import argparse
import io
from statistics import mean
import zipfile

import numpy as np

from collect_matched_focal_heads import tensor_digest
from dense_rgb_depth import ARMS, FACTORS
from depth_spatial_support import support_masks
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import REF
from replay_focal_track_decisions import load_folds
from run_dense_rgb_depth import PARENT, PLAN, PUBLIC, inputs, verify_choices
from run_depth_scale_study import TRACKS
from score_dense_focal_pilot import REFS
from score_saved_geometry import read


REGIONS = ('near','far')
METHODS = ARMS+('oracle',)


def similarity(p,q):
    x,y = p.astype(float),q.astype(float)
    mx,my = x.mean(axis=0),y.mean(axis=0)
    x,y = x-mx,y-my
    u,s,vt = np.linalg.svd(x.T@y)
    correction = np.array([1.,1.,np.sign(np.linalg.det(u@vt))])
    rot = (u*correction)@vt
    scale = (s@correction)/np.sum(x*x)
    return scale,rot,my-scale*mx@rot


def stats(errors,indices,region):
    selected = np.take_along_axis(errors,indices[None],axis=0)[0]
    changed = (indices!=0)&region
    return dict(changed=int(changed.sum()),better=int(np.count_nonzero(changed & (selected<errors[0]))),
                worse=int(np.count_nonzero(changed & (selected>errors[0]))),sse=float(selected[region].sum()))


def measure(seal,directory=PUBLIC):
    plan = read(PLAN)
    parent = read(PARENT/'predictions.json')['rows']
    observations = {(o['camera']['scene'],o['camera']['batch']):o['tracks'] for o in read(TRACKS/'observations.json')}
    result = {}
    with zipfile.ZipFile(directory/'choices.zip') as selections, zipfile.ZipFile(TRACKS/'redetected_tracks.zip') as z:
        for key,b in seal.items():
            _,depth,k,poses,_ = inputs(key,parent,plan)
            camera = plan['inputs'][key]
            meta = observations[camera['scene'],camera['batch']]
            if meta['rgb_sha256']!=camera['rgb_sha256']:
                raise ValueError('RGB features came from different images')
            fit = support_masks(load_folds(z,meta),depth.shape)['fit']
            ref = plan['references'][key]
            if sha(REF/ref['file'])!=ref['sha256']:
                raise ValueError('Depth reference changed')
            with np.load(REF/ref['file'],allow_pickle=False) as f:
                refs = dict(f)
            with np.load(io.BytesIO(selections.read(key+'.npz')),allow_pickle=False) as f:
                choices = dict(f)
            maps = np.stack([point_map(depth*factor,k,poses,np.ones(5)) for factor in FACTORS])
            if tensor_digest(maps[0])!=b['direct_sha256']:
                raise ValueError('Frozen comparison changed')
            for arm,idx in choices.items():
                selected = np.take_along_axis(maps,idx[None,...,None],axis=0)[0]
                if tensor_digest(selected)!=b['point_hashes'][arm]:
                    raise ValueError('Choice grid does not reproduce the exact dense output')
            valid = refs['valid_mask']
            values = {}
            for label in REFS:
                target = refs[label][valid]
                scale,rot,shift = similarity(maps[0][valid],target)
                errors = np.stack([np.sum((scale*p[valid].astype(float)@rot+shift-target)**2,axis=-1) for p in maps])
                selections_by_arm = {arm:idx[valid] for arm,idx in choices.items()}
                selections_by_arm['oracle'] = errors.argmin(axis=0)
                by_region = {}
                masks = {'near':fit[valid],'far':~fit[valid]}
                for region,mask in masks.items():
                    by_region[region] = dict(n=int(mask.sum()),baseline_sse=float(errors[0,mask].sum()),
                                             arms={a:stats(errors,idx,mask) for a,idx in selections_by_arm.items()})
                values[label] = by_region
            result[key] = dict(reference_sha256=ref['sha256'],track_sha256=meta['matches_sha256'],results=values)
            print(key,'compared RGB depth changes inside and outside the sparse features',flush=True)
    return result


def aggregate(batches,rows,methods=METHODS):
    def group(subset,ref,regions):
        out = {}
        for arm in methods:
            fractions,reductions,initial,final = [],[],[],[]
            better = changed = 0
            for r in subset:
                cells = [batches[r['input_key']]['results'][ref][k] for k in regions]
                before = sum(c['baseline_sse'] for c in cells)
                after = sum(c['arms'][arm]['sse'] for c in cells)
                n = sum(c['n'] for c in cells)
                modified = sum(c['arms'][arm]['changed'] for c in cells)
                fractions.append(modified/n)
                reductions.append(1-after/before)
                initial.append(float(np.sqrt(before/n)))
                final.append(float(np.sqrt(after/n)))
                better += sum(c['arms'][arm]['better'] for c in cells)
                changed += modified
            out[arm] = dict(mean_changed_fraction=mean(fractions),mean_sse_reduction=mean(reductions),
                            pooled_improved_fraction=better/changed if changed else None,
                            baseline_mean_rmse_m=mean(initial),mean_rmse_m=mean(final))
        return out
    return {ref:{name:group(rows,ref,regions) for name,regions in (('all',REGIONS),('near',('near',)),('far',('far',)))}
            for ref in REFS}


def check(batches,seal,rows,methods=METHODS):
    plan = read(PLAN)
    origins = read(PARENT/'geometry_headroom.json')['results']['rows']
    for r,o in zip(rows,origins):
        key = r['input_key']
        if r['condition']!=o['condition'] or key not in seal:
            raise ValueError('Case correspondence differs')
        data = batches[key]
        ref = plan['references'][key]
        if data['reference_sha256']!=ref['sha256'] or data['track_sha256']!=r['matches_sha256']:
            raise ValueError('Different geometric or visual evidence')
        for label in REFS:
            cells = list(data['results'][label].values())
            if sum(c['n'] for c in cells)!=ref['valid_pixels']:
                raise ValueError('Different valid-depth mask')
            np.testing.assert_allclose(sum(c['baseline_sse'] for c in cells),o['arms']['direct'][label]['global']['sse_m2'],rtol=1e-9)
            for c in cells:
                for a in methods:
                    m = c['arms'][a]
                    if (not 0<=m['better']+m['worse']<=m['changed']<=c['n'] or not np.isfinite(m['sse'])
                            or not 0<=c['arms']['oracle']['sse']<=min(m['sse'],c['baseline_sse'])+1e-9):
                        raise ValueError('Invalid oracle bound or change counts')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','verify'))
    action = p.parse_args().action
    seal = verify_choices()['batches']
    rows = read(PARENT/'predictions.json')['rows']
    path = PUBLIC/'pixel_check.json'
    if action=='evaluate':
        if path.exists():
            raise FileExistsError(path)
        batches = measure(seal)
        check(batches,seal,rows)
        dump(path,dict(seal_sha256=sha(PUBLIC/'predictions.json'),batches=batches,results=aggregate(batches,rows)))
    else:
        saved = read(path)
        check(saved['batches'],seal,rows)
        if saved['seal_sha256']!=sha(PUBLIC/'predictions.json') or saved['results']!=aggregate(saved['batches'],rows):
            raise ValueError('Paired geometry result changed')
        print('Verified the paired-pixel, oracle and sparse-neighborhood comparisons.')
