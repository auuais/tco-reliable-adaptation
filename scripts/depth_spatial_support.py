"""Locate within-view shape error and measure proximity to existing RGB matches."""
import argparse
from statistics import mean
import zipfile

import numpy as np
from scipy.ndimage import distance_transform_edt

from collect_matched_focal_heads import tensor_digest
from depth_geometry_headroom import optima
from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import REF
from replay_focal_track_decisions import load_folds
from run_depth_scale_study import PLAN, PUBLIC, RAW, ROOT, TRACKS
from score_dense_focal_pilot import REFS
from score_saved_geometry import read


DESIGN = ROOT/'docs/DEPTH_SPATIAL_SUPPORT.md'
REGIONS = ('missing_neighbor','jump_neighbor','interior')
SUPPORT = ('fit','both')


def near(mask,radius):
    return distance_transform_edt(~mask)<=radius if mask.any() else np.zeros_like(mask)


def partition(points,valid):
    jump = np.zeros_like(valid)
    for axis in (0,1):
        left = (slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
        right = (slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
        gap = valid[left] & valid[right] & (np.linalg.norm(points[left]-points[right],axis=-1)>.05)
        jump[left] |= gap
        jump[right] |= gap
    missing = valid & near(~valid,2)
    edge = valid & ~missing & near(jump,2)
    return (missing,edge,valid & ~missing & ~edge)


def support_masks(folds,shape):
    views,height,width = shape
    seeds = np.zeros(shape,dtype=bool)
    out = {}
    for label,pairs in zip(SUPPORT,folds):
        for i,j,a,b in pairs:
            for view,xy in ((i,a),(j,b)):
                x,y = np.rint(xy).astype(int).T
                keep = (x>=0)&(x<width)&(y>=0)&(y<height)
                seeds[view,y[keep],x[keep]] = True
        out[label] = np.stack([near(seeds[v],8) for v in range(views)])
    return out


def residuals(points,target,valid,moment):
    u,s,vt = np.linalg.svd(moment['xy'])
    sign = np.sign(np.linalg.det(u@vt))
    correction = np.array([1.,1.,sign])
    scale = (s@correction)/np.trace(moment['xx'])
    rotation = (u*correction)@vt
    aligned = scale*(points[valid].astype(float)-moment['mx'])@rotation+moment['my']
    squared = np.zeros(valid.shape)
    squared[valid] = ((aligned-target[valid])**2).sum(axis=1)
    return squared


def collect(seal):
    refs = read(PLAN)['references']
    obs = {(r['camera']['scene'],r['camera']['batch']):r['tracks'] for r in read(TRACKS/'observations.json')}
    inputs = read(PLAN)['inputs']
    geometry = read(PUBLIC/'geometry_headroom.json')['observations']
    batches = {}
    with zipfile.ZipFile(TRACKS/'redetected_tracks.zip') as z:
        for r in seal:
            key = r['input_key']
            if key in batches:
                continue
            meta = obs[r['scene'],inputs[key]['batch']]
            if meta['matches_sha256']!=r['matches_sha256'] or meta['rgb_sha256']!=r['rgb_sha256']:
                raise ValueError('Different RGB tracks')
            folds = load_folds(z,meta)
            h = r['predictions']['direct']
            g = geometry[key+'_direct']
            ref = refs[key]
            if g['tensor_sha256']!=h or g['ref_sha256']!=sha(REF/ref['file']):
                raise ValueError('Reference-assisted fit does not match this prediction')
            with np.load(REF/ref['file'],allow_pickle=False) as f:
                target = dict(f)
            with np.load(RAW/'points'/(h+'.npz'),allow_pickle=False) as f:
                points = f['points']
            if tensor_digest(points)!=h:
                raise ValueError('Point map digest mismatch')
            valid = target['valid_mask']
            supports = support_masks(folds,valid.shape)
            observed = {}
            for label in REFS:
                cells = []
                for i,m in enumerate(g['moments'][label]):
                    sq = residuals(points[i],target[label][i],valid[i],m)
                    regions = partition(target[label][i],valid[i])
                    view = {}
                    for s in SUPPORT:
                        view[s] = {}
                        for name,mask in zip(REGIONS,regions):
                            near_match = supports[s][i]
                            view[s][name] = {}
                            for coverage,subset in [('near',mask&near_match),('far',mask&~near_match)]:
                                view[s][name][coverage] = dict(n=int(subset.sum()),sse_m2=float(sq[subset].sum()))
                    cells.append(view)
                observed[label] = cells
            batches[key] = dict(matches_sha256=r['matches_sha256'],direct_sha256=h,ref_sha256=ref['sha256'],cells=observed)
            print(key,'measured boundary and RGB coverage',flush=True)
    return batches


def summarize(batches,seal):
    def combine(rows,ref,support):
        cell_sums = {}
        for region in REGIONS:
            for coverage in ('near','far'):
                n_values,ss_values,counts,errors = [],[],[],[]
                for r in rows:
                    cells = batches[r['input_key']]['cells'][ref]
                    groups = [v[support][region][coverage] for v in cells]
                    all_groups = [g for v in cells for pair in v[support].values() for g in pair.values()]
                    n,ss = sum(g['n'] for g in groups),sum(g['sse_m2'] for g in groups)
                    counts.append(n)
                    errors.append(ss)
                    n_values.append(n/sum(g['n'] for g in all_groups))
                    ss_values.append(ss/sum(g['sse_m2'] for g in all_groups))
                cell_sums.setdefault(region,{})[coverage] = dict(pixel_fraction=mean(n_values),error_fraction=mean(ss_values),
                    pooled_rmse_m=float(np.sqrt(sum(errors)/sum(counts))) if sum(counts) else None)
        return cell_sums
    def all_groups(rows):
        return {ref:{s:combine(rows,ref,s) for s in SUPPORT} for ref in REFS}
    return dict(overall=all_groups(seal),by_scene={s:all_groups([r for r in seal if r['scene']==s]) for s in sorted({r['scene'] for r in seal})})


def verify(batches,seal):
    geometry = read(PUBLIC/'geometry_headroom.json')['observations']
    inputs = read(PLAN)
    if len(batches)!=9:
        raise ValueError('Incomplete image batches')
    for r in seal:
        key = r['input_key']
        item = batches[key]
        if (item['direct_sha256']!=r['predictions']['direct'] or item['matches_sha256']!=r['matches_sha256']
                or item['ref_sha256']!=inputs['references'][key]['sha256']):
            raise ValueError('Mismatched image or geometry')
        for label in REFS:
            moments = geometry[key+'_direct']['moments'][label]
            views = item['cells'][label]
            if len(views)!=len(moments):
                raise ValueError('Incomplete views')
            for v,m in zip(views,moments):
                oracle = optima([m])['scales']['sse_m2']
                for s in SUPPORT:
                    groups = [g for label in REGIONS for g in v[s][label].values()]
                    if any(g['n']<0 or g['sse_m2']<0 or not np.isfinite(g['sse_m2']) for g in groups):
                        raise ValueError('Invalid squared residual or count')
                    if sum(g['n'] for g in groups)!=m['n']:
                        raise ValueError('Area partition differs from reference mask')
                    np.testing.assert_allclose(sum(g['sse_m2'] for g in groups),oracle,rtol=1e-9)
                for region in REGIONS:
                    near0,near1 = (v[s][region]['near'] for s in SUPPORT)
                    if near0['n']>near1['n'] or near0['sse_m2']>near1['sse_m2']+1e-9:
                        raise ValueError('Union lost fitting-fold support')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('measure','verify'))
    action = p.parse_args().action
    target = PUBLIC/'spatial_support.json'
    seal = read(PUBLIC/'predictions.json')['rows']
    if action=='measure':
        if target.exists():
            raise FileExistsError(target)
        batches = collect(seal)
        verify(batches,seal)
        dump(target,dict(design_sha256=sha(DESIGN),batches=batches,results=summarize(batches,seal)))
    else:
        saved = read(target)
        verify(saved['batches'],seal)
        if saved['design_sha256']!=sha(DESIGN) or summarize(saved['batches'],seal)!=saved['results']:
            raise ValueError('Spatial summary or design differs')
        print('Verified region totals against all per-view similarity fits and replayed both depth references.')
