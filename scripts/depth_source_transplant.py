"""Cross camera and source depth quality on the same target pixels, using reference information offline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
from scipy.ndimage import uniform_filter

from dense_rgb_depth import FACTORS,LOG_STEPS,bilinear
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import centers
from photometric_positive_control import STARTS,reference_camera
from prepare_dense_focal_pilot import REF,ROOT
from run_dense_rgb_depth import PARENT,PLAN,inputs
from score_saved_geometry import read
from visibility_depth_evidence import screen
from visibility_positive_control import PUBLIC


DESIGN = ROOT/'docs/DEPTH_SOURCE_TRANSPLANT.md'
RIGS = ('reference','frozen')
STATES = tuple(STARTS)+('natural',)
CELLS = tuple(a+'_'+b for a in RIGS for b in RIGS)


def costs_for_sources(depth,beam,offset,k,sources):
    footprint = np.broadcast_to((np.isfinite(depth)&(depth>0))[...,None],sources.shape).copy()
    losses,identity_visible = [],None
    for i,factor in enumerate(FACTORS):
        xyz = beam*(depth*factor)[...,None]+offset
        z = xyz[...,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            x = k[0,0]*xyz[...,0]/z+k[0,2]
            y = k[1,1]*xyz[...,1]/z+k[1,2]
        sampled,inside = bilinear(sources,x,y)
        z = z[...,None]
        footprint &= inside[...,None]&np.isfinite(z)&(z>0)&np.isfinite(sampled)&(sampled>0)
        visible = z<=1.1*sampled
        if i==0:
            identity_visible = visible
        with np.errstate(divide='ignore',invalid='ignore'):
            cost = .2*np.minimum(np.abs(np.log(z/sampled))/np.log(1.1),1.)
        losses.append(np.where(visible,cost,.2))
    pixels = footprint&identity_visible
    density = uniform_filter(pixels.astype(float),size=(5,5,1),mode='constant')
    eligible = pixels&(density>=15/25-1e-9)
    patches = np.stack([uniform_filter(np.where(pixels,v,0.),size=(5,5,1),mode='constant')/np.maximum(density,1/25)
                        for v in losses])
    return [(patches[...,j],eligible[...,j]) for j in range(2)]


def select(i,target,k,e,sources):
    c = centers(e)
    neighbors = sorted((j for j in range(5) if j!=i),key=lambda j:(np.linalg.norm(c[j]-c[i]),j))
    y,x = np.indices(target.shape,dtype=float)
    rays = np.stack(((x-k[i,0,2])/k[i,0,0],(y-k[i,1,2])/k[i,1,1],np.ones_like(x)),axis=-1)@e[i,:3,:3]
    candidates = []
    for j in neighbors:
        rot = e[j,:3,:3].T
        candidates.append(costs_for_sources(target,rays@rot,c[i]@rot+e[j,:3,3],k[j],sources[j]))
    return [screen([g[j][0] for g in candidates],[g[j][1] for g in candidates])['two_witness'] for j in range(2)]


def summarize_pixels(idx,eligible,mask,initial):
    start = initial[mask]
    labels = idx[mask]
    error = start+LOG_STEPS[labels]
    update = labels!=0
    subset = eligible[mask]
    hist = np.bincount(labels[subset],minlength=len(LOG_STEPS)).tolist()
    return dict(n=len(start),eligible=int(subset.sum()),eligible_hist=hist,changed=int(update.sum()),
        improved=int(np.sum(update & (np.abs(error)<np.abs(start)-1e-10))),exact=int(np.sum(np.abs(error)<1e-8)),
        eligible_exact=int(np.sum(subset & (np.abs(error)<1e-8))),initial_sse=float(start@start),final_sse=float(error@error),
        oracle_sse=float(np.min((start[None]+LOG_STEPS[:,None])**2,axis=0).sum()))


def measure(key):
    plan = read(PLAN)
    parent = read(PARENT/'predictions.json')['rows']
    _,estimated,k_frozen,e_frozen,direct = inputs(key,parent,plan)
    ref = plan['references'][key]
    if sha(REF/ref['file'])!=ref['sha256']:
        raise ValueError('Depth reference changed')
    with np.load(REF/ref['file'],allow_pickle=False) as f:
        valid = f['valid_mask']
        cameras = [reference_camera(p,m) for p,m in zip(f['registered_rgb'],valid)]
    k_ref,e_ref,truth = [np.stack(v) for v in zip(*(r[:3] for r in cameras))]
    factor = float(np.median(truth[valid]/estimated[valid]))
    if not np.isfinite(factor) or factor<=0:
        raise ValueError('Invalid axial-depth scale')
    frozen = factor*estimated
    e_frozen = e_frozen.copy()
    e_frozen[:,:3,3] *= factor
    sources = np.stack((truth,np.where(valid,frozen,np.nan)),axis=-1).astype(np.float32)
    views = []
    for i,mask in enumerate(valid):
        states = {}
        for state in STATES:
            target = frozen[i] if state=='natural' else truth[i]*np.exp(STARTS[state])
            target = np.where(mask,target,np.nan)
            initial = np.log(target/truth[i])
            predictions = {}
            for rig,(k,e) in zip(RIGS,((k_ref,e_ref),(k_frozen,e_frozen))):
                pair = select(i,target,k,e,sources)
                predictions.update({rig+'_'+source:out for source,out in zip(RIGS,pair)})
            common = mask & np.logical_and.reduce([support for _,support in predictions.values()])
            states[state] = {cell:{subset:summarize_pixels(idx,ok,m,initial) for subset,m in (('all',mask),('common',common))}
                             for cell,(idx,ok) in predictions.items()}
        views.append(states)
    print(key,'crossed four camera/source settings and four target states',flush=True)
    return key,dict(scale_to_metres=factor,reference_sha256=ref['sha256'],direct_sha256=direct,views=views)


def summarize(batches,rows):
    def group(subset):
        out = {}
        for state in STATES:
            out[state] = {}
            for cell in CELLS:
                by_mask = {}
                for mask in ('all','common'):
                    averages = []
                    totals = np.zeros(4,dtype=np.int64)
                    for row in subset:
                        views = [v[state][cell][mask] for v in batches[row['input_key']]['views']]
                        def total(k):
                            return sum(v[k] for v in views)
                        n = total('n')
                        if n:
                            averages.append([total('eligible')/n,total('changed')/n,total('exact')/n,
                                             total('initial_sse')/n,total('final_sse')/n,total('oracle_sse')/n])
                        totals += [total(k) for k in ('eligible','eligible_exact','changed','improved')]
                    mean_vals = np.mean(averages,axis=0).tolist() if averages else [None]*6
                    result = dict(zip(('eligible_fraction','changed_fraction','exact_fraction','initial_mse','final_mse','oracle_mse'),mean_vals))
                    result.update(conditions_with_pixels=len(averages),
                        exact_among_eligible=float(totals[1]/totals[0]) if totals[0] else None,
                        improved_among_changed=float(totals[3]/totals[2]) if totals[2] else None)
                    by_mask[mask] = result
                out[state][cell] = by_mask
        return out
    return dict(overall=group(rows),by_scene={s:group([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})})


def verify(batches,rows):
    old = read(PUBLIC/'positive_control.json')['batches']
    if set(old)!=set(batches):
        raise ValueError('Missing transplant batch')
    for row in rows:
        key = row['input_key']
        b = batches[key]
        ref = old[key]
        if (b['direct_sha256']!=row['predictions']['direct'] or b['reference_sha256']!=ref['ref_sha256']
                or not np.isfinite(b['scale_to_metres']) or b['scale_to_metres']<=0 or len(b['views'])!=5):
            raise ValueError('Transplant input changed')
        for v,previous in zip(b['views'],ref['views']):
            for state in STATES:
                group = v[state]
                if state in STARTS:
                    a = group['reference_reference']['all']
                    earlier = previous['states'][state]['starting_depth_depth_two_witness']
                    if a['eligible_hist']!=earlier['hist'] or a['eligible']!=earlier['eligible']:
                        raise ValueError('Reference corner failed prior exact replay')
                common_count = group['reference_reference']['common']['n']
                for cell in CELLS:
                    for mask in ('all','common'):
                        m = group[cell][mask]
                        expected_n = common_count if mask=='common' else previous['camera_check']['valid_count']
                        if (m['n']!=expected_n or sum(m['eligible_hist'])!=m['eligible']
                                or not 0<=m['improved']<=m['changed']<=m['eligible']<=m['n']
                                or not 0<=m['eligible_exact']<=m['exact']<=m['n']
                                or m['oracle_sse']>min(m['initial_sse'],m['final_sse'])+1e-9):
                            raise ValueError('Invalid axial depth accounting')
                        if mask=='common' and m['eligible']!=m['n']:
                            raise ValueError('Noncommon pixel in intersection')


def scale_test():
    h,w = 24,50
    e = np.tile(np.eye(4),(5,1,1))
    e[:,0,3] = [-.18,-.09,0,.09,.18]
    k = np.tile([[35.,0,25],[0,35.,12],[0,0,1]],(5,1,1))
    source = np.full((5,h,w,2),2.,dtype=np.float32)
    target = np.full((h,w),2*np.exp(.08))
    initial = select(2,target,k,e,source)
    e[:,:3,3] *= 4
    rescaled = select(2,4*target,k,e,4*source)
    for (d,m),(a,b) in zip(initial,rescaled):
        np.testing.assert_array_equal(d,a)
        np.testing.assert_array_equal(m,b)
        if m.sum()<(h*w)/2 or not np.all(d[m]==3):
            raise ValueError('Synthetic metric scale test failed to recover true depth')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('test','evaluate','verify'))
    p.add_argument('--workers',type=int,choices=(1,2,3),default=2)
    args = p.parse_args()
    scale_test()
    if args.action!='test':
        target = PUBLIC/'source_transplant.json'
        rows = read(PARENT/'predictions.json')['rows']
        previous = sha(PUBLIC/'positive_control.json')
        if args.action=='evaluate':
            if target.exists():
                raise FileExistsError(target)
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                batches = dict(pool.map(measure,sorted({r['input_key'] for r in rows})))
        else:
            saved = read(target)
            batches = saved['batches']
        verify(batches,rows)
        result = summarize(batches,rows)
        if args.action=='evaluate':
            dump(target,dict(design_sha256=sha(DESIGN),control_sha256=previous,batches=batches,results=result))
        elif saved['design_sha256']!=sha(DESIGN) or saved['control_sha256']!=previous or saved['results']!=result:
            raise ValueError('Transplant summary changed')
        else:
            print('Verified all camera/source/target cells and matched-support intersections against the previous control.')
