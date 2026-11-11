"""Inject known target-depth errors and measure RGB recovery with reference cameras."""
import argparse
from statistics import mean

import numpy as np
from scipy.ndimage import uniform_filter

from dense_rgb_depth import ARMS, LOG_STEPS, choose, source_cost
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from pointmap_camera_diagnostic import fit_camera
from prepare_dense_focal_pilot import INPUT, REF, ROOT
from run_dense_rgb_depth import PARENT, PLAN, PUBLIC
from score_saved_geometry import read


DESIGN = ROOT/'docs/PHOTOMETRIC_POSITIVE_CONTROL.md'
STARTS = {'short':-.08,'clean':0.,'long':.08}
METHODS = ('nearest',)+ARMS


def reference_camera(world,valid):
    y,x = np.nonzero(valid)
    sample = np.linspace(0,len(x)-1,min(512,len(x)),dtype=int)
    uv = np.stack((x,y),axis=-1)
    rec = fit_camera(world[y[sample],x[sample]],uv[sample])
    e,k = rec['extrinsics'],rec['intrinsics']
    xyz = world[valid].astype(float)@e[:3,:3].T+e[:3,3]
    pix = xyz@k.T
    error = np.linalg.norm(pix[:,:2]/pix[:,2:]-uv,axis=1)
    if not np.isfinite(error).all() or error.max()>.02 or np.min(xyz[:,2])<=0:
        raise ValueError('Reference camera failed full-image reprojection')
    depth = np.full(valid.shape,np.nan)
    depth[valid] = xyz[:,2]
    return k,e,depth,dict(fit_count=len(sample),valid_count=len(error),max_pixel_error=float(error.max()))


def measure(key,plan):
    rgb,ref = plan['inputs'][key],plan['references'][key]
    if sha(INPUT/rgb['file'])!=rgb['sha256'] or sha(REF/ref['file'])!=ref['sha256']:
        raise ValueError('Released controlled inputs changed')
    with np.load(INPUT/rgb['file'],allow_pickle=False) as z:
        images = z['images'].transpose(0,2,3,1)
    with np.load(REF/ref['file'],allow_pickle=False) as z:
        valid = z['valid_mask']
        cameras = [reference_camera(p,m) for p,m in zip(z['registered_rgb'],valid)]
    k,e,depth = [np.stack(items) for items in zip(*(c[:3] for c in cameras))]
    c = centers(e)
    desc = images-uniform_filter(images,size=(1,9,9,1),mode='reflect')
    sources = np.concatenate((desc,depth[...,None].astype(np.float32)),axis=-1)
    y,x = np.indices(depth.shape[1:],dtype=float)
    views = []
    for i in range(5):
        neighbors = sorted((j for j in range(5) if j!=i),key=lambda j:(np.linalg.norm(c[j]-c[i]),j))
        rays = np.stack(((x-k[i,0,2])/k[i,0,0],(y-k[i,1,2])/k[i,1,1],np.ones_like(x)),axis=-1)@e[i,:3,:3]
        states = {}
        for state,offset in STARTS.items():
            a,v = [],[]
            for j in neighbors:
                rot = e[j,:3,:3].T
                cost,ok = source_cost(desc[i],sources[j],depth[i]*np.exp(offset),rays@rot,c[i]@rot+e[j,:3,3],k[j])
                a.append(cost)
                v.append(ok)
            decisions,_ = choose(a,v)
            decisions['nearest'] = np.where(v[0],a[0].argmin(axis=0),0)
            fit = v[0]&v[2]
            all_four = fit&v[1]&v[3]
            support = dict(nearest=v[0],two_source=fit,four_source=all_four,guarded=all_four)
            obs = {}
            for arm in METHODS:
                eligible = support[arm]&valid[i]
                d = decisions[arm][eligible]
                obs[arm] = dict(eligible=int(eligible.sum()),hist=np.bincount(d,minlength=len(LOG_STEPS)).tolist())
            states[state] = obs
        views.append(dict(view=i,neighbors=neighbors,camera_check=cameras[i][3],states=states))
    print(key,'tested clean depth and both injected errors',flush=True)
    return dict(rgb_sha256=rgb['sha256'],ref_sha256=ref['sha256'],views=views)


def summarize(batches,rows):
    def group(rows):
        result = {}
        for state,start in STARTS.items():
            result[state] = {}
            for arm in METHODS:
                all_fractions,eligibles = [],[]
                histogram = np.zeros(len(LOG_STEPS),dtype=int)
                for r in rows:
                    views = batches[r['input_key']]['views']
                    n = sum(v['camera_check']['valid_count'] for v in views)
                    h = np.sum([v['states'][state][arm]['hist'] for v in views],axis=0)
                    eligible = h.sum()
                    exact = h[np.abs(start+LOG_STEPS)<1e-9].sum()+(n-eligible if start==0 else 0)
                    all_fractions.append(float(exact/n))
                    eligibles.append(float(eligible/n))
                    histogram += h
                changed = histogram[1:].sum()
                errors = np.abs(start+LOG_STEPS)
                result[state][arm] = dict(mean_eligible_fraction=mean(eligibles),mean_exact_fraction=mean(all_fractions),
                    pooled_changed_among_eligible=float(changed/histogram.sum()) if histogram.sum() else None,
                    pooled_better_among_changed=float(histogram[errors<abs(start)-1e-9].sum()/changed) if changed else None,
                    pooled_exact_among_eligible=float(histogram[errors<1e-9].sum()/histogram.sum()) if histogram.sum() else None)
        return result
    return dict(overall=group(rows),by_scene={s:group([r for r in rows if r['scene']==s]) for s in sorted({r['scene'] for r in rows})})


def verify(batches,plan,rows):
    if set(batches)!={r['input_key'] for r in rows}:
        raise ValueError('Missing a positive-control batch')
    for key,b in batches.items():
        if b['rgb_sha256']!=plan['inputs'][key]['sha256'] or b['ref_sha256']!=plan['references'][key]['sha256']:
            raise ValueError('Reference identity changed')
        views = b['views']
        if len(views)!=5 or sum(v['camera_check']['valid_count'] for v in views)!=plan['references'][key]['valid_pixels']:
            raise ValueError('Invalid camera evaluations')
        for i,v in enumerate(views):
            c = v['camera_check']
            if (v['view']!=i or sorted(v['neighbors']+[i])!=list(range(5)) or not 0<=c['max_pixel_error']<=.02
                    or c['fit_count']!=min(512,c['valid_count'])):
                raise ValueError('Invalid positive-control camera check')
            for methods in v['states'].values():
                for arm in METHODS:
                    m = methods[arm]
                    if (len(m['hist'])!=5 or min(m['hist'])<0 or sum(m['hist'])!=m['eligible']
                            or m['eligible']>c['valid_count']):
                        raise ValueError('Invalid positive-control pixel counts')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('evaluate','verify'))
    action = p.parse_args().action
    plan = read(PLAN)
    rows = read(PARENT/'predictions.json')['rows']
    path = PUBLIC/'positive_control.json'
    if action=='evaluate':
        if path.exists():
            raise FileExistsError(path)
        batches = {k:measure(k,plan) for k in sorted({r['input_key'] for r in rows})}
        verify(batches,plan,rows)
        dump(path,dict(design_sha256=sha(DESIGN),batches=batches,results=summarize(batches,rows)))
    else:
        saved = read(path)
        verify(saved['batches'],plan,rows)
        if saved['design_sha256']!=sha(DESIGN) or saved['results']!=summarize(saved['batches'],rows):
            raise ValueError('Reference-camera control summary differs')
        print('Verified all nine reference-camera checks and exact-recovery counts for three injected depth states.')
