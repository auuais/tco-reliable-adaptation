"""Calculate exact reference-assisted pixel-error optima for nested geometry families."""
import argparse
from statistics import mean

import numpy as np

from collect_matched_focal_heads import PUBLIC as CAPTURES, RAW as HEADS, tensor_digest
from evaluation_stability import dump, sha
from prepare_dense_focal_pilot import REF
from run_depth_scale_study import PLAN, PUBLIC, RAW, ROOT
from score_dense_focal_pilot import REFS
from score_saved_geometry import read


LEVELS = ('global','translations','rotations','scales','affine')
DESIGN = ROOT/'docs/DEPTH_GEOMETRY_HEADROOM.md'


def moments(x,y):
    x,y = x.astype(float),y.astype(float)
    mx,my = x.mean(axis=0),y.mean(axis=0)
    x,y = x-mx,y-my
    return dict(n=len(x),mx=mx.tolist(),my=my.tolist(),xx=(x.T@x).tolist(),
                xy=(x.T@y).tolist(),yy=float(np.sum(y*y)))


def trace_optimum(h):
    u,s,vt = np.linalg.svd(h)
    return s[0]+s[1]+np.sign(np.linalg.det(u@vt))*s[2]


def optima(rows):
    n = np.array([r['n'] for r in rows])
    mx,my,xx,h = (np.array([r[k] for r in rows]) for k in ('mx','my','xx','xy'))
    yy = sum(r['yy'] for r in rows)
    var = np.trace(xx,axis1=1,axis2=2)
    aligned = np.array([trace_optimum(hi) for hi in h])
    if np.any(n<=3) or np.any(var<=0) or np.any(aligned<=0):
        raise ValueError('Degenerate study geometry')
    x = mx - np.average(mx,axis=0,weights=n)
    y = my - np.average(my,axis=0,weights=n)
    total_h = h.sum(axis=0) + x.T@(n[:,None]*y)
    total_x = var.sum()+np.sum(n[:,None]*x*x)
    total_y = yy+np.sum(n[:,None]*y*y)
    affine_gain = sum(np.sum(hi*np.linalg.lstsq(block,hi,rcond=1e-12)[0]) for hi,block in zip(h,xx))
    residuals = np.array([total_y-trace_optimum(total_h)**2/total_x,
                          yy-trace_optimum(h.sum(axis=0))**2/var.sum(),
                          yy-aligned.sum()**2/var.sum(),
                          yy-np.sum(aligned**2/var),yy-affine_gain])
    tol = max(total_y,1.)*1e-10
    if min(residuals)<-tol or max(np.diff(residuals))>tol:
        raise ValueError('Nested optimal residuals violate geometric ordering')
    return {level:dict(sse_m2=float(max(0,res)),rmse_m=float(np.sqrt(max(0,res)/n.sum())))
            for level,res in zip(LEVELS,residuals)}


def synthetic():
    rng = np.random.default_rng(126)
    clouds = [rng.normal(size=(100,3)) for _ in range(5)]
    for expected in range(5):
        rows = []
        for i,x in enumerate(clouds):
            theta = .3*(i if expected>=2 else 1)
            c,s = np.cos(theta),np.sin(theta)
            rot = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            y = (1.+.1*(i if expected>=3 else 1))*x@rot
            y += [2.,i if expected>=1 else 0.,1.]
            if expected==4:
                y[:,0] += .35*y[:,1]
            rows.append(moments(x,y))
        errors = [v['rmse_m'] for v in optima(rows).values()]
        if max(errors[expected:])>1e-6 or (expected and errors[expected-1]<.01):
            raise ValueError('Analytic oracle failed a known transformation')


def summarize(records,seal):
    rows = []
    for case in seal:
        arms = {}
        for arm in ('frozen','direct'):
            k = case['input_key']+'_'+arm
            o = records[k]
            arms[arm] = {ref:optima(o['moments'][ref]) for ref in REFS}
        rows.append(dict(condition=case['condition'],scene=case['scene'],arms=arms))
    def average(group):
        return {ref:{arm:{level:mean(row['arms'][arm][ref][level]['rmse_m'] for row in group)
                          for level in LEVELS} for arm in ('frozen','direct')} for ref in REFS}
    return dict(overall_rmse_m=average(rows),scene_rmse_m={s:average([r for r in rows if r['scene']==s])
                for s in sorted({r['scene'] for r in rows})},rows=rows)


def collect(seal,captures,refs):
    result = {}
    for key in sorted({r['input_key'] for r in seal}):
        direct = {r['predictions']['direct'] for r in seal if r['input_key']==key}
        if len(direct)!=1:
            raise ValueError('Two distinct direct predictions share an RGB key in this study')
        h, = direct
        ref = refs[key]
        if sha(REF/ref['file'])!=ref['sha256']:
            raise ValueError('Depth reference changed')
        with np.load(REF/ref['file'],allow_pickle=False) as z:
            reference = dict(z)
        for arm,path,field,digest in [('frozen',HEADS/key/'heads_and_points.npz','fp32',captures[key]['fp32_sha256']),
                                       ('direct',RAW/'points'/(h+'.npz'),'points',h)]:
            with np.load(path,allow_pickle=False) as z:
                points = z[field]
            if tensor_digest(points)!=digest:
                raise ValueError('Input point map changed')
            obs = {label:[moments(x[m],y[m]) for x,y,m in zip(points,reference[label],reference['valid_mask'])]
                   for label in REFS}
            result[key+'_'+arm] = dict(tensor_sha256=digest,ref_sha256=ref['sha256'],moments=obs)
        print(key,'collected view covariances',flush=True)
    return result


def main(action):
    synthetic()
    if action=='test':
        print('Verified exact fits across five synthetic geometry families.')
        return
    target = PUBLIC/'geometry_headroom.json'
    path = PUBLIC/'predictions.json'
    seal = read(path)['rows']
    captures = {r['input_key']:r for r in read(CAPTURES/'capture.json')['rows']}
    refs = read(PLAN)['references']
    if action=='measure':
        if target.exists():
            raise FileExistsError(target)
        records = collect(seal,captures,refs)
        dump(target,dict(design_sha256=sha(DESIGN),seal_sha256=sha(path),observations=records,results=summarize(records,seal)))
    else:
        data = read(target)
        records = data['observations']
        if data['design_sha256']!=sha(DESIGN) or data['seal_sha256']!=sha(path) or len(records)!=18:
            raise ValueError('Reference-assisted study provenance changed')
        for r in seal:
            key = r['input_key']
            for arm,digest in [('frozen',captures[key]['fp32_sha256']),('direct',r['predictions']['direct'])]:
                rec = records[key+'_'+arm]
                if rec['tensor_sha256']!=digest or rec['ref_sha256']!=refs[key]['sha256']:
                    raise ValueError('Pixel moments refer to different inputs')
                for ref in REFS:
                    m = rec['moments'][ref]
                    if len(m)!=5 or sum(v['n'] for v in m)!=refs[key]['valid_pixels']:
                        raise ValueError('Invalid view counts')
        rebuilt = summarize(records,seal)
        pixel = read(PUBLIC/'pixel_diagnostic.json')['rows']
        for row,p,prior in zip(rebuilt['rows'],pixel,data['results']['rows']):
            if row['condition']!=p['condition'] or row['condition']!=prior['condition']:
                raise ValueError('Reordered camera cases')
            for arm in ('frozen','direct'):
                for ref in REFS:
                    vals = row['arms'][arm][ref]
                    expected = p['metrics'][ref]['fp32' if arm=='frozen' else arm]['rmse_m']
                    np.testing.assert_allclose(vals['global']['rmse_m'],expected,rtol=1e-9)
                    for level in LEVELS:
                        for metric in ('rmse_m','sse_m2'):
                            np.testing.assert_allclose(vals[level][metric],prior['arms'][arm][ref][level][metric],rtol=1e-9)
        for ref in REFS:
            for group,old in [(rebuilt['overall_rmse_m'],data['results']['overall_rmse_m'])]+[
                    (v,data['results']['scene_rmse_m'][k]) for k,v in rebuilt['scene_rmse_m'].items()]:
                for arm in ('frozen','direct'):
                    np.testing.assert_allclose([group[ref][arm][k] for k in LEVELS],[old[ref][arm][k] for k in LEVELS],rtol=1e-9)
        print('Replayed nested transformation optima and cross-checked global errors against individual pixels.')


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('test','measure','verify'))
    main(p.parse_args().action)
