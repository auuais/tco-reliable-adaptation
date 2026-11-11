"""Attribute true-depth reprojection error to five measurable camera components."""
import argparse
import json
from math import factorial

import numpy as np

from camera_depth_swap import Rig
from collect_spacing_heads import CAPTURE as HEAD_CAPTURE, HEADS, tag as head_tag
from dense_rgb_depth import bilinear
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from nearby_image_flow import CAPTURE, FLOW, tag
from nearby_motion_depth import GAUGES, INDEX, PRIOR
from prepare_source_spacing import FRAMES, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read
from track_camera_selection import bearings


DESIGN = ROOT/'docs/NEARBY_RIG_COMPONENTS.md'
RESULT = PUBLIC/'rig_components.json'
PARTS = ('intrinsics','rotation','direction','bundle_scale','length_pattern')
METRICS = ('pixel_sse','linear_log_depth_sse')
FULL = 2**len(PARTS)-1


def relative(i,k,e,scale):
    c = centers(e)
    displacement = scale*((c-c[i])@e[i,:3,:3].T)
    radius = np.linalg.norm(displacement,axis=1)
    used = np.arange(len(e))!=i
    if np.any(radius[used]<=0):
        raise ValueError('Coincident cameras in component study')
    common = float(np.exp(np.log(radius[used]).mean()))
    direction = np.zeros_like(displacement)
    direction[used] = displacement[used]/radius[used,None]
    return k,e[:,:3,:3]@e[i,:3,:3].T,direction,common,radius/common


def project(xy,depth,i,j,config):
    k,rot,direction,common,pattern = config
    hom = np.column_stack((xy,np.ones(len(xy))))
    beam = depth[:,None]*(hom@np.linalg.inv(k[i]).T@rot[j].T)
    xyz = beam-(common*pattern[j]*direction[j])@rot[j].T
    image = xyz@k[j].T
    d_image = beam@k[j].T
    with np.errstate(divide='ignore',invalid='ignore'):
        points = image[:,:2]/image[:,2,None]
        derivative = (d_image[:,:2]-points*d_image[:,2,None])/image[:,2,None]
    return points,derivative,xyz[:,2]


def world_projection(xy,depth,i,j,k,e,scale):
    rig = Rig(e[:,:3,:3],scale*centers(e),k)
    xyz = (rig.center[i]+depth[:,None]*bearings(rig,i,xy)-rig.center[j])@rig.rotation[j].T
    hom = xyz@k[j].T
    return hom[:,:2]/hom[:,2,None]


def measure():
    heads = read(HEAD_CAPTURE)['records']
    refs = read(INDEX)['frames']
    flows = read(CAPTURE)['records']
    gauges = read(GAUGES)['batches']
    metric_scales = read(PRIOR/'source_transplant.json')['batches']
    y,x = np.mgrid[4:392:8,4:518:8]
    xy = np.column_stack((x.ravel(),y.ravel()))
    rows = []
    for scene in SCENES:
        key = scene.split('/')[0]+'_base'
        _,_,base_k,_,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        for i,frame in enumerate(FRAMES):
            name = head_tag(scene,'short',frame)
            record = heads[name]
            path = HEADS/(name+'.npz')
            if sha(path)!=record['archive_sha256'] or record['ids'][i]!=frame:
                raise ValueError('Changed camera bundle')
            with np.load(path,allow_pickle=False) as f:
                fk,fe = f['intrinsics'].astype(float),f['extrinsics'].astype(float)
            fk[:,[0,1],[0,1]] = base_k[i,[0,1],[0,1]]
            scale = gauges[key+'_short']['views'][i]['scale_to_original']*metric_scales[key]['scale_to_metres']
            observed = []
            for number in record['ids']:
                ref = RAW/'reference'/scene.split('/')[0]/f'{number:06d}.npz'
                if sha(ref)!=refs[scene][str(number)]['reference_sha256']:
                    raise ValueError('Changed reference geometry')
                with np.load(ref,allow_pickle=False) as f:
                    observed.append(dict(f))
            rk,re = (np.stack([r[n] for r in observed]) for n in ('k','e'))
            frozen, measured = relative(i,fk,fe,scale),relative(i,rk,re,1.)
            depth = observed[i]['depth'][y,x].ravel()
            target_ok = np.isfinite(depth)
            depth = np.where(target_ok,depth,1.)
            edges = []
            for j,number in enumerate(record['ids']):
                if i==j:
                    continue
                label = tag(scene,frame,number)
                flow = FLOW/(label+'.npz')
                if sha(flow)!=flows[label]['archive_sha256']:
                    raise ValueError('Changed pixel motion')
                with np.load(flow,allow_pickle=False) as f:
                    match = xy+f['forward'][y,x].reshape(-1,2)
                    reverse,inside = bilinear(f['reverse'],match[:,0],match[:,1])
                cycle = inside & (np.linalg.norm(match+reverse-xy,axis=1)<=1)
                truth,g,z = project(xy,depth,i,j,measured)
                source,in_frame = bilinear(observed[j]['depth'][...,None],truth[:,0],truth[:,1])
                visible = in_frame & (z>0) & (np.abs(z-source[:,0])<=.02+.02*z)
                grad_square = np.sum(g*g,axis=1)
                usable = cycle & target_ok & visible & (grad_square>=1.)
                initial = int(usable.sum())
                projections = []
                for mask in range(FULL+1):
                    config = [b if mask&(1<<n) else a for n,(a,b) in enumerate(zip(frozen,measured))]
                    p,_,positive = project(xy,depth,i,j,config)
                    usable &= (positive>0) & np.isfinite(p).all(axis=1)
                    projections.append(p)
                if not usable.any():
                    raise ValueError('A source has no pixels after the fixed support checks')
                differences = []
                for mask,intr,ext,unit in [(0,fk,fe,scale),(FULL,rk,re,1.)]:
                    previous = world_projection(xy[usable],depth[usable],i,j,intr,ext,unit)
                    differences.append(float(np.max(np.abs(projections[mask][usable]-previous))))
                step = 1e-5
                plus = project(xy[usable],depth[usable]*np.exp(step),i,j,measured)[0]
                minus = project(xy[usable],depth[usable]*np.exp(-step),i,j,measured)[0]
                derivative_error = float(np.max(np.abs((plus-minus)/(2*step)-g[usable])))
                if max(differences)>1e-3 or derivative_error>1e-4:
                    raise ValueError('Camera projection or analytic derivative failed its independent check')
                coefficients = []
                for p in projections:
                    residual = (p-truth)[usable]
                    along = np.sum(residual*g[usable],axis=1)/grad_square[usable]
                    coefficients.append([float(np.sum(residual**2)),float(along@along)])
                edges.append(dict(source=number,grid=len(xy),visible_cycle=int((cycle & target_ok & visible).sum()),
                    before_common=initial,n=int(usable.sum()),projection_difference=differences,
                    derivative_error=derivative_error,sums=coefficients))
            rows.append(dict(scene=scene,frame=frame,edges=edges))
        print(scene,'completed all 32 camera substitutions',flush=True)
    return rows


def attribute(values):
    out = {}
    for i,name in enumerate(PARTS):
        bit = 1<<i
        share = 0.
        for mask in range(FULL+1):
            if not mask&bit:
                n = bin(mask).count('1')
                weight = factorial(n)*factorial(len(PARTS)-1-n)/factorial(len(PARTS))
                share += weight*(values[mask]-values[mask|bit])
        out[name] = dict(share=float(share),single_gain=float(values[0]-values[bit]),
                         percent=float(100*share/(values[0]-values[-1])))
    np.testing.assert_allclose(sum(v['share'] for v in out.values()),values[0]-values[-1],rtol=1e-10)
    return dict(frozen=float(values[0]),reference=float(values[-1]),components=out)


def summarize(rows):
    def group(subset):
        by_target = [np.mean([np.asarray(e['sums'])/e['n'] for e in r['edges']],axis=0) for r in subset]
        values = np.mean(by_target,axis=0)
        return {m:attribute(values[:,i]) for i,m in enumerate(METRICS)}
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r['scene']==s]) for s in SCENES})


def support_sensitivity(rows):
    """Reweight the same measured edge sums to check influence of sparse edges."""
    result = {}
    for minimum in (1,25):
        targets = []
        kept_edges = 0
        for row in rows:
            edges = [e for e in row['edges'] if e['n']>=minimum]
            if edges:
                kept_edges += len(edges)
                targets.append(sum(np.asarray(e['sums']) for e in edges)/sum(e['n'] for e in edges))
        pooled = np.mean(targets,axis=0)
        result[minimum] = dict(targets=len(targets),edges=kept_edges,
                               metrics={m:attribute(pooled[:,i]) for i,m in enumerate(METRICS)})
    return result


def provenance():
    return {p.name:sha(p) for p in (DESIGN,CAPTURE,HEAD_CAPTURE,INDEX,GAUGES,PRIOR/'source_transplant.json')}


def verify(report):
    rows = report['targets']
    if report['provenance']!=provenance() or [(r['scene'],r['frame']) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError('Changed rig component inputs')
    for row in rows:
        if sorted(e['source']-row['frame'] for e in row['edges'])!=[-12,-4,4,12]:
            raise ValueError('Wrong physical neighbors')
        for e in row['edges']:
            s = np.asarray(e['sums'])
            if (s.shape!=(32,2) or not np.isfinite(s).all() or np.any(s<0) or np.any(s[-1]>1e-16)
                    or not 0<e['n']<=e['before_common']<=e['visible_cycle']<=e['grid']
                    or max(e['projection_difference'])>1e-3 or e['derivative_error']>1e-4):
                raise ValueError('Invalid shared pixel support, derivative or projection check')
    if report['results']!=summarize(rows):
        raise ValueError('Component attribution differs')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('measure','verify','sensitivity'))
    action = parser.parse_args().action
    if action=='measure':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = measure()
        report = dict(provenance=provenance(),targets=rows,results=summarize(rows))
        verify(report)
        dump(RESULT,report)
    else:
        report = read(RESULT)
        verify(report)
        if action=='sensitivity':
            print(json.dumps(support_sensitivity(report['targets']),indent=2))
    print('Verified 32 camera combinations, both projection endpoints, analytic derivatives and Shapley totals.')
