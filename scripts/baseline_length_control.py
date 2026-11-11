"""Separate bundle scale from relative baseline lengths in paired triangulation."""
import argparse
from statistics import mean

import numpy as np

import joint_camera_control as joint
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from prepare_source_spacing import PUBLIC, ROOT, SCENES
from score_saved_geometry import read
from triangulate_nearby_motion import combine, depth_from_match

DESIGN = ROOT/'docs/BASELINE_LENGTH_CONTROL.md'
RESULT = PUBLIC/'baseline_lengths.json'
POLICIES = ('original','scale','pattern','both')


def predictions(e, measured, i, scene, ids, factor, metres, pairs, offsets, k):
    neighbors = [j for j,*_ in pairs]
    def lengths(camera):
        c = centers(camera)
        return np.linalg.norm(c[neighbors]-c[i],axis=1)
    ratio = lengths(measured)/(lengths(e)*factor*metres)
    scale = np.exp(np.mean(np.log(ratio)))
    multipliers = dict(original=np.ones(4),scale=np.full(4,scale),pattern=ratio/scale,both=ratio)
    result = {}
    exact = joint.joint_rig(e,measured,i,scene,ids,'exact',None)
    for label,rig in [('frozen',e),('exact',exact)]:
        depths = []
        for index,(j,(p,q,ok)) in enumerate(pairs):
            z = depth_from_match(i,j,p,q,ok,k,rig)
            # Scale the entire relative translation in the source camera frame.
            trial = rig.copy()
            center = centers(rig)[i]
            rotation = rig[j,:3,:3]
            trial[j,:3,3] = ratio[index]*(rotation@center+rig[j,:3,3])-rotation@center
            sl = slice(0,None,23)
            replay = depth_from_match(i,j,p[sl],q[sl],ok[sl],k,trial)
            # Small non-orthogonality in stored FP32 rotations can affect marginal gates.
            mask = np.isfinite(replay)&np.isfinite(z[sl])
            if np.count_nonzero(np.isfinite(replay)!=np.isfinite(z[sl]))>1:
                raise ValueError('Baseline rescaling changed triangulation support')
            np.testing.assert_allclose(replay[mask],ratio[index]*z[sl][mask],rtol=1e-4,atol=1e-6)
            depths.append(factor*z.reshape(392,518))
        for policy,weights in multipliers.items():
            adjusted = [v*w for v,w in zip(depths,weights)]
            result[label+'_'+policy] = combine(adjusted,offsets)['any_source'].astype(np.float32)
    return result


def summarize(rows):
    orientation = read(joint.orientation.RESULT)['targets']
    prior = read(joint.RESULT)['targets']
    if len(rows)!=len(prior) or len(rows)!=15:
        raise ValueError('Missing target')
    for row,a,b in zip(rows,orientation,prior):
        for key in ('scene','frame'):
            if row[key]!=a[key] or row[key]!=b[key]:
                raise ValueError('Target ordering differs')
        for rig,old in [('exact',b['scores']['exact']),('frozen',a['scores']['frozen_any_source']['quarter'])]:
            for field,value in old.items():
                np.testing.assert_allclose(row['scores'][rig+'_original'][field],value,rtol=1e-8,atol=1e-8)
        for v in row['scores'].values():
            if v['n']!=old['n'] or not 0<=v['improved']<=v['changed']<=v['eligible']<=v['n']:
                raise ValueError('Invalid support')
            np.testing.assert_allclose(v['initial_sse'],old['initial_sse'])
    def group(items):
        result = {}
        for rig in ('frozen','exact'):
            for policy in POLICIES:
                arm = rig+'_'+policy
                gains = []
                for row in items:
                    scores = row['scores']
                    obs = scores[arm]
                    denominators = [obs['initial_sse'],scores['frozen_original']['final_sse'],scores[rig+'_original']['final_sse']]
                    gains.append([1-obs['final_sse']/d for d in denominators])
                result[arm] = dict(zip(('unchanged','frozen','same_rig'),map(mean,zip(*gains))))
        return result
    return dict(overall=group(rows),scenes={s:group([r for r in rows if r['scene']==s]) for s in SCENES})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('measure','verify'))
    deps = {p.name:sha(p) for p in (DESIGN,joint.RESULT,joint.orientation.RESULT,joint.INDEX,joint.orientation.HEAD_CAPTURE,joint.orientation.CAPTURE,joint.orientation.GAUGES,joint.PRIOR/'source_transplant.json')}
    if parser.parse_args().action=='measure':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = joint.measure(predictions)
        dump(RESULT,dict(inputs=deps,targets=rows,results=summarize(rows)))
    report = read(RESULT)
    if report['inputs']!=deps or report['results']!=summarize(report['targets']):
        raise ValueError('Baseline-length comparison failed replay')
    print(report['results']['overall'])
