"""Test a controlled direction aid at fixed frozen camera baseline lengths."""
import argparse
import hashlib
from statistics import mean

import numpy as np

import external_orientation_control as orientation
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from nearby_image_flow import FLOW, tag
from nearby_motion_depth import INDEX, PRIOR, pixel_counts
from prepare_source_spacing import FRAMES, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read
from triangulate_nearby_motion import combine, depth_from_match, motion_pair

DESIGN = ROOT/'docs/JOINT_CAMERA_CONTROL.md'
RESULT = PUBLIC/'joint_camera_control.json'
SEEDS = orientation.SEEDS


def joint_rig(e, measured, i, scene, ids, arm, seed):
    rotations = measured[:,:3,:3]
    if seed is not None:
        rotations = orientation.perturb(rotations, ids, scene, 1., seed)
    rig = e.copy() if arm=='direction' else orientation.anchored_camera(e,i,rotations)
    original = centers(e)
    c = centers(measured)
    for j,n in enumerate(ids):
        if j==i:
            continue
        direction = measured[i,:3,:3] @ (c[j]-c[i])
        direction /= np.linalg.norm(direction)
        if seed is not None:
            label = f'direction/{scene}/{ids[i]}/{n}/{seed}'.encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(label).digest()[:8],'big'))
            tangent = rng.normal(size=3)
            tangent -= direction*(direction@tangent)
            angle = np.deg2rad(5.)
            noisy = np.cos(angle)*direction+np.sin(angle)*tangent/np.linalg.norm(tangent)
            np.testing.assert_allclose(noisy@direction,np.cos(angle),atol=1e-12)
            direction = noisy
        length = np.linalg.norm(original[j]-original[i])
        center = original[i]+length*(e[i,:3,:3].T@direction)
        rig[j,:3,3] = -rig[j,:3,:3]@center
        np.testing.assert_allclose(np.linalg.norm(centers(rig)[j]-original[i]),length,rtol=1e-5,atol=1e-7)
    return rig


def measure():
    records = read(orientation.HEAD_CAPTURE)['records']
    flows = read(orientation.CAPTURE)['records']
    gauges = read(orientation.GAUGES)['batches']
    refs = read(INDEX)['frames']
    scales = read(PRIOR/'source_transplant.json')['batches']
    rows = []
    for scene in SCENES:
        slug = scene.split('/')[0]
        key = slug+'_base'
        _,original,k0,_,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
        metres = scales[key]['scale_to_metres']
        for i,frame in enumerate(FRAMES):
            name = orientation.head_tag(scene,'short',frame)
            record = records[name]
            ids = record['ids']
            head = orientation.HEADS/(name+'.npz')
            if ids[i]!=frame or sha(head)!=record['archive_sha256']:
                raise ValueError('Camera input changed')
            with np.load(head) as f:
                k,e = f['intrinsics'].astype(float),f['extrinsics'].astype(float)
            k[:,[0,1],[0,1]] = k0[i,[0,1],[0,1]]
            cams, pairs, offsets = [],[],[]
            for j,n in enumerate(ids):
                path = RAW/'reference'/slug/f'{n:06d}.npz'
                if sha(path)!=refs[scene][str(n)]['reference_sha256']:
                    raise ValueError('Reference camera changed')
                with np.load(path) as f:
                    cams.append(f['e'])
                if j==i:
                    continue
                label = tag(scene,frame,n)
                path = FLOW/(label+'.npz')
                if sha(path)!=flows[label]['archive_sha256']:
                    raise ValueError('Motion input changed')
                with np.load(path) as f:
                    pairs.append((j,motion_pair(f['forward'],f['reverse'])))
                offsets.append(n-frame)
            factor = gauges[key+'_short']['views'][i]['scale_to_original']
            predictions = {}
            for arm,seed in [('direction',None),('exact',None),*[('joint_'+str(s),s) for s in SEEDS]]:
                rig = joint_rig(e,np.stack(cams),i,scene,ids,arm,seed)
                depths = [factor*depth_from_match(i,j,p,q,ok,k,rig).reshape(392,518) for j,(p,q,ok) in pairs]
                predictions[arm] = combine(depths,offsets)['any_source'].astype(np.float32)
            with np.load(RAW/'reference'/slug/f'{frame:06d}.npz') as f:
                truth, valid = f['depth'],f['valid']
            scores = {a:pixel_counts(metres*original[i],truth,metres*z.astype(float),valid,.25) for a,z in predictions.items()}
            rows.append(dict(scene=scene,frame=frame,scores=scores))
        print(scene,flush=True)
    return rows


def summarize(rows):
    prior = {(r['scene'],r['frame']):r['scores'] for r in read(orientation.RESULT)['targets']}
    if [(r['scene'],r['frame']) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError('Incomplete target set')
    def group(items):
        arms = {}
        for arm in ('direction','exact','joint'):
            gains = []
            for r in items:
                old = prior[r['scene'],r['frame']]
                frozen = old['frozen_any_source']['quarter']
                for seed in SEEDS if arm=='joint' else (None,):
                    tag = f'joint_{seed}' if seed else arm
                    ref = f'noise_1_{seed}' if seed else 'exact'
                    v = r['scores'][tag]
                    np.testing.assert_allclose(v['initial_sse'],frozen['initial_sse'],rtol=1e-10)
                    if v['n']!=frozen['n'] or not 0<=v['improved']<=v['changed']<=v['eligible']<=v['n']:
                        raise ValueError('Mismatched support')
                    denominators = [v['initial_sse'],frozen['final_sse'],old[ref+'_any_source']['quarter']['final_sse']]
                    gains.append([1-v['final_sse']/d for d in denominators])
            arms[arm] = dict(zip(('unchanged','frozen','orientation'),map(mean,zip(*gains))))
        return arms
    scenes = {s:group([r for r in rows if r['scene']==s]) for s in SCENES}
    overall = group(rows)
    advance = overall['joint']['frozen']>=.02 and all(min(g['joint'].values())>0 for g in scenes.values())
    return dict(overall=overall,scenes=scenes,consider_surface=advance)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=('measure','verify'))
    provenance = {p.name:sha(p) for p in (DESIGN,INDEX,orientation.RESULT,orientation.HEAD_CAPTURE,orientation.CAPTURE,orientation.GAUGES,PRIOR/'source_transplant.json')}
    if parser.parse_args().action=='measure':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        rows = measure()
        dump(RESULT,dict(inputs=provenance,targets=rows,results=summarize(rows)))
    report = read(RESULT)
    if report['inputs']!=provenance or summarize(report['targets'])!=report['results']:
        raise ValueError('Failed joint-camera replay')
    print(report['results']['overall'],report['results']['consider_surface'])
