"""Infer and score bounded depth updates with a controlled relative-orientation aid."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from statistics import mean

import numpy as np
from scipy.spatial.transform import Rotation

from collect_spacing_heads import CAPTURE as HEAD_CAPTURE, HEADS, tag as head_tag
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from nearby_image_flow import CAPTURE, FLOW, tag
from nearby_motion_depth import ESTIMATES, GAUGES, INDEX, MAPS, PRIOR, RESULT as ORIGINAL, RULES, pixel_counts
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import FRAMES, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read
from triangulate_nearby_motion import combine, depth_from_match, motion_pair


DESIGN = ROOT/'docs/EXTERNAL_ORIENTATION_CONTROL.md'
SEAL = PUBLIC/'orientation_estimates.json'
RESULT = PUBLIC/'orientation_scores.json'
OUTPUT = RAW/'orientation_anchored'
SEEDS = (11,29,47)
DEGREES = (.25,1.,3.)
WEIGHTS = {'quarter':.25,'unit':1.}


def settings():
    yield 'frozen',None,None
    yield 'exact',0.,None
    for angle in DEGREES:
        for seed in SEEDS:
            yield f'noise_{angle:g}_{seed}',angle,seed


def perturb(rotations, ids, scene, angle, seed):
    matrices = []
    for n in ids:
        key = f'{scene}/{n}/{seed}'.encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'big'))
        axis = rng.normal(size=3)
        matrices.append(Rotation.from_rotvec(np.deg2rad(angle)*axis/np.linalg.norm(axis)).as_matrix())
    return np.stack(matrices)@rotations


def anchored_camera(e, i, rotations):
    # Express the supplied relative rotations in the frozen world coordinates.
    # Keeping the original target exactly avoids rounding-dependent support changes.
    anchored = e.copy()
    relative = rotations@rotations[i].T
    anchored[:,:3,:3] = relative@e[i,:3,:3]
    anchored[:,:3,3] = -np.einsum('nij,nj->ni',anchored[:,:3,:3],centers(e))
    anchored[i] = e[i]
    np.testing.assert_allclose(centers(anchored),centers(e),rtol=1e-5,atol=1e-6)
    return anchored


def estimate_scene(scene):
    key = scene.split('/')[0]+'_base'
    _,_,orig_k,_,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
    rotations = {}
    captures = read(HEAD_CAPTURE)['records']
    references = read(INDEX)['frames'][scene]
    gauges = read(GAUGES)['batches'][key+'_short']['views']
    flow = read(CAPTURE)['records']
    previous = read(ESTIMATES)['targets']
    targets = []
    for i,frame in enumerate(FRAMES):
        name = head_tag(scene,'short',frame)
        record = captures[name]
        ids = record['ids']
        path = HEADS/(name+'.npz')
        if ids[i]!=frame or sha(path)!=record['archive_sha256']:
            raise ValueError('Unexpected frozen camera input')
        with np.load(path,allow_pickle=False) as f:
            k,e = f['intrinsics'].astype(float),f['extrinsics'].astype(float)
        k[:,[0,1],[0,1]] = orig_k[i,[0,1],[0,1]]
        for n in ids:
            if n not in rotations:
                path = RAW/'reference'/scene.split('/')[0]/f'{n:06d}.npz'
                if sha(path)!=references[str(n)]['reference_sha256']:
                    raise ValueError('Orientation source changed')
                with np.load(path,allow_pickle=False) as f:
                    rotations[n] = f['e'][:3,:3]
        measured = np.stack([rotations[n] for n in ids])
        exact = measured@measured[i].T
        matches = []
        offsets = []
        for j,n in enumerate(ids):
            if j==i:
                continue
            label = tag(scene,frame,n)
            path = FLOW/(label+'.npz')
            if sha(path)!=flow[label]['archive_sha256']:
                raise ValueError('Motion field changed')
            with np.load(path,allow_pickle=False) as f:
                matches.append((j,motion_pair(f['forward'],f['reverse'])))
            offsets.append(n-frame)
        old_file = MAPS/(name+'.npz')
        if sha(old_file)!=previous[name]['archive_sha256']:
            raise ValueError('Previous frozen depth changed')
        arrays,bridge,errors = {},{},{}
        for label,angle,seed in settings():
            absolute = e[:,:3,:3] if angle is None else (measured if angle==0 else perturb(measured,ids,scene,angle,seed))
            orient = e if angle is None else anchored_camera(e,i,absolute)
            relative = absolute@absolute[i].T
            rot_error = Rotation.from_matrix(relative@exact.transpose(0,2,1)).magnitude()
            errors[label] = [float(np.rad2deg(rot_error[j])) for j,_ in matches]
            factor = gauges[i]['scale_to_original']
            estimates = [factor*depth_from_match(i,j,xy,q,ok,k,orient).reshape(392,518) for j,(xy,q,ok) in matches]
            for rule,z in combine(estimates,offsets).items():
                z = z.astype(np.float32)
                if label=='frozen':
                    with np.load(old_file,allow_pickle=False) as f:
                        original = f['frozen_'+rule]
                    shared = np.isfinite(z)&np.isfinite(original)
                    different = int(np.sum(np.isfinite(z)!=np.isfinite(original)))
                    largest = float(np.max(np.abs(z[shared]/original[shared]-1)))
                    bridge[rule] = dict(support_difference=different,max_relative_error=largest)
                    if different>10 or largest>1e-4:
                        raise ValueError(f'Frozen camera-coordinate bridge differs: {bridge[rule]}')
                    # Reuse the archived prediction for an exact frozen comparator.
                    z = original
                arrays[label+'_'+rule] = z
        out = OUTPUT/(name+'.npz')
        write_arrays(out,arrays)
        targets.append(dict(scene=scene,frame=frame,file=out.name,sha256=sha(out),bridge=bridge,
                            relative_error_degrees=errors,counts={a:int(np.isfinite(z).sum()) for a,z in arrays.items()}))
    print(scene,'sealed orientation-aided triangulations',flush=True)
    return targets


def provenance():
    return {p.name:sha(p) for p in (DESIGN,HEAD_CAPTURE,CAPTURE,INDEX,GAUGES,ESTIMATES)}


def verify_seal():
    seal = read(SEAL)
    if seal['provenance']!=provenance() or [(r['scene'],r['frame']) for r in seal['targets']]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError('Wrong orientation run inputs')
    for t in seal['targets']:
        if {label for label,_,_ in settings()}!=set(t['relative_error_degrees']):
            raise ValueError('Missing orientation or random seed')
        for label,angle,seed in settings():
            errors = t['relative_error_degrees'][label]
            if len(errors)!=4 or min(errors)<0 or (angle is not None and max(errors)>2*angle+1e-4):
                raise ValueError('Invalid angular corruption')
            for rule in RULES:
                if not 0<=t['counts'][label+'_'+rule]<=392*518:
                    raise ValueError('Invalid prediction coverage')
    return seal['targets']


def verify_backup(targets):
    backup = read(PUBLIC/'orientation_archive.json')
    entries = backup['files']
    if (backup['verified_physical_drives']!=2 or len(entries)!=len(targets)
            or {v['file']:v['sha256'] for v in entries}!={t['file']:t['sha256'] for t in targets}
            or backup['total_bytes']!=sum(v['bytes'] for v in entries) or any(v['bytes']<=0 for v in entries)):
        raise ValueError('Orientation backup inventory differs from sealed estimates')


def measure_scene(scene):
    key = scene.split('/')[0]+'_base'
    _,original,_,_,_ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
    scale = read(PRIOR/'source_transplant.json')['batches'][key]['scale_to_metres']
    frames = read(INDEX)['frames'][scene]
    targets = [r for r in read(SEAL)['targets'] if r['scene']==scene]
    observed = []
    for i,t in enumerate(targets):
        frame = t['frame']
        path = OUTPUT/t['file']
        ref = RAW/'reference'/scene.split('/')[0]/f'{frame:06d}.npz'
        if sha(path)!=t['sha256'] or sha(ref)!=frames[str(frame)]['reference_sha256']:
            raise ValueError('Evaluation source changed')
        with np.load(ref,allow_pickle=False) as f:
            truth,valid = f['depth'],f['valid']
        rows = {}
        with np.load(path,allow_pickle=False) as f:
            for arm in f.files:
                depth = scale*f[arm].astype(float)
                rows[arm] = {key:pixel_counts(scale*original[i],truth,depth,valid,weight) for key,weight in WEIGHTS.items()}
        observed.append(dict(scene=scene,frame=frame,scores=rows))
    return observed


def summarize(targets):
    def group(rows):
        result = {}
        for condition in ['frozen','exact',*[f'noise_{a:g}' for a in DEGREES]]:
            seeds = [condition] if condition in ('frozen','exact') else [f'{condition}_{s}' for s in SEEDS]
            result[condition] = {}
            for rule in RULES:
                result[condition][rule] = {}
                for weight in WEIGHTS:
                    baseline = [r['scores']['frozen_'+rule][weight] for r in rows]
                    seed_gain = []
                    paired = []
                    for s in seeds:
                        cells = [r['scores'][s+'_'+rule][weight] for r in rows]
                        seed_gain.append(mean(1-c['final_sse']/c['initial_sse'] for c in cells))
                        paired.append(mean(1-c['final_sse']/b['final_sse'] for c,b in zip(cells,baseline)))
                    result[condition][rule][weight] = dict(gain=mean(seed_gain),paired=mean(paired),
                        seed_gain=seed_gain,seed_paired=paired)
        return result
    overall = group(targets)
    scenes = {s:group([r for r in targets if r['scene']==s]) for s in SCENES}
    def primary(d):
        return d['noise_1']['any_source']['quarter']
    checks = dict(paired_two_percent=primary(overall)['paired']>=.02,
                  each_scene=all(min(primary(g)[k] for k in ('gain','paired'))>0 for g in scenes.values()))
    return dict(overall=overall,scenes=scenes,checks=checks,consider_surface=all(checks.values()))


def verify_scores(report):
    old = read(ORIGINAL)['targets']
    if report['seal_sha256']!=sha(SEAL) or report['original_sha256']!=sha(ORIGINAL) or len(report['targets'])!=15:
        raise ValueError('Changed orientation evaluation')
    for row,prior in zip(report['targets'],old):
        if (row['scene'],row['frame'])!=(prior['scene'],prior['frame']):
            raise ValueError('Different physical target')
        for rule in RULES:
            original = prior['cells']['frozen_'+rule]['natural']['all']
            for field,v in original.items():
                np.testing.assert_allclose(row['scores']['frozen_'+rule]['unit'][field],v,rtol=1e-10,atol=1e-8)
        for weights in row['scores'].values():
            for v in weights.values():
                if not 0<=v['improved']<=v['changed']<=v['eligible']<=v['n'] or v['final_sse']<0:
                    raise ValueError('Invalid orientation update counts')
                np.testing.assert_allclose(v['initial_sse'],original['initial_sse'],rtol=1e-10)
    if summarize(report['targets'])!=report['results']:
        raise ValueError('Orientation decision failed replay')


def world_gauge_test():
    rotations = Rotation.from_euler('xyz',np.arange(15).reshape(5,3),degrees=True).as_matrix()
    right = Rotation.from_rotvec([.1,-.2,.4]).as_matrix()
    ids = [1,2,3,4,5]
    a = perturb(rotations,ids,'synthetic',1,11)
    b = perturb(rotations@right,ids,'synthetic',1,11)
    np.testing.assert_allclose(a@a[2].T,b@b[2].T,atol=1e-12)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('infer','score','verify'))
    action = p.parse_args().action
    world_gauge_test()
    if action=='infer':
        if SEAL.exists():
            raise FileExistsError(SEAL)
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            targets = [r for group in pool.map(estimate_scene,SCENES) for r in group]
        dump(SEAL,dict(provenance=provenance(),targets=targets))
    sealed_targets = verify_seal()
    if action=='score':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            observations = [r for group in pool.map(measure_scene,SCENES) for r in group]
        report = dict(seal_sha256=sha(SEAL),original_sha256=sha(ORIGINAL),
                      targets=observations,results=summarize(observations))
        verify_scores(report)
        dump(RESULT,report)
    if action=='verify':
        verify_scores(read(RESULT))
        verify_backup(sealed_targets)
    print('Verified external orientation experiment',action)
