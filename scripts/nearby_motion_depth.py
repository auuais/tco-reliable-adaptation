"""Measure the depth signal provided by nearby RGB motion before any surface redesign."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from statistics import mean
import time

import numpy as np

from collect_matched_focal_heads import tensor_digest
from collect_spacing_heads import CAPTURE as HEAD_CAPTURE, HEADS, tag as head_tag
from evaluation_stability import dump, sha
from nearby_image_flow import CAPTURE, DESIGN, FLOW, tag, verify as verify_flow
from photometric_positive_control import STARTS
from predict_temporal_depth import SEAL as GAUGES
from prepare_dense_focal_pilot import write_arrays
from prepare_source_spacing import FRAMES, INDEX, PUBLIC, RAW, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read
from triangulate_nearby_motion import combine, synthetic_test, triangulate
from visibility_positive_control import PUBLIC as PRIOR


ESTIMATES = PUBLIC/'nearby_motion_estimates.json'
RESULT = PUBLIC/'nearby_motion_scores.json'
MAPS = RAW/'flow_depth'
RIGS = ('frozen', 'measured')
RULES = ('any_source', 'two_sided')
STATES = tuple(STARTS)+('natural',)
REGIONS = ('all', 'eligible')


def estimate_scene(scene):
    key = scene.split('/')[0]+'_base'
    _, _, original_k, _, _ = inputs(key, read(PARENT/'predictions.json')['rows'], read(PLAN))
    gauges = read(GAUGES)['batches'][key+'_short']['views']
    heads = read(HEAD_CAPTURE)['records']
    flow = read(CAPTURE)['records']
    refs = read(INDEX)['frames'][scene]
    output = {}
    for i, frame in enumerate(FRAMES):
        name = head_tag(scene,'short',frame)
        record = heads[name]
        ids = record['ids']
        if ids[i]!=frame or sha(HEADS/(name+'.npz'))!=record['archive_sha256']:
            raise ValueError('Unrecognized camera bundle')
        with np.load(HEADS/(name+'.npz'), allow_pickle=False) as f:
            k = f['intrinsics'].astype(float)
            e = f['extrinsics'].astype(float)
        k[:,[0,1],[0,1]] = original_k[i,[0,1],[0,1]]
        scale = gauges[i]['scale_to_original']
        cameras = []
        for number in ids:
            path = RAW/'reference'/scene.split('/')[0]/f'{number:06d}.npz'
            if sha(path)!=refs[str(number)]['reference_sha256']:
                raise ValueError('Unknown reference-camera file')
            with np.load(path, allow_pickle=False) as f:
                cameras.append((f['k'],f['e']))
        measured_k, measured_e = (np.stack(group) for group in zip(*cameras))
        single = {r:[] for r in RIGS}
        offsets = []
        start = time.perf_counter()
        for j, source in enumerate(ids):
            if j==i:
                continue
            pair = tag(scene,frame,source)
            path = FLOW/(pair+'.npz')
            if sha(path)!=flow[pair]['archive_sha256']:
                raise ValueError('Signed flow field changed')
            with np.load(path, allow_pickle=False) as f:
                forward, reverse = f['forward'], f['reverse']
            single['frozen'].append(scale*triangulate(i,j,forward,reverse,k,e))
            single['measured'].append(triangulate(i,j,forward,reverse,measured_k,measured_e))
            offsets.append(source-frame)
        arrays = {}
        for rig in RIGS:
            for rule, depth in combine(single[rig],offsets).items():
                arrays[rig+'_'+rule] = depth.astype(np.float32)
        elapsed = time.perf_counter()-start
        path = MAPS/(name+'.npz')
        write_arrays(path, arrays)
        output[name] = dict(scene=scene, frame=frame, seconds=elapsed, scale_to_original=scale,
            archive_sha256=sha(path), tensor_sha256={a:tensor_digest(v) for a,v in arrays.items()},
            eligible={a:int(np.isfinite(v).sum()) for a,v in arrays.items()})
    print(scene,'sealed calibrated triangulations',flush=True)
    return output


def provenance():
    return {p.name:sha(p) for p in (DESIGN,CAPTURE,HEAD_CAPTURE,GAUGES,INDEX)}


def verify_estimates():
    verify_flow()
    report = read(ESTIMATES)
    arms = {r+'_'+a for r in RIGS for a in RULES}
    expected = {head_tag(s,'short',f) for s in SCENES for f in FRAMES}
    if report['provenance']!=provenance() or set(report['targets'])!=expected:
        raise ValueError('Changed motion-depth manifest')
    for name, item in report['targets'].items():
        if (name!=head_tag(item['scene'],'short',item['frame']) or item['seconds']<=0 or item['scale_to_original']<=0
                or set(item['eligible'])!=arms or set(item['tensor_sha256'])!=arms
                or any(not 0<=v<=392*518 for v in item['eligible'].values())
                or any(item['eligible'][r+'_two_sided']>item['eligible'][r+'_any_source'] for r in RIGS)):
            raise ValueError('Invalid motion-depth capture')
    return report['targets']


def pixel_counts(start, truth, estimate, mask):
    active = np.isfinite(estimate) & mask
    modification = np.zeros_like(truth)
    modification[active] = np.clip(np.log(estimate[active]/start[active]),-.08,.08)
    initial = np.log(start[mask]/truth[mask])
    final = initial+modification[mask]
    changed = modification[mask]!=0
    return dict(n=len(initial), eligible=int(active.sum()), changed=int(changed.sum()),
        improved=int(np.sum(changed & (np.abs(final)<np.abs(initial)))),
        close=int((np.abs(final)<=.02).sum()), initial_sse=float(initial@initial),
        final_sse=float(final@final), final_abs=float(np.abs(final).sum()))


def measure_scene(scene):
    key = scene.split('/')[0]+'_base'
    _, original, _, _, _ = inputs(key, read(PARENT/'predictions.json')['rows'], read(PLAN))
    scale = read(PRIOR/'source_transplant.json')['batches'][key]['scale_to_metres']
    targets = read(ESTIMATES)['targets']
    refs = read(INDEX)['frames'][scene]
    observations = []
    for i, frame in enumerate(FRAMES):
        name = head_tag(scene,'short',frame)
        path = MAPS/(name+'.npz')
        target_ref = RAW/'reference'/scene.split('/')[0]/f'{frame:06d}.npz'
        if sha(path)!=targets[name]['archive_sha256'] or sha(target_ref)!=refs[str(frame)]['reference_sha256']:
            raise ValueError('Sealed depth or reference changed')
        with np.load(target_ref,allow_pickle=False) as f:
            truth, valid = f['depth'], f['valid']
        states = {s:truth*np.exp(v) for s,v in STARTS.items()}
        states['natural'] = original[i]*scale
        cells = {}
        with np.load(path,allow_pickle=False) as maps:
            for arm in maps.files:
                estimate = maps[arm].astype(float)
                if arm.startswith('frozen_'):
                    estimate *= scale
                support = np.isfinite(estimate) & valid
                cells[arm] = {s:{region:pixel_counts(start,truth,estimate,mask)
                    for region,mask in zip(REGIONS,(valid,support))} for s,start in states.items()}
        observations.append(dict(scene=scene,frame=frame,cells=cells))
    return observations


def summarize(rows):
    def subset(group):
        output = {}
        for arm in group[0]['cells']:
            states = {}
            for state in STATES:
                states[state] = {}
                for region in REGIONS:
                    items = [r['cells'][arm][state][region] for r in group]
                    usable = [v for v in items if v['n']]
                    def fraction(a,b):
                        return mean(v[a]/v[b] for v in usable) if usable else None
                    edits = sum(v['changed'] for v in items)
                    reduction = (1-fraction('final_sse','initial_sse') if usable and state!='clean' else None)
                    states[state][region] = dict(targets=len(usable),coverage=fraction('eligible','n'),
                        close=fraction('close','n'),final_mae=fraction('final_abs','n'),sse_reduction=reduction,
                        improved_among_changed=sum(v['improved'] for v in items)/edits if edits else None)
            output[arm] = states
        return output
    overall = subset(rows)
    scenes = {s:subset([r for r in rows if r['scene']==s]) for s in SCENES}
    def primary(cell, rule='two_sided'):
        return cell['frozen_'+rule]['natural']['all']['sse_reduction']
    checks = dict(each_scene=all(primary(s)>0 for s in scenes.values()), overall=primary(overall)>0,
                  simpler=primary(overall)>primary(overall,'any_source'))
    return dict(overall=overall,by_scene=scenes,checks=checks,surface_redesign=all(checks.values()))


def check_scores(report, estimates):
    if (report['estimates_sha256']!=sha(ESTIMATES) or report['prior_sha256']!=sha(PRIOR/'source_transplant.json')
            or report['cloud_usd']!=0 or [(r['scene'],r['frame']) for r in report['targets']]!=[(s,f) for s in SCENES for f in FRAMES]):
        raise ValueError('Motion depth score provenance differs')
    refs = read(INDEX)['frames']
    for row in report['targets']:
        scene, frame = row['scene'],row['frame']
        targets = estimates[head_tag(scene,'short',frame)]
        n = refs[scene][str(frame)]['projection_check']['valid_count']
        for arm, states in row['cells'].items():
            for state, regions in states.items():
                a,b = regions['all'],regions['eligible']
                if (a['n']!=n or a['eligible']!=b['n'] or b['n']!=b['eligible']
                        or b['n']>targets['eligible'][arm] or a['changed']!=b['changed'] or a['improved']!=b['improved']
                        or not 0<=a['improved']<=a['changed']<=b['n'] or min(a['final_sse'],b['final_sse'])<0):
                    raise ValueError('Motion depth accounting differs')
                np.testing.assert_allclose(a['initial_sse']-a['final_sse'],b['initial_sse']-b['final_sse'],rtol=1e-8,atol=1e-8)
                if state in STARTS:
                    np.testing.assert_allclose(a['initial_sse'],n*STARTS[state]**2,rtol=1e-10,atol=1e-10)
    if report['results']!=summarize(report['targets']):
        raise ValueError('Motion depth decision differs')


def verify_backup(estimates):
    archive = read(PUBLIC/'motion_archive.json')
    expected = {f'flow_depth/{label}.npz':r['archive_sha256'] for label,r in estimates.items()}
    for label, record in read(CAPTURE)['records'].items():
        expected[f'flow/{label}.npz'] = record['archive_sha256']
        content = (json.dumps(record, indent=2, allow_nan=False)+'\n').encode()
        expected[f'flow/{label}.json'] = hashlib.sha256(content).hexdigest()
    entries = archive['files']
    if (archive['verified_physical_drives']!=2 or {e['file']:e['sha256'] for e in entries}!=expected
            or len(entries)!=len(expected) or any(e['bytes']<=0 for e in entries)
            or archive['total_bytes']!=sum(e['bytes'] for e in entries)):
        raise ValueError('Optical flow or triangulation backup inventory differs')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('infer','score','verify'))
    action = parser.parse_args().action
    synthetic_test()
    if action=='infer':
        verify_flow()
        if ESTIMATES.exists():
            raise FileExistsError(ESTIMATES)
        MAPS.mkdir(parents=True,exist_ok=True)
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            records = {k:v for scene in pool.map(estimate_scene,SCENES) for k,v in scene.items()}
        dump(ESTIMATES,dict(provenance=provenance(),targets=records))
    estimates = verify_estimates()
    if action=='score':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [r for group in pool.map(measure_scene,SCENES) for r in group]
        report = dict(estimates_sha256=sha(ESTIMATES),prior_sha256=sha(PRIOR/'source_transplant.json'),
                      targets=rows,results=summarize(rows),cloud_usd=0)
        check_scores(report,estimates)
        dump(RESULT,report)
    if action=='verify':
        check_scores(read(RESULT),estimates)
        verify_backup(estimates)
    print('Verified calibrated RGB-motion depth',action)
