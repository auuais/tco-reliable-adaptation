"""Cross motion correspondences and camera geometry at identical visible input pixels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from statistics import mean

import numpy as np

from camera_depth_swap import Rig
from collect_spacing_heads import CAPTURE as HEAD_CAPTURE, HEADS, tag as head_tag
from dense_rgb_depth import bilinear
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import centers
from nearby_image_flow import CAPTURE, FLOW, tag
from nearby_motion_depth import GAUGES, INDEX, PRIOR, RULES, pixel_counts
from prepare_source_spacing import FRAMES, PUBLIC, RAW, ROOT, SCENES
from run_dense_rgb_depth import PARENT, PLAN, inputs
from score_saved_geometry import read
from track_camera_selection import bearings
from triangulate_nearby_motion import combine, depth_from_match, motion_pair, synthetic_test


DESIGN = ROOT/'docs/MOTION_SOURCE_SUBSTITUTION.md'
RESULT = PUBLIC/'motion_substitution.json'
CORNERS = ('frozen_observed','frozen_exact','measured_observed','measured_exact')


def reference_match(i, j, xy, depth, source, k, e):
    rig = Rig(e[:,:3,:3], centers(e), k)
    world = bearings(rig,i,xy)*depth.ravel()[:,None]+rig.center[i]
    xyz = (world-rig.center[j])@rig.rotation[j].T
    image = xyz@k[j].T
    with np.errstate(invalid='ignore',divide='ignore'):
        match = image[:,:2]/image[:,2,None]
    sampled, inside = bilinear(source[...,None],match[:,0],match[:,1])
    z = xyz[:,2]
    visible = inside & (z>0) & np.isfinite(sampled[:,0]) & (np.abs(sampled[:,0]-z)<=.02+.02*z)
    return match, visible


def measure_scene(scene):
    key = scene.split('/')[0]+'_base'
    _, original, orig_k, _, _ = inputs(key,read(PARENT/'predictions.json')['rows'],read(PLAN))
    scale = read(PRIOR/'source_transplant.json')['batches'][key]['scale_to_metres']
    gauges = read(GAUGES)['batches'][key+'_short']['views']
    heads = read(HEAD_CAPTURE)['records']
    frames = read(INDEX)['frames'][scene]
    flows = read(CAPTURE)['records']
    results = []
    for i, frame in enumerate(FRAMES):
        name = head_tag(scene,'short',frame)
        h = heads[name]
        path = HEADS/(name+'.npz')
        if h['ids'][i]!=frame or sha(path)!=h['archive_sha256']:
            raise ValueError('Wrong camera capture')
        with np.load(path,allow_pickle=False) as f:
            k, e = f['intrinsics'].astype(float),f['extrinsics'].astype(float)
        k[:,[0,1],[0,1]] = orig_k[i,[0,1],[0,1]]
        metric = scale*gauges[i]['scale_to_original']
        refs = []
        for n in h['ids']:
            path = RAW/'reference'/scene.split('/')[0]/f'{n:06d}.npz'
            if sha(path)!=frames[str(n)]['reference_sha256']:
                raise ValueError('Source reference changed')
            with np.load(path,allow_pickle=False) as f:
                refs.append(dict(f))
        k_ref, e_ref = (np.stack([r[n] for r in refs]) for n in ('k','e'))
        valid, truth = refs[i]['valid'],refs[i]['depth']
        cells = {c:[] for c in CORNERS}
        offsets, support_counts = [],[]
        for j,number in enumerate(h['ids']):
            if j==i:
                continue
            label = tag(scene,frame,number)
            path = FLOW/(label+'.npz')
            if sha(path)!=flows[label]['archive_sha256']:
                raise ValueError('Signed image motion changed')
            with np.load(path,allow_pickle=False) as f:
                xy, observed, cycle = motion_pair(f['forward'], f['reverse'])
            exact, visible = reference_match(i,j,xy,truth,refs[j]['depth'],k_ref,e_ref)
            shared = valid.ravel() & visible & cycle
            support_counts.append(dict(source=number, visible=int((visible&valid.ravel()).sum()),
                                       cycle=int((cycle&valid.ravel()).sum()), shared=int(shared.sum())))
            for rig, intr, ext, unit in [('frozen',k,e,metric),('measured',k_ref,e_ref,1.)]:
                for channel, matches in [('observed',observed),('exact',exact)]:
                    # Coordinates outside the shared source mask have no role in the solution.
                    matches = np.where(shared[:,None],matches,xy)
                    z = depth_from_match(i,j,xy,matches,shared,intr,ext).reshape(valid.shape)*unit
                    cells[rig+'_'+channel].append(z)
            offsets.append(number-frame)
        fused = {c:combine(z,offsets) for c,z in cells.items()}
        modes = {}
        for rule in RULES:
            common = valid & np.logical_and.reduce([np.isfinite(d[rule]) for d in fused.values()])
            oracle = fused['measured_exact'][rule]
            ok = np.isfinite(oracle) & valid
            maximum = float(np.max(np.abs(oracle[ok]/truth[ok]-1))) if ok.any() else None
            if maximum is None or maximum>1e-4:
                raise ValueError('Exact motion with measured cameras failed to recover target depth')
            scores = {c:{region:pixel_counts(original[i]*scale,truth,v[rule],mask)
                         for region,mask in [('all',valid),('common',common)]} for c,v in fused.items()}
            modes[rule] = dict(cells=scores,max_oracle_relative_error=maximum)
        results.append(dict(scene=scene,frame=frame,source_support=support_counts,rules=modes))
    print(scene,'verified oracle depths and paired source footprints',flush=True)
    return results


def summarize(rows):
    def group(targets):
        modes = {}
        for rule in RULES:
            regions = {}
            for region in ('all','common'):
                cells = {}
                for corner in CORNERS:
                    obs = [r['rules'][rule]['cells'][corner][region] for r in targets]
                    valid = [v for v in obs if v['n'] and v['initial_sse']>0]
                    changed = sum(v['changed'] for v in valid)
                    cells[corner] = dict(targets=len(valid),
                        coverage=mean(v['eligible']/v['n'] for v in valid) if valid else None,
                        gain=mean(1-v['final_sse']/v['initial_sse'] for v in valid) if valid else None,
                        improved_among_changed=sum(v['improved'] for v in valid)/changed if changed else None)
                regions[region] = cells
            modes[rule] = regions
        return modes
    return dict(overall=group(rows),by_scene={s:group([r for r in rows if r['scene']==s]) for s in SCENES})


def provenance():
    return {p.name:sha(p) for p in (DESIGN,INDEX,CAPTURE,GAUGES,HEAD_CAPTURE,PRIOR/'source_transplant.json')}


def verify(report):
    rows = report['targets']
    frames = read(INDEX)['frames']
    if report['provenance']!=provenance() or [(r['scene'],r['frame']) for r in rows]!=[(s,f) for s in SCENES for f in FRAMES]:
        raise ValueError('Changed source substitution inputs')
    for row in rows:
        n = frames[row['scene']][str(row['frame'])]['projection_check']['valid_count']
        if len(row['source_support'])!=4 or any(not 0<=v['shared']<=min(v['cycle'],v['visible'])<=n for v in row['source_support']):
            raise ValueError('Invalid paired source footprint')
        for info in row['rules'].values():
            cells = info['cells']
            if set(cells)!=set(CORNERS) or not 0<=info['max_oracle_relative_error']<=1e-4:
                raise ValueError('Invalid oracle control')
            for region in ('all','common'):
                reference = cells['measured_exact'][region]
                if reference['final_sse']>reference['initial_sse']+1e-8:
                    raise ValueError('Oracle increased bounded depth error')
                for obs in cells.values():
                    v = obs[region]
                    if (v['n']!=reference['n'] or (region=='all' and v['n']!=n)
                            or (region=='common' and v['eligible']!=v['n'])
                            or not 0<=v['improved']<=v['changed']<=v['eligible']<=v['n']):
                        raise ValueError('Paired support differs')
                    np.testing.assert_allclose(v['initial_sse'],reference['initial_sse'],rtol=1e-12,atol=1e-10)
    if report['results']!=summarize(rows):
        raise ValueError('Paired substitution summary differs')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('measure','verify'))
    args = parser.parse_args()
    synthetic_test()
    if args.action=='measure':
        if RESULT.exists():
            raise FileExistsError(RESULT)
        with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = [r for scene in pool.map(measure_scene,SCENES) for r in scene]
        result = dict(provenance=provenance(),targets=rows,results=summarize(rows))
        verify(result)
        dump(RESULT,result)
    else:
        verify(read(RESULT))
    print('Verified exact correspondence control, matched input support and all four camera/motion scores.')
