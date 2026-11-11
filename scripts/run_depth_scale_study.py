"""Infer twelve controlled cases from frozen heads and disjoint RGB match folds."""
import io
import time
import zipfile

import numpy as np

from collect_matched_focal_heads import RAW as HEADS, PUBLIC as CAPTURES, tensor_digest
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import fit, fold_system, guarded_fit, point_map
from focal_metadata_control import focal_proposal
from prepare_dense_focal_pilot import INPUT, ROOT, jobs, write_arrays
from relative_focal_confirmation import confirmation
from replay_focal_track_decisions import load_folds
from score_saved_geometry import read


RAW = ROOT/'reports/fixed_camera_depth_scale'
PUBLIC = ROOT/'results/fixed_camera_depth_scale'
DESIGN = ROOT/'docs/DEPTH_SCALE_PROTOCOL.md'
PLAN = ROOT/'manifests/dense_focal_inputs.json'
TRACKS = ROOT/'results/relative_focal_windows'
ARMS = ('direct','shared','five','guarded')


def grid():
    return [j for j in jobs() if j['seed']==42 and j['arm']=='retain']


def name(j):
    return j['prior_key']+'_'+j['case']


def calibrated_focal(supplied, predicted):
    axes = [0,1]
    view, options = confirmation(supplied[:,axes,axes], predicted[:,axes,axes], False)
    k = predicted.copy()
    k[:,axes,axes] = supplied[:,axes,axes]
    if options['relative']:
        _, new = focal_proposal(supplied)
        k[view,axes,axes] = new[axes,axes]
    return k, view, options['relative']


def encode_systems(systems):
    fields = {}
    for i, system in enumerate(systems):
        if system is not None:
            fields.update({f'{i}_{k}':v for k,v in zip(('a','y','w','pairs'), system)})
    buf = io.BytesIO()
    np.savez_compressed(buf,**fields)
    return buf.getvalue()


def decode_systems(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return [None if f'{i}_a' not in z else
                (z[f'{i}_a'],z[f'{i}_y'],z[f'{i}_w'],z[f'{i}_pairs'].tolist()) for i in range(2)]


def infer(j, plan, captures, observations, archive):
    key = j['input_key']
    rgb, prior = plan['inputs'][key], plan['priors'][j['prior_key']]
    capture = captures[key]
    tracks = observations[j['scene'],rgb['batch']]
    heads_file = HEADS/key/'heads_and_points.npz'
    if (sha(heads_file)!=capture['archive_sha256'] or sha(INPUT/prior['file'])!=prior['sha256']
            or tracks['rgb_sha256']!=rgb['rgb_sha256'] or capture['rgb_sha256']!=rgb['rgb_sha256']):
        raise ValueError('Camera, image or depth inputs differ from the capture')
    folds = load_folds(archive,tracks)
    with np.load(heads_file,allow_pickle=False) as z:
        depth = z['depth'][...,0].astype(float)
        predicted = z['intrinsics'].astype(float)
        poses = z['extrinsics'].astype(float)
    with np.load(INPUT/prior['file'],allow_pickle=False) as z:
        supplied = z['intrinsics'].astype(float)
    start = time.perf_counter()
    k, view, edit = calibrated_focal(supplied,predicted)
    systems = [fold_system(p,depth,k,poses) for p in folds]
    five, guarded, decision = guarded_fit(*systems,len(k))
    guarded_map = point_map(depth,k,poses,guarded)
    seconds = time.perf_counter()-start
    shared, decision['fitting'] = fit(systems[0],len(k),shared=True)
    scales = dict(direct=np.ones(len(k)),shared=shared,five=five,guarded=guarded)
    hashes = {}
    for arm,s in scales.items():
        points = guarded_map if np.array_equal(s,guarded) else point_map(depth,k,poses,s)
        digest = tensor_digest(points)
        maps = RAW/'points'
        maps.mkdir(parents=True,exist_ok=True)
        path = maps/(digest+'.npz')
        if not path.is_file():
            write_arrays(path,dict(points=points))
        hashes[arm] = digest
    row = dict(condition=name(j), input_key=key, scene=j['scene'], case=j['case'], direction=j['direction'],
               rgb_sha256=rgb['rgb_sha256'], heads_sha256=capture['archive_sha256'], prior_sha256=prior['sha256'],
               matches_sha256=tracks['matches_sha256'], candidate_view=view, edited=edit, focal=k[:,[0,1],[0,1]].tolist(),
               scales={k:v.tolist() for k,v in scales.items()}, decision=decision, predictions=hashes,
               seconds=dict(instrumented_3080_heads=capture['instrumented_seconds'],
                            fresh_cpu_matches=tracks['elapsed_seconds'], guarded_cpu_postprocess=seconds))
    return row, encode_systems(systems)


def main():
    target = PUBLIC/'predictions.json'
    if target.exists():
        raise FileExistsError(target)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    plan = read(PLAN)
    captures = {r['input_key']:r for r in read(CAPTURES/'capture.json')['rows']}
    obs = {(r['camera']['scene'],r['camera']['batch']):r['tracks'] for r in read(TRACKS/'observations.json')}
    rows = []
    final = PUBLIC/'match_systems.zip'
    part = final.with_suffix('.partial')
    with zipfile.ZipFile(TRACKS/'redetected_tracks.zip') as source, zipfile.ZipFile(part,'w') as packed:
        for j in grid():
            row, arrays = infer(j,plan,captures,obs,source)
            packed.writestr(name(j)+'.npz', arrays)
            rows.append(row)
            print(name(j), row['decision']['decision'],flush=True)
    part.replace(final)
    dump(target,dict(design_sha256=sha(DESIGN), input_plan_sha256=sha(PLAN), systems_sha256=sha(final),
                     solver_sha256=sha(ROOT/'scripts/fixed_camera_depth_scale.py'), rows=rows))


if __name__=='__main__':
    main()
