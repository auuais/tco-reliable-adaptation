"""Seal dense photometric depth corrections without reading evaluation references."""
import io
import time
import zipfile

import numpy as np

from collect_matched_focal_heads import RAW as HEADS, tensor_digest
from dense_rgb_depth import ARMS, FACTORS, estimate
from evaluation_stability import dump, sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import INPUT, ROOT, write_arrays
from run_depth_scale_study import PLAN, PUBLIC as PARENT, calibrated_focal, grid, name
from score_saved_geometry import read


RAW = ROOT/'reports/dense_rgb_depth'
PUBLIC = ROOT/'results/dense_rgb_depth'
DESIGN = ROOT/'docs/DENSE_RGB_DEPTH.md'


def inputs(key,rows,plan):
    records = [r for r in rows if r['input_key']==key]
    rgb = plan['inputs'][key]
    head_file = HEADS/key/'heads_and_points.npz'
    if sha(INPUT/rgb['file'])!=rgb['sha256'] or sha(head_file)!=records[0]['heads_sha256']:
        raise ValueError('Changed frozen heads or RGB data')
    with np.load(head_file,allow_pickle=False) as z:
        pred = z['intrinsics'].astype(float)
        poses = z['extrinsics'].astype(float)
        depth = z['depth'][...,0].astype(float)
    with np.load(INPUT/rgb['file'],allow_pickle=False) as z:
        images = z['images']
    cameras = []
    for r in records:
        job, = [j for j in grid() if name(j)==r['condition']]
        prior = plan['priors'][job['prior_key']]
        if sha(INPUT/prior['file'])!=r['prior_sha256'] or r['heads_sha256']!=records[0]['heads_sha256']:
            raise ValueError('Changed camera prior or head identity')
        with np.load(INPUT/prior['file'],allow_pickle=False) as z:
            k,view,edited = calibrated_focal(z['intrinsics'].astype(float),pred)
        if view!=r['candidate_view'] or edited!=r['edited']:
            raise ValueError('Earlier focal decision failed replay')
        np.testing.assert_array_equal(k[:,[0,1],[0,1]],r['focal'])
        cameras.append(k)
    if any(not np.array_equal(k,cameras[0]) for k in cameras):
        raise ValueError('RGB batch has different selected camera hypotheses')
    direct = point_map(depth,cameras[0],poses,np.ones(5))
    direct_hash = tensor_digest(direct)
    if any(r['predictions']['direct']!=direct_hash for r in records):
        raise ValueError('Direct-focal point map differs')
    return images,depth,cameras[0],poses,direct_hash


def main():
    parent = PARENT/'predictions.json'
    rows = read(parent)['rows']
    plan = read(PLAN)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    maps = RAW/'points'
    maps.mkdir(parents=True,exist_ok=True)
    target = PUBLIC/'predictions.json'
    if target.exists():
        raise FileExistsError(target)
    final = PUBLIC/'choices.zip'
    part = final.with_suffix('.partial')
    observations = {}
    with zipfile.ZipFile(part,'w') as archive:
        for key in sorted({r['input_key'] for r in rows}):
            images,depth,k,poses,direct_hash = inputs(key,rows,plan)
            start = time.perf_counter()
            choices,views = estimate(images,depth,k,poses)
            selection_seconds = time.perf_counter()-start
            digests,times = {},{}
            for arm,idx in choices.items():
                start = time.perf_counter()
                points = point_map(depth*FACTORS[idx],k,poses,np.ones(5))
                times[arm] = time.perf_counter()-start
                digests[arm] = tensor_digest(points)
                write_arrays(maps/(digests[arm]+'.npz'),dict(points=points))
            buf = io.BytesIO()
            np.savez_compressed(buf,**choices)
            archive.writestr(key+'.npz',buf.getvalue())
            observations[key] = dict(direct_sha256=direct_hash,point_hashes=digests,selection_seconds=selection_seconds,
                point_construction_seconds=times,choice_hashes={a:tensor_digest(c) for a,c in choices.items()},views=views)
            print(key,'sealed',round(selection_seconds,2),'s RGB scoring',flush=True)
    part.replace(final)
    dump(target,dict(design_sha256=sha(DESIGN),parent_sha256=sha(parent),choices_sha256=sha(final),
                     estimator_sha256=sha(ROOT/'scripts/dense_rgb_depth.py'),batches=observations))


def verify_choices():
    saved = read(PUBLIC/'predictions.json')
    parent = PARENT/'predictions.json'
    rows = read(parent)['rows']
    path = PUBLIC/'choices.zip'
    batches = saved['batches']
    if (set(batches)!={r['input_key'] for r in rows} or saved['design_sha256']!=sha(DESIGN)
            or saved['parent_sha256']!=sha(parent) or saved['choices_sha256']!=sha(path)
            or saved['estimator_sha256']!=sha(ROOT/'scripts/dense_rgb_depth.py')):
        raise ValueError('Dense protocol or inference inputs changed')
    with zipfile.ZipFile(path) as z:
        for key,b in batches.items():
            if any(b['direct_sha256']!=r['predictions']['direct'] for r in rows if r['input_key']==key):
                raise ValueError('Changed baseline identity')
            with np.load(io.BytesIO(z.read(key+'.npz')),allow_pickle=False) as data:
                maps = dict(data)
            for arm in ARMS:
                d = maps[arm]
                if d.shape!=(5,392,518) or d.dtype!=np.uint8 or np.max(d)>=len(FACTORS) or tensor_digest(d)!=b['choice_hashes'][arm]:
                    raise ValueError('Invalid dense choice map')
                for i,s in enumerate(b['views']):
                    if sorted(s['source_order']+[i])!=list(range(5)) or s['view']!=i:
                        raise ValueError('Invalid source split')
                    if s['histograms'][arm]!=np.bincount(d[i].ravel(),minlength=len(FACTORS)).tolist():
                        raise ValueError('Choice histogram differs')
                    if not (0<=s['joint_pixels']<=min(s['fit_pixels'],s['validation_pixels'])<=d[i].size):
                        raise ValueError('Invalid common visibility counts')
            active = maps['guarded']!=0
            if np.any(maps['guarded'][active]!=maps['two_source'][active]):
                raise ValueError('Validation changed a proposed direction')
    return saved


if __name__=='__main__':
    main()
