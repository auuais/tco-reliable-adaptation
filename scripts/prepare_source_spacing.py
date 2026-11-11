"""Prepare hashed RGB inputs and separate evaluation geometry for fixed target cameras."""
import argparse
from pathlib import Path
import sys

import numpy as np

from audit_sensor_frame import effective_camera
from calibration_audit import resized_k
from collect_matched_focal_heads import tensor_digest
from collect_relative_focal_windows import FRAMES,SCENES,UPSTREAM
from evaluation_stability import dump,sha
from focal_zoom_observations import SCALE,batches
from photometric_positive_control import reference_camera
from prepare_dense_focal_pilot import INPUT,REF,ROOT,write_arrays
from run_depth_scale_study import PLAN
from score_saved_geometry import read
from zoom_dense_reference import zoom_reference


PUBLIC = ROOT/'results/source_frame_spacing'
RAW = ROOT/'reports/source_frame_spacing'
DESIGN = ROOT/'docs/SOURCE_FRAME_SPACING.md'
INDEX = PUBLIC/'inputs.json'
REGIMES = ('dispersed','short','medium')


def sources(frame,regime):
    if regime=='dispersed':
        return [v for v in FRAMES if v!=frame]
    offsets = {'short':(-12,-4,4,12),'medium':(-24,-8,8,24)}
    return [frame+v for v in offsets[regime]]


def scene_frames():
    return sorted({u for t in FRAMES for g in REGIMES for u in [t,*sources(t,g)]})


def prepare(assets):
    sys.path[:0] = [str(UPSTREAM),str(UPSTREAM/'base_models/vggt')]
    from datasets.sevenscenes import SevenScenes
    cache = ROOT/'environment/development_dataset_cache_v7.npy'
    if sha(cache)!=read(ROOT/'manifests/canonical_sensor_factorial.json')['dataset_cache_sha256']:
        raise ValueError('Previous 7-Scenes preparation cache changed')
    dataset = SevenScenes(SEVENSCENES_DIR=str(assets/'data/7scenes'),split='test',load_img_size=518,cache_file=str(cache))
    transform = np.asarray(read(ROOT/'manifests/sensor_frame_v14.json')['calibration'])
    k = resized_k(525.,518,392,uniform=False)
    old = read(PLAN)
    frames = scene_frames()
    results = {}
    for scene in SCENES:
        slug = scene.split('/')[0]
        key = slug+'_base'
        rgb = old['inputs'][key]
        ref = old['references'][key]
        if sha(INPUT/rgb['file'])!=rgb['sha256'] or sha(REF/ref['file'])!=ref['sha256']:
            raise ValueError('Historical target reference changed')
        with np.load(INPUT/rgb['file'],allow_pickle=False) as f:
            original_rgb = f['images']
        with np.load(REF/ref['file'],allow_pickle=False) as f:
            original_world,original_mask = f['registered_rgb'],f['valid_mask']
        image_dir,reference_dir = RAW/'rgb'/slug,RAW/'reference'/slug
        for dst in (image_dir,reference_dir):
            dst.mkdir(parents=True,exist_ok=True)
        views = {}
        for first in range(0,len(frames),5):
            ids = frames[first:first+5]
            data = dataset.get_data(sequence_name=scene,ids=ids)
            tag,images = next(batches(data['images'].numpy()))
            if tag!='base':
                raise ValueError('Unexpected common RGB crop')
            released = np.stack([effective_camera(e) for e in data['extrs']])
            depths = np.einsum('nhwj,nj->nhw',data['pointclouds'],released[:,2,:3])+released[:,2,3,None,None]
            for j,frame in enumerate(ids):
                xyz,m,_ = zoom_reference(depths[j],data['valid_mask'][j],k,transform@released[j],k[:2,2],SCALE)
                xyz = xyz.astype(np.float32)
                fit_k,e,depth,check = reference_camera(xyz,m)
                if frame in FRAMES:
                    i = FRAMES.index(frame)
                    for a,b in ((xyz,original_world[i]),(m,original_mask[i]),(images[j],original_rgb[i])):
                        np.testing.assert_array_equal(a,b)
                name = f'{frame:06d}.npz'
                write_arrays(image_dir/name,dict(rgb=images[j]))
                write_arrays(reference_dir/name,dict(depth=depth,k=fit_k,e=e,valid=m))
                source = assets/'data/7scenes'/scene
                views[str(frame)] = dict(rgb_sha256=sha(image_dir/name),rgb_tensor_sha256=tensor_digest(images[j]),
                    reference_sha256=sha(reference_dir/name),projection_check=check,
                    original={s:sha(source/f'frame-{frame:06d}.{s}') for s in ('color.png','depth.proj.png','pose.txt')})
        results[scene] = views
        print(scene,len(views),'frames staged, all five historical targets reproduced',flush=True)
    record = dict(design_sha256=sha(DESIGN),parent_sha256=sha(PLAN),frames=results,baseline_targets=list(FRAMES))
    if INDEX.exists() and read(INDEX)!=record:
        raise ValueError('Existing preparation differs')
    dump(INDEX,record)


def verify():
    record = read(INDEX)
    if (record['design_sha256']!=sha(DESIGN) or record['parent_sha256']!=sha(PLAN)
            or record['baseline_targets']!=list(FRAMES) or set(record['frames'])!=set(SCENES)):
        raise ValueError('Spacing input manifest differs from the plan')
    for scene,items in record['frames'].items():
        if set(map(int,items))!=set(scene_frames()) or len(items)!=45:
            raise ValueError('Unexpected spacing frame set')
        for row in items.values():
            if (row['projection_check']['max_pixel_error']>.02 or row['projection_check']['valid_count']<=0
                    or any(len(h)!=64 for h in [row['rgb_sha256'],row['rgb_tensor_sha256'],row['reference_sha256'],*row['original'].values()])):
                raise ValueError('Source identity or reference-camera check differs')
    return record


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','verify'))
    p.add_argument('--assets',type=Path)
    args = p.parse_args()
    if args.action=='prepare':
        if args.assets is None:
            p.error('--assets is required for preparation')
        prepare(args.assets)
    verify()
    print('Verified 15 targets and the 120 distinct temporal source frames.')
