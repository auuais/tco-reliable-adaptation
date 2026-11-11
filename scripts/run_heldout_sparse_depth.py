"""Seal held-out sparse-depth reconstructions before evaluating reserved target labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from audit_sensor_frame import effective_camera
from calibration_audit import resized_k
from capture_heldout_sparse_depth import PUBLIC, RAW, images
from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump,sha
from fixed_camera_depth_scale import point_map
from focal_zoom_observations import SCALE,UPSTREAM
from neighbor_sparse_depth import corrections,project_samples
from prepare_dense_focal_pilot import ROOT,write_arrays
from score_saved_geometry import read
from supplied_pose_sparse_depth import METHODS,place
from zoom_dense_reference import zoom_reference

GRIDS={'centered':(40,30),'shifted':(20,15)}
REFS=('registered_rgb','released')


def predict(assets):
    index=read(PUBLIC/'input_index.json')
    calibration=np.asarray(read(ROOT/'manifests/sensor_frame_v14.json')['calibration'])
    target=resized_k(525)
    target[[0,1],[0,1]]*=SCALE
    native=resized_k(525,640,480)
    batches={}
    packets={}
    for scene,entry in index['sequences'].items():
        key=scene.replace('/','_')
        meta=read(PUBLIC/'captures'/(key+'.json'))
        path=RAW/(key+'.npz')
        assert sha(path)==meta['archive_sha256']
        with np.load(path) as heads:
            depth=heads['depth'][...,0].astype(float)
            model=heads['extrinsics'].astype(float)
            k=heads['intrinsics'].astype(float)
            frozen=heads['native']
        assert len(depth)==len(entry['targets'])
        k[:,[0,1],[0,1]]=target[[0,1],[0,1]].astype(np.float32)
        def pose(f):
            path=assets/scene/f'frame-{f:06d}.pose.txt'
            assert sha(path)==entry['files'][str(f)]['pose.txt']
            return calibration@np.linalg.inv(np.loadtxt(path))
        poses={f:pose(f) for f in entry['sources']+entry['targets']}
        supplied=np.stack([poses[f] for f in entry['targets']])
        arms={'frozen':frozen,'direct':point_map(depth,k,model,np.ones(len(depth)))}
        details={}
        for grid,origin in GRIDS.items():
            u,v=np.meshgrid(np.arange(origin[0],640,80),np.arange(origin[1],480,60))
            uv=np.column_stack([u.ravel(),v.ravel()])
            rays=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(native).T
            samples=[]
            for frame in entry['targets']:
                observations=[]
                for source in (frame-4,frame+4):
                    assert source not in entry['targets']
                    path=assets/scene/f'frame-{source:06d}.depth.proj.png'
                    assert sha(path)==entry['files'][str(source)]['depth.proj.png']
                    with Image.open(path) as image:
                        mm=np.asarray(image)[uv[:,1],uv[:,0]]
                    observations.extend(project_samples(mm,uv,rays,poses[frame]@np.linalg.inv(poses[source]),target,source))
                nearest={}
                for row in sorted(observations,key=lambda r:r[2]):
                    nearest.setdefault(tuple(row[:2]),row)
                sample=np.array(list(nearest.values()))
                assert len(sample)>=8
                packets[f'{key}_{grid}_{frame}']=sample
                samples.append(sample)
            updates,gauge=corrections(depth,samples)
            updates['direct']=depth
            details[grid]=dict(gauge=gauge,counts=list(map(len,samples)))
            for arm in METHODS:
                original=point_map(np.asarray(updates[arm]),k,model,np.ones(len(depth)))
                arms[grid+'_'+arm]=place(original,model,supplied,gauge)
        hashes={}
        (RAW/'points').mkdir(exist_ok=True)
        for arm,points in arms.items():
            h=tensor_digest(points)
            write_arrays(RAW/'points'/(h+'.npz'),dict(points=points))
            hashes[arm]=h
        batches[scene]=dict(hashes=hashes,grids=details,capture_sha256=sha(PUBLIC/'captures'/(key+'.json')))
        print(scene,'ten predictions sealed',flush=True)
    write_arrays(PUBLIC/'measurements.npz',packets)
    result=dict(input_sha256=sha(PUBLIC/'input_index.json'),packet_sha256=sha(PUBLIC/'measurements.npz'),batches=batches)
    path=PUBLIC/'predictions.json'
    if path.exists() and read(path)!=result:
        raise ValueError('Changed held-out predictions')
    dump(path,result)


def evaluate_scene(scene,assets):
    import torch
    import open3d as o3d
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'base_models/vggt')]
    from datasets.sevenscenes import SevenScenes
    from run_repeat_calibration import evaluate
    torch.set_num_threads(2)
    seal=read(PUBLIC/'predictions.json')
    assert seal['input_sha256']==sha(PUBLIC/'input_index.json')
    assert seal['packet_sha256']==sha(PUBLIC/'measurements.npz')
    entry=read(PUBLIC/'input_index.json')['sequences'][scene]
    dataset=SevenScenes.__new__(SevenScenes)
    dataset.SEVENSCENES_DIR=str(assets)
    dataset.load_img_size=518
    dataset.sequence_list=[scene]
    dataset.metadata={scene:max(entry['targets'])+1}
    data=dataset.get_data(sequence_name=scene,ids=entry['targets'])
    released=np.stack([effective_camera(e) for e in data['extrs']])
    depth=np.einsum('nhwj,nj->nhw',data['pointclouds'],released[:,2,:3])+released[:,2,3,None,None]
    calibration=np.asarray(read(ROOT/'manifests/sensor_frame_v14.json')['calibration'])
    nominal=resized_k(525)
    center=nominal[:2,2]
    refs={}
    mask=None
    for label in REFS:
        points=[]
        valid=[]
        for i in range(len(depth)):
            k=nominal if label=='registered_rgb' else data['intrs'][i]
            e=calibration@released[i] if label=='registered_rgb' else released[i]
            xyz,m,_=zoom_reference(depth[i],data['valid_mask'][i],k,e,center,SCALE)
            points.append(xyz.astype(np.float32))
            valid.append(m)
        refs[label]=np.stack(points)
        current=np.stack(valid)
        if mask is not None:
            np.testing.assert_array_equal(mask,current)
        mask=current
    rgb=torch.from_numpy(images(assets,scene,entry))
    observed={}
    for digest in sorted(set(seal['batches'][scene]['hashes'].values())):
        observed[digest]={}
        with np.load(RAW/'points'/(digest+'.npz')) as z:
            points=z['points']
        assert tensor_digest(points)==digest
        for label in REFS:
            destination=RAW/'evaluation'/scene.replace('/','_')/digest/label
            path=destination/'checked.json'
            if path.exists():
                result=read(path)
                assert result['prediction']==digest and result['reference']==label
            else:
                destination.mkdir(parents=True,exist_ok=True)
                np.random.seed(42)
                torch.manual_seed(42)
                o3d.utility.random.seed(42)
                metrics=evaluate(points,dict(images=rgb,pointclouds=refs[label],valid_mask=mask),destination)
                result=dict(prediction=digest,reference=label,metrics=metrics)
                dump(path,result)
            observed[digest][label]=result
        print(scene,digest[:8],'both references scored',flush=True)
    return scene,observed


def summarize(seal,observations):
    outcomes={}
    for grid in GRIDS:
        references={}
        for ref in REFS:
            sequences={}
            for scene,b in seal['batches'].items():
                values=observations[scene]
                frozen=values[b['hashes']['frozen']][ref]['metrics']
                cell={}
                for arm,h in b['hashes'].items():
                    if arm in ('frozen','direct') or arm.startswith(grid+'_'):
                        metrics=values[h][ref]['metrics']
                        ratios=[metrics[m]/frozen[m] for m in ('accuracy_mean_m','completeness_mean_m')]
                        cell[arm]=dict(score=float(np.mean(ratios)),accuracy=ratios[0],completeness=ratios[1],raw=metrics)
                sequences[scene]=cell
            scenes={}
            for slug in sorted({s.split('/')[0] for s in sequences}):
                group=[v for s,v in sequences.items() if s.split('/')[0]==slug]
                scenes[slug]={a:{m:float(np.mean([v[a][m] for v in group])) for m in ('score','accuracy','completeness')} for a in cell}
            overall={a:{m:float(np.mean([v[a][m] for v in scenes.values()])) for m in ('score','accuracy','completeness')} for a in cell}
            references[ref]=dict(overall=overall,scenes=scenes,sequences=sequences)
        primary=references[REFS[0]]
        candidate=grid+'_guarded'
        best=min((a for a in overall if a!=candidate),key=lambda a:primary['overall'][a]['score'])
        gain=1-primary['overall'][candidate]['score']/primary['overall'][best]['score']
        per_scene={s:1-v[candidate]['score']/v[best]['score'] for s,v in primary['scenes'].items()}
        checks=dict(gain=gain>=.05,components=all(primary['overall'][candidate][m]<=1.02*primary['overall'][best][m] for m in ('accuracy','completeness')),
                    all_scenes=min(per_scene.values())>=0,released=references['released']['overall'][candidate]['score']<=references['released']['overall'][best]['score'])
        outcomes[grid]=dict(references=references,best_control=best,gain=gain,per_scene_gain=per_scene,checks=checks,advance=all(checks.values()))
    return outcomes


def score(assets):
    seal=read(PUBLIC/'predictions.json')
    with ProcessPoolExecutor(3) as pool:
        futures=[pool.submit(evaluate_scene,scene,assets) for scene in seal['batches']]
        observations=dict(f.result() for f in futures)
    dump(PUBLIC/'observations.json',observations)
    result=summarize(seal,observations)
    dump(PUBLIC/'scores.json',dict(seal_sha256=sha(PUBLIC/'predictions.json'),grids=result))
    print({g:(r['gain'],r['checks']) for g,r in result.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('predict','score'))
    p.add_argument('--assets',type=Path,required=True)
    args=p.parse_args()
    (predict if args.action=='predict' else score)(args.assets)
