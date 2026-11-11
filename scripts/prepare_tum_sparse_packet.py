"""Prepare selected TUM RGB images and sparse neighboring depth without opening target depth."""
import argparse
import io
import hashlib
from pathlib import Path
import tarfile

import cv2
import numpy as np
from PIL import Image

from collect_matched_focal_heads import tensor_digest
from evaluation_stability import dump,sha
from focal_zoom_observations import SCALE
from neighbor_sparse_depth import project_samples
from prepare_dense_focal_pilot import ROOT,write_arrays
from score_saved_geometry import read
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES

RAW=ROOT/'reports/tum_sparse_depth'
GRIDS={'centered':(40,30),'shifted':(20,15)}


def camera():
    original=np.array([[535.4,0,320.1],[0,539.2,247.6],[0,0,1.]])
    resize=np.diag([518/640,392/480,1.])
    resize[:2,2]=(resize[[0,1],[0,1]]-1)/2
    intrinsic=resize@original
    zoom=np.eye(3)
    zoom[[0,1],[0,1]]=SCALE
    zoom[:2,2]=(1-SCALE)*intrinsic[:2,2]
    return original,zoom@intrinsic,zoom[:2]


def prepare(destination):
    selection=read(PUBLIC/'inference_selection.json')
    native,target,zoom=camera()
    arrays={}
    records={}
    for sequence in SEQUENCES:
        entry=selection[sequence]
        folder='rgbd_dataset_freiburg3_'+sequence
        path=destination/(folder+'.tgz')
        download=read(PUBLIC/(sequence+'_download.json'))
        if sha(path)!=download['archive_sha256']:
            raise ValueError('Archive changed')
        input_files={}
        allowed={t['rgb_file'] for t in entry['targets']}
        allowed.update(s['depth_file'] for t in entry['targets'] for s in t['sources'])
        expected={folder+'/'+name:name for name in allowed}
        selected={}
        with tarfile.open(path,'r|gz') as archive:
            for member in archive:
                if member.name in expected:
                    if not member.isfile():
                        raise ValueError('Expected an ordinary image file')
                    name=expected[member.name]
                    with archive.extractfile(member) as stream:
                        selected[name]=stream.read()
                    input_files[name]=hashlib.sha256(selected[name]).hexdigest()
        if set(selected)!=allowed:
            raise ValueError('Inference images missing from archive')
        rgb=[]
        poses=[]
        views=[]
        for i,t in enumerate(entry['targets']):
            with Image.open(io.BytesIO(selected[t['rgb_file']])) as image:
                if image.size!=(640,480):
                    raise ValueError('Unexpected RGB size')
                resized=image.convert('RGB').resize((518,392),Image.Resampling.LANCZOS)
                pixels=np.asarray(resized,dtype=np.float32)/255
            rgb.append(cv2.warpAffine(pixels,zoom,(518,392),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).transpose(2,0,1))
            world_to_target=np.linalg.inv(np.asarray(t['c2w']))
            poses.append(world_to_target)
            counts={}
            for grid,origin in GRIDS.items():
                u,v=np.meshgrid(np.arange(origin[0],640,80),np.arange(origin[1],480,60))
                uv=np.column_stack([u.ravel(),v.ravel()])
                rays=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(native).T
                measurements=[]
                for source in t['sources']:
                    with Image.open(io.BytesIO(selected[source['depth_file']])) as image:
                        if image.size!=(640,480):
                            raise ValueError('Unexpected registered depth size')
                        # project_samples consumes millimeters, TUM stores 5000 units/m.
                        mm=np.asarray(image)[uv[:,1],uv[:,0]].astype(float)/5
                    relative=world_to_target@np.asarray(source['c2w'])
                    measurements.extend(project_samples(mm,uv,rays,relative,target,source['depth_index']))
                nearest={}
                for row in sorted(measurements,key=lambda r:r[2]):
                    nearest.setdefault(tuple(row[:2]),row)
                packet=np.array(list(nearest.values()))
                if len(packet)<8:
                    raise ValueError('Insufficient source-depth coverage')
                arrays[f'{sequence}_{grid}_{i}']=packet
                counts[grid]=len(packet)
            views.append(dict(rgb_time=t['rgb_time'],counts=counts))
        rgb=np.stack(rgb)
        folder=RAW/'inference'
        folder.mkdir(parents=True,exist_ok=True)
        write_arrays(folder/(sequence+'.npz'),dict(images=rgb))
        arrays[sequence+'_poses']=np.stack(poses)
        arrays[sequence+'_intrinsics']=np.repeat(target[None],len(rgb),axis=0)
        records[sequence]=dict(rgb_file=sequence+'.npz',rgb_sha256=sha(folder/(sequence+'.npz')),
                               rgb_tensor_sha256=tensor_digest(rgb),files=input_files,views=views)
        print(sequence,'RGB and two sparse grids prepared',flush=True)
    write_arrays(PUBLIC/'measurements.npz',arrays)
    record=dict(selection_sha256=sha(PUBLIC/'inference_selection.json'),packet_sha256=sha(PUBLIC/'measurements.npz'),sequences=records)
    path=PUBLIC/'inputs.json'
    if path.exists() and read(path)!=record:
        raise ValueError('Sealed inference packet changed')
    dump(path,record)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    prepare(p.parse_args().destination)
