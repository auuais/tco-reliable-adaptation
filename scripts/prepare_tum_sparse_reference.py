"""Project timestamped registered depth into the target RGB camera with a nearest-depth buffer."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from evaluation_stability import dump,sha
from fixed_camera_depth_scale import point_map
from prepare_dense_focal_pilot import write_arrays
from prepare_tum_sparse_depth import PUBLIC,SEQUENCES
from prepare_tum_sparse_packet import RAW,camera
from score_saved_geometry import read


def compensate(depth,intrinsic,target_intrinsic,depth_to_rgb,shape=(392,518)):
    if depth.ndim!=2 or not np.isfinite(depth).all() or np.any(depth<0):
        raise ValueError('Expected finite nonnegative axial depth in meters')
    yy,xx=np.indices(depth.shape,dtype=float)
    pixels=np.column_stack([xx.ravel(),yy.ravel(),np.ones(depth.size)])
    z=depth.ravel()
    valid=(z>0)&(z<=10)
    xyz=(pixels[valid]@np.linalg.inv(intrinsic).T)*z[valid,None]
    xyz=xyz@depth_to_rgb[:3,:3].T+depth_to_rgb[:3,3]
    xyz=xyz[np.isfinite(xyz).all(axis=1)&(xyz[:,2]>0)]
    homogeneous=xyz@target_intrinsic.T
    uv=np.rint(homogeneous[:,:2]/homogeneous[:,2:]).astype(int)
    height,width=shape
    inside=((uv>=0)&(uv<[width,height])).all(axis=1)
    uv=uv[inside]
    output=np.full(height*width,np.inf)
    np.minimum.at(output,uv[:,1]*width+uv[:,0],xyz[inside,2])
    output=output.reshape(shape)
    valid=np.isfinite(output)
    return np.where(valid,output,0.),valid


def prepare(destination):
    sealed=PUBLIC/'predictions.json'
    if not sealed.exists():
        raise FileNotFoundError('Seal TUM predictions before preparing target depth references')
    targets=read(PUBLIC/'inference_selection.json')
    labels=read(PUBLIC/'evaluation_selection.json')
    native,intrinsic,_=camera()
    records={}
    for sequence in SEQUENCES:
        prefix='rgbd_dataset_freiburg3_'+sequence
        archive_path=destination/(prefix+'.tgz')
        if sha(archive_path)!=read(PUBLIC/(sequence+'_download.json'))['archive_sha256']:
            raise ValueError('Original archive changed')
        allowed={prefix+'/'+r['depth_file']:i for i,r in enumerate(labels[sequence])}
        loaded={}
        with tarfile.open(archive_path,'r|gz') as archive:
            for member in archive:
                if member.name in allowed:
                    with archive.extractfile(member) as stream:
                        content=stream.read()
                    with Image.open(io.BytesIO(content)) as image:
                        if image.size!=(640,480):
                            raise ValueError('Unexpected depth image size')
                        loaded[allowed[member.name]]=(np.asarray(image).astype(float)/5000,hashlib.sha256(content).hexdigest())
        if set(loaded)!=set(range(5)):
            raise ValueError('Missing target depth image')
        depths,masks,world,sources=[],[],[],{}
        for i,(target,label) in enumerate(zip(targets[sequence]['targets'],labels[sequence])):
            assert target['association_index']==label['association_index']
            raw,digest=loaded[i]
            rgb_camera=np.linalg.inv(np.asarray(target['c2w']))
            relative=rgb_camera@np.asarray(label['c2w'])
            depth,mask=compensate(raw,native,intrinsic,relative)
            if mask.sum()==0:
                raise ValueError('No visible evaluation depth')
            points=point_map(depth[None],intrinsic[None],rgb_camera[None,:3],np.ones(1))[0]
            depths.append(depth)
            masks.append(mask)
            world.append(points)
            sources[label['depth_file']]=digest
        output=RAW/'references'/(sequence+'.npz')
        output.parent.mkdir(parents=True,exist_ok=True)
        write_arrays(output,dict(depth=np.stack(depths),valid_mask=np.stack(masks),points=np.stack(world)))
        records[sequence]=dict(file=output.name,sha256=sha(output),target_depth_files=sources,valid_pixels=[int(m.sum()) for m in masks])
    dump(PUBLIC/'references.json',dict(prediction_seal_sha256=sha(sealed),evaluation_selection_sha256=sha(PUBLIC/'evaluation_selection.json'),sequences=records))
    print('Prepared motion-compensated labels after sealing all TUM predictions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    prepare(p.parse_args().destination)
