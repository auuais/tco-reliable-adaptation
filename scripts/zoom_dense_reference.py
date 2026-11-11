"""Generate offline RGB-D evaluation geometry on exactly the cropped RGB raster."""
import numpy as np


def zoom_reference(depth, valid, intrinsic, world_to_camera, center, factor):
    """Lift nearest-sampled axial depth through the continuous ray of each output pixel."""
    depth=np.asarray(depth,dtype=np.float64)
    valid=np.asarray(valid,dtype=bool)
    k=np.asarray(intrinsic,dtype=np.float64)
    e=np.asarray(world_to_camera,dtype=np.float64)
    center=np.asarray(center,dtype=np.float64)
    if depth.ndim!=2 or valid.shape!=depth.shape or k.shape!=(3,3) or e.shape!=(4,4) or center.shape!=(2,):
        raise ValueError("Unexpected dense input shape")
    if (not np.isfinite(factor) or factor<=0 or not all(np.isfinite(a).all() for a in (k,e,center))
            or np.any(~np.isfinite(depth[valid])) or np.any(depth[valid]<=0)):
        raise ValueError("Invalid camera or valid depth sample")
    affine=np.eye(3)
    affine[[0,1],[0,1]]=factor
    affine[:2,2]=(1-factor)*center
    updated=affine@k
    h,w=depth.shape
    yy,xx=np.indices((h,w),dtype=np.float64)
    source_x=(xx-center[0])/factor+center[0]
    source_y=(yy-center[1])/factor+center[1]
    inside=(source_x>=0)&(source_x<=w-1)&(source_y>=0)&(source_y<=h-1)
    ix=np.clip(np.rint(source_x).astype(np.int64),0,w-1)
    iy=np.clip(np.rint(source_y).astype(np.int64),0,h-1)
    mask=inside & valid[iy,ix]
    z=np.where(mask,depth[iy,ix],0.)
    rays=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(updated).T
    c2w=np.linalg.inv(e)
    points=(rays*z[...,None])@c2w[:3,:3].T+c2w[:3,3]
    return points,mask,updated
