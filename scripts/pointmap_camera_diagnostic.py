"""Recover a pinhole camera from its own dense pointmap/pixel correspondences.

Offline diagnostic only. Fits no reference camera or depth and rejects ambiguous
or poorly reprojecting solutions. This does not improve any reconstruction.
"""
import numpy as np


def fit_camera(points, pixels):
    points=np.asarray(points,dtype=np.float64);pixels=np.asarray(pixels,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or pixels.shape!=(len(points),2) or len(points)<12:
        raise ValueError("Insufficient camera correspondences")
    if not np.isfinite(points).all() or not np.isfinite(pixels).all():raise ValueError("Nonfinite correspondences")
    center=points.mean(axis=0);image_center=pixels.mean(axis=0)
    size=np.mean(np.linalg.norm(points-center,axis=1));image_size=np.mean(np.linalg.norm(pixels-image_center,axis=1))
    if min(size,image_size)<1e-12:raise ValueError("Collapsed correspondences")
    transform3=np.eye(4);transform3[:3,:3]*=np.sqrt(3)/size;transform3[:3,3]=-center*np.sqrt(3)/size
    transform2=np.eye(3);transform2[:2,:2]*=np.sqrt(2)/image_size;transform2[:2,2]=-image_center*np.sqrt(2)/image_size
    xyz=np.c_[points,np.ones(len(points))]@transform3.T
    uv=(np.c_[pixels,np.ones(len(pixels))]@transform2.T)[:,:2]
    a=np.zeros((2*len(points),12));a[::2,:4]=xyz;a[1::2,4:8]=xyz
    a[::2,8:]=-uv[:,0,None]*xyz;a[1::2,8:]=-uv[:,1,None]*xyz
    _,singular,vh=np.linalg.svd(a,full_matrices=False)
    gap=float(singular[-2]/max(singular[-1],np.finfo(float).tiny))
    if not np.isfinite(gap) or gap<1000:raise ValueError(f"Ambiguous pinhole camera: null-space gap {gap}")
    projection=np.linalg.solve(transform2,vh[-1].reshape(3,4)@transform3)
    if np.linalg.det(projection[:,:3])<0:projection=-projection
    q,r=np.linalg.qr(np.flipud(projection[:,:3]).T)
    intrinsic=np.fliplr(np.flipud(r.T));rotation=np.flipud(q.T)
    diagonal=np.sign(np.diag(intrinsic))
    if np.any(diagonal==0):raise ValueError("Singular camera matrix")
    intrinsic=intrinsic@np.diag(diagonal);rotation=np.diag(diagonal)@rotation
    translation=np.linalg.solve(intrinsic,projection[:,3]);intrinsic=intrinsic/intrinsic[2,2]
    if np.linalg.det(rotation)<.999999 or np.max(np.abs(rotation@rotation.T-np.eye(3)))>1e-8:
        raise ValueError("Invalid proper camera rotation")
    extrinsic=np.eye(4);extrinsic[:3,:3]=rotation;extrinsic[:3,3]=translation
    projected=points@rotation.T+translation;projected=projected@intrinsic.T
    if np.any(projected[:,2]<=0):raise ValueError("Nonpositive fitted depth")
    residual=np.linalg.norm(projected[:,:2]/projected[:,2,None]-pixels,axis=1)
    return dict(intrinsics=intrinsic,extrinsics=extrinsic,null_space_gap=gap,fit_pixel_rmse=float(np.sqrt(np.mean(residual**2))))


def recover(pointmap):
    pointmap=np.asarray(pointmap);height,width,_=pointmap.shape
    ys=np.unique(np.linspace(0,height-1,16).astype(int));xs=np.unique(np.linspace(0,width-1,16).astype(int))
    xx,yy=np.meshgrid(xs,ys);pixels=np.c_[xx.ravel(),yy.ravel()]
    result=fit_camera(pointmap[yy,xx].reshape(-1,3),pixels)
    # Validate all pixels, not only the DLT fit support.
    world=pointmap.astype(np.float64);rotation=result["extrinsics"][:3,:3];translation=result["extrinsics"][:3,3]
    camera=world@rotation.T+translation;projected=camera@result["intrinsics"].T
    if not np.isfinite(projected).all() or np.any(camera[...,2]<=0):raise ValueError("Invalid dense camera geometry")
    x,y=np.meshgrid(np.arange(width),np.arange(height));uv=projected[...,:2]/projected[...,2,None]
    residual=np.linalg.norm(uv-np.stack([x,y],axis=-1),axis=-1)
    result.update(dense_reprojection_max_pixels=float(residual.max()),dense_reprojection_rmse_pixels=float(np.sqrt(np.mean(residual**2))))
    if result["dense_reprojection_max_pixels"]>.02:raise ValueError("Dense pinhole reprojection check failed")
    return result,camera[...,2]
