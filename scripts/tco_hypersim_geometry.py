"""Camera and surface conventions for the registered Hypersim replication."""
import numpy as np

GL_TO_CV = np.diag([1., -1., -1.])


class BridgeSupportError(ValueError):
    """The locked training frame cannot validate independent released positions."""


def camera_from_metadata(row):
    w,h=(int(float(row["settings_output_img_"+d])) for d in ("width","height"))
    if min(w,h)<14:
        raise ValueError("Invalid source raster")
    rays=np.array([[float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)] for i in range(3)])
    proj=np.array([[float(row[f"M_proj_{i}{j}"]) for j in range(4)] for i in range(4)])
    pixel_to_uv=np.array([[2/w,0,1/w-1],[0,-2/h,1-1/h],[0,0,1.]])
    native_to_ray=GL_TO_CV@rays@pixel_to_uv
    ray_to_native=np.linalg.inv(native_to_ray)
    ray_to_native/=ray_to_native[2,2]
    # K for the rectified pinhole matches the central source coordinate and derivative.
    centre=ray_to_native[:2,2]
    derivative=ray_to_native[:2,:2]-np.outer(centre,ray_to_native[2,:2])
    if not np.isfinite(derivative).all() or (np.diag(derivative)<=0).any():
        raise ValueError("Unusable central focal lengths")
    width=518
    height=14*int(round(width*h/w/14))
    affine=np.array([[width/w,0,(width/w-1)/2],[0,height/h,(height/h-1)/2],[0,0,1.]])
    K=affine@np.array([[derivative[0,0],0,centre[0]],[0,derivative[1,1],centre[1]],[0,0,1.]])
    y,x=np.indices((height,width))
    grid=np.stack((x,y,np.ones_like(x)),axis=-1)
    # The mapping is projective. Checking its four corners and sign of the
    # denominator bounds the mapped rectangle, including nonstandard tilted cameras.
    corners=grid[[0,0,-1,-1],[0,-1,0,-1]]
    for n in range(71):
        kc=K.copy()
        zoom=1.01**n
        kc[0,0]*=zoom
        kc[1,1]*=zoom
        hom=ray_to_native@np.linalg.inv(kc)
        q=corners@hom.T
        xy=q[:,:2]/q[:,2:]
        if np.all(q[:,2]>0) and np.all(xy >= 0) and np.all(xy <= [w-1,h-1]):
            break
    else:
        raise ValueError("No bounded rectified raster")
    xyz=grid@hom.T
    xy=xyz[...,:2]/xyz[...,2:]
    if not np.isfinite(xy).all():
        raise ValueError("Invalid native ray map")
    # Independent published 4x4 projection should invert the ray matrix at native centres.
    sentinel=np.array([[0.,0.,1.],[w-1,0,1.],[0,h-1,1.],[w-1,h-1,1.],[w/2,h/2,1.]])
    native_rays=sentinel@native_to_ray.T
    view_gl=native_rays@GL_TO_CV
    clip=np.column_stack((view_gl,np.ones(len(sentinel))))@proj.T
    uv=clip[:,:2]/clip[:,3:]
    px=np.column_stack(((uv[:,0]+1)*w/2-.5,(1-uv[:,1])*h/2-.5))
    if not np.isfinite(px).all() or np.abs(px-sentinel[:,:2]).max()>.002:
        raise ValueError("Publisher projection and ray matrices disagree")
    return dict(size=(w,h),raster=(width,height),K=kc,native_to_ray=native_to_ray,ray_to_native=ray_to_native,
                xy=xy,zoom=zoom,projection_check_max_px=float(np.abs(px-sentinel[:,:2]).max()))


def world_to_camera(gl_orientation, centre_asset, scale):
    rotation=np.asarray(gl_orientation,dtype=float)
    centre=np.asarray(centre_asset,dtype=float)*scale
    if (rotation.shape!=(3,3) or centre.shape!=(3,) or not np.isfinite(centre).all()
            or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=2e-5)
            or abs(np.linalg.det(rotation)-1)>2e-5 or not np.isfinite(scale) or scale<=0):
        raise ValueError("Nonrigid or unscaled source camera")
    w2c=GL_TO_CV@rotation.T
    return np.column_stack((w2c,-w2c@centre))


def convert_frame(rgb, radial, cam, extr, position_asset=None, asset_scale=None):
    w,h=cam["size"]
    if rgb.shape!=(h,w,3) or rgb.dtype!=np.uint8 or radial.shape!=(h,w):
        raise ValueError("Source modality raster mismatch")
    xy=cam["xy"]
    ix=np.floor(xy[...,0]).astype(int)
    iy=np.floor(xy[...,1]).astype(int)
    dx=(xy[...,0]-ix)[...,None]
    dy=(xy[...,1]-iy)[...,None]
    xp=np.minimum(ix+1,w-1)
    yp=np.minimum(iy+1,h-1)
    color=((1-dy)*((1-dx)*rgb[iy,ix]+dx*rgb[iy,xp])+dy*((1-dx)*rgb[yp,ix]+dx*rgb[yp,xp]))/255.
    u=np.floor(xy[...,0]+.5).astype(int)
    v=np.floor(xy[...,1]+.5).astype(int)
    source=np.stack((u,v,np.ones_like(u)),axis=-1)
    direction=source@cam["native_to_ray"].T
    direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
    dist=radial[v,u].astype(float)
    z=dist*direction[...,2]
    mask=np.isfinite(dist)&(direction[...,2]>0)&(z>=.2)&(z<=10.)
    if int(mask.sum())==0:
        if position_asset is not None:
            raise BridgeSupportError("No evaluable geometry in fixed training frame")
        raise ValueError("No evaluable geometry in fixed frame")
    view=direction[mask]*dist[mask,None]
    surface=np.zeros(z.shape+(3,),np.float32)
    surface[mask]=(view-extr[:,3])@extr[:,:3]
    evidence=dict(valid=int(mask.sum()),valid_fraction=float(mask.mean()),
                  max_native_sample_offset_px=float(np.max(np.abs(source[...,:2]-xy))))
    if position_asset is not None:
        if position_asset.shape!=(h,w,3) or not asset_scale:
            raise ValueError("Invalid training bridge position annotation")
        pos=position_asset[v[mask],u[mask]]
        finite=np.isfinite(pos).all(axis=1)
        evidence["bridge_support_fraction"]=float(finite.mean())
        if finite.mean()<.99 or finite.sum()<1000:
            raise BridgeSupportError("Insufficient released training bridge position coverage")
        reference=pos[finite].astype(float)*asset_scale
        world=surface[mask][finite].astype(float)
        delta=np.linalg.norm(reference-world,axis=-1)
        # The publisher derives range from positions and then independently rounds
        # each raster to FP16. Allow a full last-place unit in each stored quantity.
        raw_dist=radial[v[mask],u[mask]][finite]
        quant=asset_scale*np.linalg.norm(np.spacing(np.abs(pos[finite])).astype(float),axis=-1)
        quant+=np.spacing(np.abs(raw_dist)).astype(float)
        error=np.maximum(delta-quant,0.)
        projected=(reference@extr[:,:3].T+extr[:,3])@cam["ray_to_native"].T
        native_xy=projected[:,:2]/projected[:,2:]
        native_ref=source[...,:2][mask][finite]
        px=np.linalg.norm(native_xy-native_ref,axis=-1)
        evidence.update(median_position_error_m=float(np.median(delta)),p95_position_error_m=float(np.quantile(delta,.95)),
                        corrected_median_m=float(np.median(error)),corrected_p95_m=float(np.quantile(error,.95)),
                        median_native_reprojection_px=float(np.median(px)),p95_native_reprojection_px=float(np.quantile(px,.95)))
        evidence["bridge_passed"]=bool(evidence["corrected_median_m"]<=.002 and evidence["corrected_p95_m"]<=.01
            and evidence["median_native_reprojection_px"]<=.5 and evidence["p95_native_reprojection_px"]<=2.)
    return color.transpose(2,0,1).astype(np.float32),surface,mask,evidence
