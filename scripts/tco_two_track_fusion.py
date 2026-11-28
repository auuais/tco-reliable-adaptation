"""Fuse homologous views of independently adapted TCO predictions without reference depth."""
import numpy as np


def centres(extrinsics):
    rotations = extrinsics[:, :3, :3]
    return -np.einsum("vij,vi->vj", rotations, extrinsics[:, :3, 3])


def fit_rig(extrinsics, prior):
    """Find a positive similarity using camera centres and axes alone, in metres."""
    pose = np.asarray(extrinsics, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[0] != prior.shape[0] or pose.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError("Unexpected camera rig")
    rot, ref = pose[:, :3, :3], prior[:, :3, :3]
    if not (np.isfinite(pose).all() and np.isfinite(prior).all()) or min(np.linalg.det(rot).min(),np.linalg.det(ref).min()) <= 0:
        return None
    source = centres(pose)
    target = centres(prior)
    i,j = np.triu_indices(len(source),k=1)
    a,b = np.linalg.norm(source[i]-source[j],axis=1),np.linalg.norm(target[i]-target[j],axis=1)
    valid = (a > 1e-6) & (b > 1e-6)
    if valid.sum() < 2:
        return None
    scale = float(np.median(b[valid] / a[valid]))
    u,_,vt = np.linalg.svd((ref.transpose(0,2,1) @ rot).sum(axis=0))
    u[:, -1] *= np.linalg.det(u @ vt)
    q = u @ vt
    shift = target.mean(axis=0) - scale * (source.mean(axis=0) @ q.T)
    aligned_centres = scale * (source @ q.T) + shift
    rms = float(np.sqrt(np.mean(np.sum((aligned_centres-target)**2,axis=1))))
    return dict(scale=scale, rotation=q, translation=shift, centres=aligned_centres,
                axes=rot @ q.T, centre_rms_m=rms, pairs=int(valid.sum()))


def unproject(depth, intrinsics, axes, centre):
    h,w = depth.shape
    y,x = np.ogrid[:h,:w]
    rays = np.stack(((x-intrinsics[0,2])*depth/intrinsics[0,0],
                     (y-intrinsics[1,2])*depth/intrinsics[1,1],depth),axis=-1)
    return rays @ axes + centre


def splat(world, intrinsics, axes, centre, shape):
    """Nearest visible z wins at nearest integer pixel, with ties to the lower source index."""
    h,w = shape
    camera = (world.reshape(-1,3) - centre) @ np.linalg.inv(axes)
    z = camera[:,2]
    eligible = np.flatnonzero(np.isfinite(camera).all(axis=1) & (z>0))
    x = intrinsics[0,0]*camera[eligible,0]/z[eligible]+intrinsics[0,2]
    y = intrinsics[1,1]*camera[eligible,1]/z[eligible]+intrinsics[1,2]
    inside = (x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    idx = eligible[inside]
    p = np.floor(y[inside]+.5).astype(np.int64)*w + np.floor(x[inside]+.5).astype(np.int64)
    order = np.lexsort((idx, z[idx], p))
    sorted_p = p[order]
    keep = np.r_[True, sorted_p[1:] != sorted_p[:-1]] if len(p) else np.zeros(0,bool)
    depth = np.full(h*w,np.nan,dtype=np.float64)
    depth[sorted_p[keep]] = z[idx[order[keep]]]
    return depth.reshape(h,w), int(len(p)-keep.sum())


def fuse(predictions, prior):
    """Return reference-free points and support receipts. Input order preassigns the first anchor."""
    if len(predictions)!=3:
        raise ValueError("Expected a registered triplet")
    rigs = [fit_rig(p["extrinsics"],prior) for p in predictions]
    anchor = predictions[0]
    shape = anchor["depth"].shape
    if len(shape)!=3 or not np.isfinite(anchor["depth"]).all() or np.any(anchor["depth"]<=0):
        raise ValueError("Anchor cannot supply full finite positive pixel support")
    for p in predictions:
        if p["depth"].shape!=shape or len(p["intrinsics"])!=shape[0] or (p["intrinsics"][:,(0,1),(0,1)]<=0).any():
            raise ValueError("Mismatched input or invalid intrinsics")
    ar = rigs[0]
    failed = [i for i,r in enumerate(rigs) if r is None]
    if ar is None:
        e = anchor["extrinsics"].astype(np.float64)
        ar = dict(axes=e[:, :3,:3], centres=centres(e),scale=1.)
    names = ("project0","project1","project2","fusion")
    outputs = {name:[] for name in names}
    support_rows = []
    for v in range(shape[0]):
        z0 = anchor["depth"][v].astype(np.float64)*ar["scale"]
        k0,r0,c0 = anchor["intrinsics"][v],ar["axes"][v],ar["centres"][v]
        maps,collisions = [],[]
        for p,r in zip(predictions,rigs):
            if r is None or rigs[0] is None:
                maps.append(np.full_like(z0,np.nan))
                collisions.append(0)
                continue
            depth = p["depth"][v].astype(np.float64)*r["scale"]
            depth[~np.isfinite(depth)|(depth<=0)] = np.nan
            world = unproject(depth,p["intrinsics"][v],r["axes"][v],r["centres"][v])
            projected,collision = splat(world,k0,r0,c0,z0.shape)
            maps.append(projected)
            collisions.append(collision)
        if rigs[0] is not None and not np.allclose(maps[0],z0,rtol=1.e-8,atol=1.e-9):
            raise ValueError("Anchor did not reproduce its dense output grid")
        maps[0] = z0
        stack = np.stack(maps)
        support = np.isfinite(stack)
        count = support.sum(axis=0)
        # A median in log depth uses a geometric midpoint for exactly two visible sources.
        fused = np.exp(np.nanmedian(np.log(stack),axis=0))
        fallback = (count<2) | bool(failed)
        fused[fallback] = z0[fallback]
        depth_outputs = [np.where(support[i],stack[i],z0) for i in range(3)]+[fused]
        for name,z in zip(names,depth_outputs):
            outputs[name].append(unproject(z,k0,r0,c0).astype(np.float32))
        supported = count>=2
        spread = np.nanmax(np.log(stack),axis=0) - np.nanmin(np.log(stack),axis=0)
        support_rows.append(dict(pixels=z0.size, sources=[int(i.sum()) for i in support],
                                 collisions=collisions, anchor_fill=[int((~b).sum()) for b in support],
                                 sources_per_pixel=np.bincount(count.ravel(),minlength=4).tolist(),
                                 fusion_anchor_fallback=int(fallback.sum()),
                                 median_supported_log_range=float(np.median(spread[supported])) if supported.any() else None))
    receipt = dict(failed_gauges=failed,views=support_rows,
                   rig_fits=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()
                              if k not in ("axes","centres")} if r is not None else None for r in rigs])
    return {k:np.stack(v) for k,v in outputs.items()},receipt
