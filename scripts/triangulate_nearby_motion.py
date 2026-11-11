"""Estimate axial depths from RGB flow and calibrated cameras in the same gauge."""
import warnings

import numpy as np

from camera_depth_swap import Rig
from dense_rgb_depth import bilinear
from fixed_camera_depth_scale import centers
from track_camera_selection import bearings


def triangulate(target, source, forward, reverse, k, e):
    height, width = forward.shape[:2]
    y, x = np.indices((height,width), dtype=float)
    xy = np.column_stack((x.ravel(),y.ravel()))
    match = xy+forward.reshape(-1,2)
    back, inside = bilinear(reverse, match[:,0], match[:,1])
    cycle_error = np.linalg.norm(match+back-xy, axis=1)
    rig = Rig(rotation=e[:,:3,:3], center=centers(e), intrinsic=k)
    ray = bearings(rig, target, xy)
    line = bearings(rig, source, match)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    normal = ray-line*np.sum(ray*line, axis=1, keepdims=True)
    square = np.sum(normal**2, axis=1)
    baseline = rig.center[source]-rig.center[target]
    depth = (normal@baseline)/np.maximum(square,1e-15)
    source_xyz = (depth[:,None]*ray-baseline)@rig.rotation[source].T
    image = source_xyz@k[source].T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_error = np.linalg.norm(image[:,:2]/image[:,2,None]-match, axis=1)
    valid = (inside & (cycle_error<=1) & (depth>0) & (source_xyz[:,2]>0) & (pixel_error<=1)
             & (square>=np.sum(ray**2,axis=1)*np.sin(np.deg2rad(1.))**2))
    return np.where(valid,depth,np.nan).reshape(height,width)


def combine(estimates, offsets):
    logs = np.log(np.stack(estimates))
    # All-missing pixels represent an explicit abstention.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        pooled = np.nanmedian(logs, axis=0)
        before = np.nanmedian(logs[np.asarray(offsets)<0], axis=0)
        after = np.nanmedian(logs[np.asarray(offsets)>0], axis=0)
    both = np.isfinite(before) & np.isfinite(after) & (np.abs(before-after)<=.08)
    return dict(any_source=np.exp(pooled), two_sided=np.where(both,np.exp(pooled),np.nan))


def synthetic_test():
    h,w = 40,80
    k = np.tile([[60.,0,40],[0,60.,20],[0,0,1]],(5,1,1))
    e = np.tile(np.eye(4),(5,1,1))
    e[:,0,3] = np.array([.20,.10,0,-.10,-.20])
    plane = 2.
    samples = []
    for j in (0,1,3,4):
        forward = np.zeros((h,w,2),dtype=np.float32)
        forward[...,0] = 60*e[j,0,3]/plane
        depth = triangulate(2,j,forward,-forward,k,e)
        np.testing.assert_allclose(depth[3:-3,8:-8],plane,rtol=1e-9)
        if np.isfinite(triangulate(2,j,forward,-forward+3,k,e)).any():
            raise ValueError('Inconsistent return flow passed')
        if np.isfinite(triangulate(2,j,-forward,forward,k,e)).any():
            raise ValueError('Negative triangulated depth passed')
        scaled = e.copy()
        scaled[:,:3,3] *= 4
        np.testing.assert_allclose(triangulate(2,j,forward,-forward,k,scaled),4*depth)
        samples.append(depth)
    both = combine(samples, [-12,-4,4,12])
    np.testing.assert_allclose(both['two_sided'][3:-3,8:-8],plane)
    for i in (2,3):
        samples[i] = samples[i]*1.2
    if np.isfinite(combine(samples,[-12,-4,4,12])['two_sided']).any():
        raise ValueError('Disagreeing directions passed the two-sided check')
    pure_rotation = np.tile(np.eye(4),(5,1,1))
    zeros = np.zeros((h,w,2),dtype=np.float32)
    if np.isfinite(triangulate(2,3,zeros,zeros,k,pure_rotation)).any():
        raise ValueError('Pure rotation provided a triangulation depth')


if __name__=='__main__':
    synthetic_test()
    print('Verified known-plane depth, gauge, cheirality, cycle rejection and two-sided disagreement.')
