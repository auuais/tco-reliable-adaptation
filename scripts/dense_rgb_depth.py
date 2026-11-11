"""Select per-pixel depth corrections with disjoint source-view evidence."""
import numpy as np
from scipy.ndimage import uniform_filter

from fixed_camera_depth_scale import centers


LOG_STEPS = np.array([0.,-.04,.04,-.08,.08])
FACTORS = np.exp(LOG_STEPS)
ARMS = ('two_source','four_source','guarded')


def bilinear(image,x,y):
    h,w = image.shape[:2]
    ok = np.isfinite(x)&np.isfinite(y)&(x>=0)&(x<w-1)&(y>=0)&(y<h-1)
    x,y = np.where(ok,x,0),np.where(ok,y,0)
    ix,iy = x.astype(int),y.astype(int)
    dx,dy = (x-ix)[...,None],(y-iy)[...,None]
    values = (1-dy)*((1-dx)*image[iy,ix]+dx*image[iy,ix+1])+dy*((1-dx)*image[iy+1,ix]+dx*image[iy+1,ix+1])
    return values,ok


def source_cost(target,source,depth,beam,offset,k):
    costs = []
    common = np.isfinite(depth)&(depth>0)
    for factor in FACTORS:
        xyz = beam*(depth*factor)[...,None]+offset
        z = xyz[...,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            x = k[0,0]*xyz[...,0]/z+k[0,2]
            y = k[1,1]*xyz[...,1]/z+k[1,2]
        sampled,ok = bilinear(source,x,y)
        observed = sampled[...,3]
        common &= ok & np.isfinite(z) & (z>0) & np.isfinite(observed) & (observed>0) & (z<=1.1*observed)
        costs.append(np.minimum(np.abs(sampled[...,:3]-target),.2).mean(axis=-1))
    density = uniform_filter(common.astype(float),size=5,mode='constant')
    valid = common & (density >= 15/25-1e-9)
    def patch(cost):
        return uniform_filter(np.where(common,cost,0.),size=5,mode='constant')/np.maximum(density,1/25)
    return np.stack([patch(cost) for cost in costs]),valid


def choose(costs,valid):
    train = (costs[0]+costs[2])/2
    test = (costs[1]+costs[3])/2
    seen = valid[0]&valid[2]
    checked = valid[1]&valid[3]
    proposed = np.where(seen,train.argmin(axis=0),0).astype(np.uint8)
    all_four = np.where(seen&checked,(train+test).argmin(axis=0),0).astype(np.uint8)
    passing = seen&checked
    for group in (train,test):
        chosen = np.take_along_axis(group,proposed[None],axis=0)[0]
        passing &= (group[0]>1e-6)&(chosen<=.95*group[0])
    guarded = np.where(passing,proposed,0).astype(np.uint8)
    return dict(two_source=proposed,four_source=all_four,guarded=guarded),dict(
        fit_pixels=int(seen.sum()),validation_pixels=int(checked.sum()),joint_pixels=int((seen&checked).sum()))


def estimate(images,depths,k,poses):
    if images.shape!=(len(depths),3,*depths.shape[1:]) or len(depths)!=5 or not np.isfinite(images).all():
        raise ValueError('Expected five finite channel-first RGB views matching axial depth')
    rgb = images.transpose(0,2,3,1)
    desc = rgb-uniform_filter(rgb,size=(1,9,9,1),mode='reflect')
    sources = np.concatenate((desc,depths[...,None].astype(np.float32)),axis=-1)
    c = centers(poses)
    y,x = np.indices(depths.shape[1:],dtype=float)
    maps = {a:[] for a in ARMS}
    observations = []
    for i in range(5):
        neighbors = sorted((j for j in range(5) if j!=i),key=lambda j:(np.linalg.norm(c[j]-c[i]),j))
        rays = np.stack(((x-k[i,0,2])/k[i,0,0],(y-k[i,1,2])/k[i,1,1],np.ones_like(x)),axis=-1) @ poses[i,:3,:3]
        costs,valid = [],[]
        for j in neighbors:
            rot = poses[j,:3,:3].T
            a,b = source_cost(desc[i],sources[j],depths[i],rays@rot,c[i]@rot+poses[j,:3,3],k[j])
            costs.append(a)
            valid.append(b)
        decisions,support = choose(costs,valid)
        for arm,indices in decisions.items():
            maps[arm].append(indices)
        observations.append(dict(view=i,source_order=neighbors,**support,histograms={
            arm:np.bincount(indices.ravel(),minlength=len(FACTORS)).tolist() for arm,indices in decisions.items()}))
    return {a:np.stack(v) for a,v in maps.items()},observations
