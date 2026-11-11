"""Factorial RGB/depth costs on common footprints with disjoint source witnesses."""
import numpy as np
from scipy.ndimage import uniform_filter

from dense_rgb_depth import FACTORS, bilinear, choose
from fixed_camera_depth_scale import centers


MASKS = ('all_candidates','starting_depth')
CUES = ('rgb','depth')
RULES = ('single','two_witness','four_view')
PRIMARY = ('starting_depth','depth','two_witness')
MODEL_ARMS = dict(guarded=PRIMARY,single_depth=('starting_depth','depth','single'),
    checked_rgb=('starting_depth','rgb','two_witness'),four_view_depth=('starting_depth','depth','four_view'),
    all_candidate_depth=('all_candidates','depth','two_witness'))


def pair_costs(target,source,depth,beam,offset,k):
    footprint = np.isfinite(depth)&(depth>0)
    seen,rgb_cost,depth_cost = [],[],[]
    for factor in FACTORS:
        xyz = beam*(depth*factor)[...,None]+offset
        z = xyz[...,2]
        with np.errstate(divide='ignore',invalid='ignore'):
            x = k[0,0]*xyz[...,0]/z+k[0,2]
            y = k[1,1]*xyz[...,1]/z+k[1,2]
        sampled,ok = bilinear(source,x,y)
        measured = sampled[...,3]
        ok &= np.isfinite(z)&(z>0)&np.isfinite(measured)&(measured>0)
        footprint &= ok
        seen.append(z<=1.1*measured)
        rgb_cost.append(np.minimum(np.abs(sampled[...,:3]-target),.2).mean(axis=-1))
        with np.errstate(divide='ignore',invalid='ignore'):
            discrepancy = np.abs(np.log(z/measured))/np.log(1.1)
        depth_cost.append(.2*np.minimum(discrepancy,1.))
    visible = np.stack(seen)
    costs,support = {},{}
    for mode in MASKS:
        pixels = footprint & (np.all(visible,axis=0) if mode=='all_candidates' else visible[0])
        density = uniform_filter(pixels.astype(float),size=5,mode='constant')
        support[mode] = pixels&(density>=15/25-1e-9)
        for cue,raw in zip(CUES,(rgb_cost,depth_cost)):
            patches = []
            for vis,loss in zip(visible,raw):
                values = np.where(vis,loss,.2) if mode=='starting_depth' else loss
                patches.append(uniform_filter(np.where(pixels,values,0.),size=5,mode='constant')/np.maximum(density,1/25))
            costs[mode,cue] = np.stack(patches)
    return costs,support


def screen(costs,eligible):
    """Ranks were fixed using geometry. Gather the first two supported witnesses."""
    masks = np.stack(eligible)
    seen = masks.cumsum(axis=0)
    n = seen[-1]
    first = np.zeros_like(costs[0])
    second = np.zeros_like(first)
    for rank,loss in enumerate(costs):
        first = np.where(masks[rank]&(seen[rank]==1),loss,first)
        second = np.where(masks[rank]&(seen[rank]==2),loss,second)
    single = np.where(n>=1,first.argmin(axis=0),0).astype(np.uint8)
    take = n>=2
    for group in (first,second):
        chosen = np.take_along_axis(group,single[None],axis=0)[0]
        take &= (group[0]>1e-6)&(chosen<=.95*group[0])
    predictions, _ = choose(costs,eligible)
    return dict(single=(single,n>=1),two_witness=(np.where(take,single,0).astype(np.uint8),n>=2),
                four_view=(predictions['guarded'],n==4))


def estimate_views(images,depths,k,poses,offsets):
    """Yield a target and condition at a time to bound peak memory."""
    rgb = images.transpose(0,2,3,1)
    desc = rgb-uniform_filter(rgb,size=(1,9,9,1),mode='reflect')
    sources = np.concatenate((desc,depths[...,None].astype(np.float32)),axis=-1)
    c = centers(poses)
    y,x = np.indices(depths.shape[1:],dtype=float)
    for i in range(5):
        neighbors = sorted((j for j in range(5) if j!=i),key=lambda j:(np.linalg.norm(c[j]-c[i]),j))
        rays = np.stack(((x-k[i,0,2])/k[i,0,0],(y-k[i,1,2])/k[i,1,1],np.ones_like(x)),axis=-1)@poses[i,:3,:3]
        for state,offset in offsets.items():
            measured = []
            for j in neighbors:
                rot = poses[j,:3,:3].T
                measured.append(pair_costs(desc[i],sources[j],depths[i]*np.exp(offset),rays@rot,c[i]@rot+poses[j,:3,3],k[j]))
            outputs = {}
            for mode in MASKS:
                eligible = [obs[1][mode] for obs in measured]
                for cue in CUES:
                    costs = [obs[0][mode,cue] for obs in measured]
                    outputs.update({(mode,cue,rule):out for rule,out in screen(costs,eligible).items()})
            yield i,neighbors,state,outputs
