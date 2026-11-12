"""Fresh float32 inference, matching and alignment for MASt3R neural-transfer studies."""
import time
import numpy as np
import torch
import torch.nn.functional as F
import mast3r.cloud_opt.sparse_ga as alignment
from dust3r.image_pairs import make_pairs
from evaluation_stability import dump,sha
from prepare_dense_focal_pilot import ROOT,INPUT,write_arrays
from run_depth_scale_study import PLAN


def views_for(key):
    from score_saved_geometry import read
    meta=read(PLAN)['inputs'][key]
    path=INPUT/meta['file']
    assert sha(path)==meta['sha256']
    with np.load(path) as z: rgb=torch.from_numpy(z['images'])
    rgb=2*F.interpolate(rgb,size=(384,512),mode='bilinear',align_corners=False,antialias=True)-1
    names=[f'{key}_{meta["sha256"][:12]}_{i}' for i in range(len(rgb))]
    views=[dict(img=im[None],true_shape=np.array([[384,512]],dtype=np.int32),idx=i,instance=names[i]) for i,im in enumerate(rgb)]
    return meta,names,views


def capture(model,key,cache,raw,public,tag):
    meta,names,views=views_for(key)
    pairs=make_pairs(views,scene_graph='complete',prefilter=None,symmetrize=True)
    prior_files=sum(p.is_file() for p in cache.rglob('*'))
    cache.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);np.random.seed(42)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter()
    result=alignment.sparse_global_alignment(names,pairs,str(cache),model,device='cuda',
        lr1=.07,niter1=300,lr2=.01,niter2=300,opt_depth=True,shared_intrinsics=False,matching_conf_thr=0.)
    points,depths,_=result.get_dense_pts3d(clean_depth=False)
    depth=torch.stack([z.reshape(384,512) for z in depths])
    c2w=result.cam2w.detach();k=result.intrinsics.detach()
    for i,p in enumerate(points):
        z=(p.reshape(384,512,3)-c2w[i,:3,3])@c2w[i,:3,2]
        torch.testing.assert_close(z,depth[i],rtol=2e-4,atol=2e-5)
    assert torch.isfinite(depth).all() and torch.isfinite(c2w).all() and torch.isfinite(k).all()
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    raw.mkdir(parents=True,exist_ok=True)
    path=raw/(tag+'.npz')
    write_arrays(path,dict(depth=depth.detach().cpu().numpy(),intrinsics=k.cpu().numpy(),c2w=c2w.cpu().numpy()))
    metadata=dict(input_sha256=meta['sha256'],output_sha256=sha(path),pairs=len(pairs),
        prior_cache_files=prior_files,seconds=seconds,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        nonpositive_depth_pixels=(depth<=0).sum(dim=(1,2)).tolist(),native_size=[384,512],
        device=torch.cuda.get_device_name(),inference_precision='float32',coarse_iterations=300,fine_iterations=300)
    dump(public/(tag+'.json'),metadata)
    return metadata


def training_features(model,views):
    result=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for view in views:
            shape=torch.as_tensor(view['true_shape'],device='cuda')
            feat,pos,_=model._encode_image(view['img'].cuda(),shape)
            result.append((feat.detach(),pos.detach(),shape))
    return result


def training_pair(model,features,i,j):
    f1,p1,shape1=features[i];f2,p2,shape2=features[j]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        d1,d2=model._decoder(f1,p1,f2,p2)
    with torch.autocast('cuda',enabled=False):
        first=model._downstream_head(1,[t.float() for t in d1],shape1)
        other=model._downstream_head(2,[t.float() for t in d2],shape2)
    other['pts3d_in_other_view']=other.pop('pts3d')
    return first,other
