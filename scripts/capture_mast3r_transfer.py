"""Run ordinary MASt3R inference and sparse global alignment on the sealed RGB inputs."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.image_pairs import make_pairs
from prepare_dense_focal_pilot import INPUT, ROOT, write_arrays
from run_depth_scale_study import PLAN
from score_saved_geometry import read
from evaluation_stability import dump, sha

SOURCE = 'f5209afc300cec36239a7ac992263f36847bbba0'
WEIGHTS = '0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb'
RAW = ROOT/'reports/mast3r_transfer'
PUBLIC = ROOT/'results/mast3r_transfer'


def capture(model, key):
    meta = read(PLAN)['inputs'][key]
    path = INPUT/meta['file']
    assert sha(path)==meta['sha256']
    with np.load(path) as archive:
        rgb=torch.from_numpy(archive['images'])
    rgb=2*F.interpolate(rgb,size=(384,512),mode='bilinear',align_corners=False,antialias=True)-1
    names=[f'{key}_{meta["sha256"][:12]}_{i}' for i in range(len(rgb))]
    views=[dict(img=im[None],true_shape=np.array([[384,512]],dtype=np.int32),idx=i,instance=names[i]) for i,im in enumerate(rgb)]
    pairs=make_pairs(views,scene_graph='complete',prefilter=None,symmetrize=True)
    cache=RAW/'cache'/key
    previous_files=sum(1 for p in cache.rglob('*') if p.is_file())
    cache.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start=time.perf_counter()
    scene=sparse_global_alignment(names,pairs,str(cache),model,lr1=.07,niter1=300,lr2=.01,niter2=300,
                                  device='cuda',opt_depth=True,shared_intrinsics=False,matching_conf_thr=0.)
    points,depths,_=scene.get_dense_pts3d(clean_depth=False)
    depth=torch.stack([v.reshape(384,512) for v in depths])
    c2w=scene.cam2w.detach()
    k=scene.intrinsics.detach()
    for i,p in enumerate(points):
        world=p.reshape(384,512,3)
        z=(world-c2w[i,:3,3]) @ c2w[i,:3,2]
        torch.testing.assert_close(z,depth[i],rtol=2e-4,atol=2e-5)
    assert torch.isfinite(depth).all()
    nonpositive = (depth<=0).sum(dim=(1,2)).tolist()
    torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    raw=RAW/(key+'.npz')
    write_arrays(raw,dict(depth=depth.detach().cpu().numpy(),intrinsics=k.cpu().numpy(),c2w=c2w.cpu().numpy()))
    record=dict(input_sha256=meta['sha256'],output_sha256=sha(raw),checkpoint_sha256=WEIGHTS,source=SOURCE,
                pairs=len(pairs),seconds=seconds,prior_cache_files=previous_files,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                nonpositive_depth_pixels=nonpositive,device=torch.cuda.get_device_name(),native_size=[384,512])
    dump(PUBLIC/(key+'.json'),record)
    print(key,record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--scenes',nargs='+',choices=('heads','chess','fire'),default=['heads','chess','fire'])
    args=p.parse_args()
    assert sha(args.weights/'model.safetensors')==WEIGHTS
    assert sha(args.weights/'config.json')=='718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4'
    model=AsymmetricMASt3R.from_pretrained(str(args.weights),local_files_only=True).eval().cuda()
    for slug in args.scenes:
        key=slug+'_base'
        result=PUBLIC/(key+'.json')
        if result.exists():
            assert read(result)['output_sha256']==sha(RAW/(key+'.npz'))
            continue
        capture(model,key)
