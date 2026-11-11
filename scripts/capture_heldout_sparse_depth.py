"""Capture frozen held-out VGGT heads from RGB only, including shorter Stairs windows."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from collect_matched_focal_heads import check_source,tensor_digest
from focal_zoom_observations import MODEL_SHA,UPSTREAM,batches,digest_large
from prepare_dense_focal_pilot import ROOT,write_arrays
from evaluation_stability import dump,sha
from score_saved_geometry import read

PUBLIC=ROOT/'results/heldout_sparse_depth'
RAW=ROOT/'reports/heldout_sparse_depth'


def images(assets,scene,entry):
    frames=[]
    for f in entry['targets']:
        path=assets/scene/f'frame-{f:06d}.color.png'
        assert sha(path)==entry['files'][str(f)]['color.png']
        with Image.open(path) as image:
            resized=image.convert('RGB').resize((518,392),Image.Resampling.LANCZOS)
        frames.append((np.asarray(resized,dtype=np.float32)/255).transpose(2,0,1))
    return next(batches(np.stack(frames)))[1]


def capture(assets,weights,public=PUBLIC,raw=RAW,rgb_directory=None):
    import torch
    from selected_readouts import enable
    from stage_initial_forward import stage_initial_forward
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'base_models/vggt')]
    from tco_vggt_lora import TCO_VGGT_LoRA
    from utils.geometry import unproject_depth_to_world_points_torch as unproject
    check_source()
    assert digest_large(weights/'model.safetensors')==MODEL_SHA
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    model=TCO_VGGT_LoRA.from_pretrained(str(weights),enable_track=False).eval()
    model.point_head=None
    enable(model)
    stage_initial_forward(model,lambda name,value:None)
    for scene,entry in read(public/'input_index.json')['sequences'].items():
        key=scene.replace('/','_')
        path=raw/(key+'.npz')
        meta=public/'captures'/(key+'.json')
        if meta.exists():
            assert read(meta)['archive_sha256']==sha(path)
            continue
        if rgb_directory is None:
            rgb=images(assets,scene,entry)
        else:
            prepared=rgb_directory/entry['rgb_file']
            assert sha(prepared)==entry['rgb_sha256']
            with np.load(prepared) as archive:
                rgb=archive['images']
            assert tensor_digest(rgb)==entry['rgb_tensor_sha256']
            assert rgb.shape==(len(entry['targets']),3,392,518)
        model.to('cuda')
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            tokens,heads=model.forward_original_vggt(torch.from_numpy(rgb).unsqueeze(0).to('cuda'))
            native=unproject(*(heads[n] for n in ('depth','extrinsics','intrinsics')))[0].float().cpu().numpy()
            values={n:heads[n][0].float().cpu().numpy() for n in ('depth','extrinsics','intrinsics')}
            del tokens,heads
        assert native.shape==(len(entry['targets']),392,518,3)
        assert np.all(np.isfinite(native)) and np.all(values['depth']>0)
        seconds=time.perf_counter()-start
        raw.mkdir(parents=True,exist_ok=True)
        write_arrays(path,dict(native=native,**values))
        dump(meta,dict(scene=scene,target_frames=entry['targets'],model_sha256=MODEL_SHA,
                       rgb_tensor_sha256=tensor_digest(rgb),archive_sha256=sha(path),
                       peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,seconds=seconds))
        print(scene,len(rgb),'heads sealed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    args=p.parse_args()
    capture(args.assets,args.weights)
