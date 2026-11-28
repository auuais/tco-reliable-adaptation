"""Isolated ordinary TCO with historical observer lifetimes on the current timing boundary."""
import argparse
import faulthandler
import hashlib
import math
from pathlib import Path
import resource
import time
from unittest.mock import patch

import numpy as np
import torch
import run_experiment as base
from run_profiled_d0 import sha, write_new
from run_tco_courtyard_camera import seed_model
from run_tco_final_ln_model import CAP, inputs, warm_local
from run_tco_frozen_prefix import CHECKPOINT, FIELDS, export, signature
from run_tco_one_percent import install_variant, observe_losses
from tco_final_ln_protocol import ALLOCATOR, ROOT, read
from tco_frozen_prefix import place
from tco_native_stalls_quality_capacity_protocol import GSPLAT
from utils.geometry import unproject_depth_to_world_points_torch


def run(job, packet, out):
    from gsplat.cuda._backend import _C
    if ((job["room"],job["arm"],job["steps"],job["seed"]) not in
            (("heads/seq-01","Full10",10,42),("fire/seq-03","Full40",40,44))
            or job["profile"]!="clean" or job["harness"]!="legacy_lifetimes" or sha(_C.__file__)!=GSPLAT):
        raise ValueError("Unregistered reference-free native bridge case")
    seed,steps=job["seed"],job["steps"]
    frac=warm_local()
    seed_model(seed)
    supplied,input_id=inputs(packet,job["input_file_sha256"],False)
    if input_id!=job["input_sha256"]:
        raise ValueError("Wrong RGB/rig packet")
    cfg=base.OmegaConf.load(ROOT/"configs/7scenes_baseline.yaml")
    cfg.tco_steps,cfg.tco_lr,cfg.seed=steps,.0003,seed
    cold=time.perf_counter()
    model=base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT,enable_track=False).eval()
    loaded=time.perf_counter()
    fields={}
    model.register_forward_hook(lambda _m,_a,output: fields.update({k:output[k] for k in FIELDS}))
    state={"updates":0}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before,total=torch.cuda.mem_get_info()
    start_counts=torch.cuda.memory_stats()
    start=time.perf_counter()
    # Exact archived production inventory and post-step observer. No observer holds Adam.
    path=install_variant(model,"four",seed,state,None)
    place(model,path)
    torch.cuda.synchronize()
    placement=time.perf_counter()-start
    frozen=tuple((p,p._version) for p in model.parameters())
    _,loss=observe_losses(state,None,False)
    torch.cuda.manual_seed(seed)
    # Match current outer stdout and the actual upstream renderer. The historical
    # production loss observer holds only first and last detached scalar tensors.
    with patch.object(base.tco_module,"compute_tco_losses",loss):
        base.infer_mv_pointclouds_w_priors_v2(supplied,model,cfg,sample_idx=1,scene_name=job["room"].replace("/","-"))
    with torch.no_grad(),torch.autocast("cuda",enabled=False):
        d,e,k=(fields[key].float() for key in ("depth","extrinsics","intrinsics"))
        canonical=unproject_depth_to_world_points_torch(d,e,k)[0].cpu().numpy()
    torch.cuda.synchronize()
    end=time.perf_counter()
    allocated,reserved=torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
    counters=torch.cuda.memory_stats()
    after,_=torch.cuda.mem_get_info()
    # All RNG diagnosis is after the timer and allocator counters are latched.
    global_end=torch.cuda.get_rng_state()
    independent=torch.Generator(device="cuda").manual_seed(seed)
    orders=[torch.randperm(5,device="cuda",generator=independent).cpu().tolist() for _ in range(steps)]
    equivalent=torch.equal(global_end,independent.get_state())
    losses=[float(state[name]) for name in ("first_loss","last_loss")]
    arrays=dict(canonical=canonical,**{name:export(t) for name,t in fields.items()})
    if (not equivalent or state["updates"]!=steps or any(p._version!=v for p,v in frozen) or
            not all(math.isfinite(x) for x in losses) or not 0 < allocated <= CAP or
            canonical.shape!=(5,*supplied["images"].shape[-2:],3) or not all(np.isfinite(v).all() for v in arrays.values())):
        raise ValueError("Native optimization, output or replayed RNG contract differs")
    np.savez_compressed(out/"prediction.npz",**arrays)
    report=dict(**job,loss_endpoints=losses,completed_adam_updates=state["updates"],rng_replay_end_state_equal=equivalent,
         rng_end_sha256=hashlib.sha256(global_end.numpy().tobytes()).hexdigest(),replayed_permutations=orders,
         replay_scope="Native kernel pinned; observed final CUDA state agrees with N independent five-view randperm calls. Per-step order in this lean job is inferred.",
         scene_seconds=end-start,cold_seconds=end-cold,checkpoint_seconds=loaded-cold,placement_seconds=placement,
         peak_allocated=allocated,peak_reserved=reserved,cap_bytes=CAP,cuda_free_before=before,cuda_free_after=after,
         device_total=total,allocation_retries=counters["num_alloc_retries"],scene_retries=counters["num_alloc_retries"]-start_counts["num_alloc_retries"],
         oom_count=counters["num_ooms"],host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         output_sha256=signature(arrays),predictions_sha256=sha(out/"prediction.npz"),rental_compute_usd=0,
         runtime=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                      fraction=frac,allocator=ALLOCATOR,gsplat_sha256=GSPLAT))
    write_new(out/"complete.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("job","inputs","out"):
        p.add_argument("--"+key,required=True,type=Path)
    args=p.parse_args()
    job=read(args.job)
    args.out.mkdir(exist_ok=False,parents=True)
    faulthandler.dump_traceback_later(150,repeat=True)
    try:
        run(job,args.inputs,args.out)
    except Exception as exc:
        write_new(args.out/"failed.json",dict(id=job["id"],error_type=type(exc).__name__,message=str(exc),rental_compute_usd=0))
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
