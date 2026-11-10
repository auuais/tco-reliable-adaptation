"""Compare two BF16-autocast Adam updates on actual pretrained backbone layers."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from evaluation_stability import dump,sha
from frozen_linear_storage import enable as enable_operands
from repeatability_capture import array_digest
from run_repeat_calibration import tco_module
from selected_readouts import ROOT,enable as select_readouts


def check(source,checkpoint,output):
    if output.exists():raise FileExistsError(output)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=tco_module.TCO_VGGT_LoRA.from_pretrained(str(checkpoint),enable_track=False).cpu().eval()
    model.point_head=None
    model.requires_grad_(False)
    layers=select_readouts(model)
    with np.load(source,allow_pickle=False) as f:
        rgb=torch.from_numpy(f["images"][:2]).float()
    images=torch.nn.functional.interpolate(rgb,size=(84,112),mode="bilinear",align_corners=False)[None]
    with torch.no_grad():
        tokens=model.extract_patch_tokens(images)
    _,params=model.backbone_peft_setup(.0003,4,16,["qkv","proj","fc1","fc2"],0.)
    model.requires_grad_(False)
    model.enable_lora_gradients()
    adapters={n:p for n,p in model.named_parameters() if "lora_A" in n or "lora_B" in n}
    initial={n:p.detach().clone() for n,p in adapters.items()}
    results={}
    for mode in ("normal","cached"):
        with torch.no_grad():
            for n,p in adapters.items():p.copy_(initial[n])
        provenance=enable_operands(model,torch.bfloat16) if mode=="cached" else {}
        opt=torch.optim.Adam(params,lr=.0003)
        records=[]
        for step in range(2):
            start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cpu",dtype=torch.bfloat16):
                features,idx=model.aggregate_with_patch_tokens(tokens,1,2,84,112,use_checkpoint=True)
                cam=model.camera_head(features)[-1]
                depth,conf=model.depth_head(features,images=images,patch_start_idx=idx)
                loss=cam.float().square().mean()+depth.float().mean()+conf.float().log().mean()
            loss.backward()
            grads={n:p.grad.detach().float().numpy() for n,p in adapters.items()}
            if not all(np.isfinite(a).all() for a in grads.values()):raise ValueError("Nonfinite adapter gradient")
            norm=torch.nn.utils.clip_grad_norm_(params,max_norm=1)
            # Hash unclipped gradients before clipping would require a copy. These are the actual clipped operands.
            sig=array_digest({n:p.grad.detach().float().numpy() for n,p in adapters.items()})
            opt.step()
            predictions={n:t.detach().float().numpy() for n,t in (("pose",cam),("depth",depth),("confidence",conf))}
            records.append(dict(step=step,loss=float(loss.detach()),gradient_norm_before_clip=float(norm),clipped_gradient_sha256=sig,
                                head_sha256=array_digest(predictions),features_sha256=array_digest({str(i):features[i].detach().float().numpy() for i in layers}),
                                updated_adapter_sha256=array_digest({n:p.detach().numpy() for n,p in adapters.items()}),seconds=time.perf_counter()-start))
            del features,cam,depth,conf,loss,grads,predictions
        results[mode]=dict(storage=provenance,updates=records)
    equal=all(all(a[k]==b[k] for k in a if k!="seconds") for a,b in zip(results["normal"]["updates"],results["cached"]["updates"]))
    with (checkpoint/"model.safetensors").open("rb") as stream:weights=hashlib.file_digest(stream,"sha256").hexdigest()
    dump(output,dict(results=results,bitwise_match=equal,checkpoint_sha256=weights,adapter_tensors=len(adapters),device="cpu",
                     image_shape=list(images.shape),torch=str(torch.__version__),source_tensor_sha256=array_digest({"images":images.numpy()}),
                     implementation_sha256=sha(ROOT/"scripts/frozen_linear_storage.py"),test_sha256=sha(Path(__file__)),
                     scope="Real pretrained model with two small images. Both arms use four consumed readouts. Two FP32 Adam updates under CPU BF16 autocast, with a scalar objective on camera, depth and confidence. No GPU renderer, device peak-memory or speed claim."))
    print("Two-update CPU equivalence:",equal,flush=True)
    if not equal:raise ValueError("The cached branch changed an observed update")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for k in ("source","checkpoint","output"):parser.add_argument("--"+k,type=Path,required=True)
    args=parser.parse_args()
    check(args.source,args.checkpoint,args.output)
