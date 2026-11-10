"""Test retained tensors, both heads and input gradients on the real pretrained model."""
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from evaluation_stability import dump,sha
from repeatability_capture import array_digest
from run_repeat_calibration import tco_module
from selected_readouts import enable,ROOT


def check(checkpoint,source,output):
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    model=tco_module.TCO_VGGT_LoRA.from_pretrained(str(checkpoint),enable_track=False).cpu().eval()
    model.point_head=None
    model.requires_grad_(False)
    with np.load(source,allow_pickle=False) as file:
        rgb=torch.from_numpy(file["images"][:2]).float()
    images=torch.nn.functional.interpolate(rgb,size=(84,112),mode="bilinear",align_corners=False)[None]
    with torch.no_grad():
        patch=model.extract_patch_tokens(images)
    outcomes={}
    selected=sorted({*model.depth_head.intermediate_layer_idx,model.aggregator.depth-1})
    for mode in ("all","selected"):
        if mode=="selected" and enable(model)!=selected:raise ValueError("Incorrect readout list")
        input_tokens=patch.detach().clone().requires_grad_(True)
        start=time.perf_counter()
        outputs,idx=model.aggregate_with_patch_tokens(input_tokens,1,2,84,112,use_checkpoint=True)
        byte_count=sum(o.numel()*o.element_size() for o in outputs if o is not None)
        retained=[i for i,o in enumerate(outputs) if o is not None]
        encoded=model.camera_head(outputs)[-1]
        depth,confidence=model.depth_head(outputs,images=images,patch_start_idx=idx)
        loss=encoded.square().mean()+depth.mean()+confidence.log().mean()
        loss.backward()
        arrays={"pose_encoding":encoded.detach().numpy(),"depth":depth.detach().numpy(),"confidence":confidence.detach().numpy(),
                "patch_gradient":input_tokens.grad.detach().numpy(),
                **{f"layer_{i}":outputs[i].detach().numpy() for i in selected}}
        if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError("Nonfinite outputs or gradients")
        outcomes[mode]=dict(stored_layers=retained,output_list_bytes=byte_count,loss=float(loss.detach()),
                            tensor_sha256={k:array_digest({k:v}) for k,v in arrays.items()},
                            seconds=time.perf_counter()-start)
        del outputs,encoded,depth,confidence,loss,arrays,input_tokens
    equivalent=outcomes["all"]["tensor_sha256"]==outcomes["selected"]["tensor_sha256"]
    with (checkpoint/"model.safetensors").open("rb") as stream:
        checkpoint_hash=hashlib.file_digest(stream,"sha256").hexdigest()
    dump(output,dict(equal_tensors_and_patch_gradient=equivalent,results=outcomes,shape=list(images.shape),device="cpu",
                     checkpoint_sha256=checkpoint_hash,torch=str(torch.__version__),
                     source_tensor_sha256=array_digest({"images":images.numpy()}),
                     patch_sha256=sha(ROOT/"patches/tco-selected-readouts.patch"),source_sha256=sha(Path(__file__)),
                     scope="Two resized real development images, pretrained FP32 camera and depth heads, gradient checkpointing, deterministic scalar loss. This test measures readout-list tensor storage, not device peak allocation, GPU TCO gradients or end-to-end speed."))
    if not equivalent:raise ValueError("Predictions, consumed features or patch gradient changed")
    print("Verified feature, camera, depth, confidence and patch-gradient identity on CPU")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("checkpoint","source","output"):p.add_argument("--"+field,type=Path,required=True)
    args=p.parse_args()
    check(args.checkpoint,args.source,args.output)
