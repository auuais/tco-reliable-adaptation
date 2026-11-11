"""Exercise the actual TCO initial-camera intervention and dense readout before a rental."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

from evaluation_stability import dump, sha
from focal_zoom_observations import MODEL_SHA, ROOT, UPSTREAM, digest_large
from prepare_dense_focal_pilot import INPUT, RAW
from score_saved_geometry import read
from tco_focal_policy import initial_focal_policy


def run(checkpoint):
    sys.path.insert(0,str(UPSTREAM))
    sys.path.insert(0,str(UPSTREAM/"base_models/vggt"))
    from tco_vggt_lora import TCO_VGGT_LoRA
    from utils.geometry import unproject_depth_to_world_points_torch
    if digest_large(checkpoint/"model.safetensors") != MODEL_SHA:
        raise ValueError("Checkpoint bytes differ")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    index=read(RAW/"inputs.json")
    image,prior=index["inputs"]["heads_base"],index["priors"]["heads_tight"]
    for file in (image,prior):
        if sha(INPUT/file["file"])!=file["sha256"]:
            raise ValueError("Unexpected smoke-test input")
    with np.load(INPUT/image["file"],allow_pickle=False) as f:
        rgb=torch.from_numpy(f["images"])[None]
    with np.load(INPUT/prior["file"],allow_pickle=False) as f:
        k=torch.from_numpy(f["intrinsics"])[None]
        e=torch.from_numpy(f["extrinsics"])[None]
    before=k.clone()
    model=TCO_VGGT_LoRA.from_pretrained(str(checkpoint),enable_track=False).eval()
    evidence={}
    def observe(heads,r):
        evidence["decision"]=r
        evidence["initial"]=unproject_depth_to_world_points_torch(heads["depth"],heads["extrinsics"],heads["intrinsics"])
    with torch.inference_mode(),initial_focal_policy(model,"adaptive","relative",None,observe):
        output=model(rgb,gt_intrinsics=k,gt_extrinsics=e,tco_steps=0)
    points=output["points"]
    if (not torch.equal(k,before) or not torch.equal(evidence["initial"],points)
            or points.shape!=(1,5,392,518,3) or not torch.isfinite(points).all()):
        raise ValueError("Normal upstream initial point map or caller priors changed")
    # This is an engineering comparison at zero updates with no reconstruction score.
    decision=evidence["decision"]
    native=read(ROOT/"reports/relative_focal_windows/heads_seq-01/base/camera.json")
    if native["rgb_sha256"]!=image["rgb_sha256"]:
        raise ValueError("Replication image hash differs")
    from measure_focal_crop_response import focal_xy
    difference=float(np.abs(focal_xy(native)-np.asarray(decision["model_focal"])).max())
    return dict(model_sha256=MODEL_SHA,input_rgb_sha256=image["rgb_sha256"],prior_sha256=prior["sha256"],
                initial_world_points_shape=list(points.shape),finite_world_points=int(points.numel()//3),
                initial_world_points_sha256=hashlib.sha256(points.cpu().float().numpy().tobytes()).hexdigest(),
                hooked_and_original_initial_point_maps_bit_equal=True,caller_prior_unchanged=True,
                model_focal_max_abs_difference_from_separate_cpu_head_px=difference,decision=decision,
                optimizer_updates=0,additional_cloud_usd=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    record=run(args.checkpoint)
    dump(args.output,record)
    print("Verified the original zero-step readout for",record["finite_world_points"],"dense point samples.")
