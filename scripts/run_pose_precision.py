"""Named camera-loss precision intervention atop the existing local offload path.

Only direct pose-loss arithmetic is moved outside CUDA autocast. Backbone,
camera head, intrinsics loss, rendering, weights and optimizer remain unchanged.
This is an engineering control, not an established novel method.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import run_memory_probe as memory
import tco_loss


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pose-precision",choices=["released","float32"],required=True)
    args,rest=parser.parse_known_args();sys.argv=[sys.argv[0]]+rest
    output=memory.base.ROOT/"runs"/rest[rest.index("--run-id")+1]
    info=dict(mode=args.pose_precision,wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Direct pose energy only; original objective and weights; offloading is explicitly recorded separately",
        observation="First pose-loss inputs saved before intervention; no reference depth used",calls=[])
    original_dump=memory.base.dump
    def dump(path,content):
        if Path(path).name=="run.json":content["pose_precision_intervention"]={k:v for k,v in info.items() if k!="calls"}
        return original_dump(path,content)
    memory.base.dump=dump
    original=tco_loss.direct_pose_energy
    def pose_energy(predicted_poses,gt_extrinsics,**kwargs):
        index=len(info["calls"])
        if index==0:
            np.savez_compressed(output/"pose_inputs_step0.npz",predicted_poses=predicted_poses.detach().float().cpu().numpy(),
                                prior_extrinsics=gt_extrinsics.detach().float().cpu().numpy())
        record=dict(step=index,input_predicted_dtype=str(predicted_poses.dtype),input_prior_dtype=str(gt_extrinsics.dtype),
                    outer_autocast_enabled=torch.is_autocast_enabled("cuda"))
        if args.pose_precision=="float32":
            with torch.autocast("cuda",enabled=False):
                rotation,translation=original(predicted_poses.float(),gt_extrinsics.float(),**kwargs)
        else:rotation,translation=original(predicted_poses,gt_extrinsics,**kwargs)
        record.update(rotation=float(rotation.detach()),translation=float(translation.detach()),
                      rotation_dtype=str(rotation.dtype),translation_dtype=str(translation.dtype))
        info["calls"].append(record)
        (output/"pose_precision.json").write_text(json.dumps(info,indent=2)+"\n")
        return rotation,translation
    tco_loss.direct_pose_energy=pose_energy
    memory.main()


if __name__=="__main__":main()
