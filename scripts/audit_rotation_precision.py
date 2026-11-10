"""Analytic small-angle controls for the unchanged upstream camera rotation loss."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from evaluation_stability import dump,sha

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/TCO"))
from constraints import direct_pose_energy


def main():
    assert torch.cuda.is_available()
    target=torch.eye(4,device="cuda").repeat(1,2,1,1)
    target[:,1,0,3]=-.2
    rows=[]
    for degrees in [0,.001,.01,.03,.1,.3,.5,1,2,3,4,5,6,8,10,30]:
        for autocast in [False,True]:
            theta=torch.tensor(np.deg2rad(degrees),dtype=torch.float32,device="cuda",requires_grad=True)
            one=theta*0+1;zero=theta*0;c,s=theta.cos(),theta.sin()
            rotation=torch.stack([c,-s,zero,s,c,zero,zero,zero,one]).reshape(3,3)
            first=torch.eye(4,device="cuda")
            second=torch.cat([torch.cat([rotation,torch.tensor([[.2],[0],[0]],device="cuda")],dim=1),first[3:4]],dim=0)
            poses=torch.stack([first,second])[None]
            capture=[];original_acos=torch.acos
            def observe(value):
                capture.append(dict(dtype=str(value.dtype),value=value.detach().float().cpu().tolist()))
                return original_acos(value)
            torch.acos=observe
            try:
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=autocast):
                    loss,_=direct_pose_energy(poses,target,rot_loss_type="angle",trans_loss_type="normed_l1",canonicalize_to_first_view=True)
                gradient=torch.autograd.grad(loss,theta)[0]
            finally:torch.acos=original_acos
            rows.append(dict(second_view_angle_degrees=degrees,mode="cuda_autocast_bfloat16" if autocast else "float32",
                rotation_loss=float(loss.detach()),expected_unclamped_mean_angle=float(np.deg2rad(degrees)/2),
                theta_gradient=float(gradient) if torch.isfinite(gradient) else None,gradient_finite=bool(torch.isfinite(gradient)),acos_inputs=capture))
    out=dict(scope="Known exact rotations; same upstream function; two views; only second view rotates",
        source_sha256=sha(Path(__file__)),upstream_constraints_sha256=sha(ROOT/"third_party/TCO/constraints.py"),
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),rows=rows,
        interpretation="Analytic loss/gradient audit, not a model accuracy experiment. The upstream clamp intentionally gives a small positive floor even at zero angle. No historical execution or frozen source is changed.")
    dump(ROOT/"results/rotation_precision_v6/summary.json",out)
    print(json.dumps([dict(angle=r["second_view_angle_degrees"],mode=r["mode"],loss=r["rotation_loss"],gradient=r["theta_gradient"],acos_dtype=r["acos_inputs"][0]["dtype"]) for r in rows],indent=2))


if __name__=="__main__":main()
