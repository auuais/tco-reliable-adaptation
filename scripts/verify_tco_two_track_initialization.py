"""Audit a trained suffix restoration and fresh native Adam in one untimed, input-free GPU process."""
import argparse
from pathlib import Path
import subprocess
import time
from tco_final_ln_protocol import ROOT, IMAGE, read, seal, sha


def inside(arm, artifact, expected, output):
    import torch
    from safetensors.torch import load_file
    import run_experiment as base
    from run_tco_courtyard_camera import seed_model
    from run_tco_final_ln_model import warm_local
    from run_tco_frozen_prefix import CHECKPOINT
    from tco_frozen_prefix import place
    from tco_two_track_hybrid import HybridAdamAudit, tensor_digest

    if sha(artifact) != expected:
        raise ValueError("Wrong training file")
    warm_local()
    seed_model(42)
    start = time.perf_counter()
    model = base.tco_module.TCO_VGGT_LoRA.from_pretrained(CHECKPOINT,enable_track=False).eval()
    audit = HybridAdamAudit(model,arm,42,"restore",artifact)
    place(model,audit.path)
    optimizer, params = model.backbone_peft_setup(lr=.0003,lora_rank=4,lora_alpha=16,
                         lora_target_modules=["qkv","proj","fc1","fc2"],lora_dropout=0.,merge_weights=False)
    torch.cuda.synchronize()
    source = load_file(str(artifact))
    named = dict(model.named_parameters())
    if (set(source)!=set(audit.factors) or len(params)!=192 or len(optimizer.state) or audit.updates!=0
          or len({id(p) for p in params})!=192):
        raise ValueError("Scope or Adam state differs from fresh 192-factor setup")
    digest_by_name = {}
    for name, goal in source.items():
        p = audit.factors[name]
        if (named[name] is not p or p.device.type!="cuda" or not p.requires_grad
              or p.grad is not None or p.dtype!=torch.float32 or goal.dtype!=p.dtype
              or not torch.isfinite(goal).all() or not torch.equal(p.detach().cpu(),goal)):
            raise ValueError("Restoration is not exact for "+name)
        digest_by_name[name] = tensor_digest(goal)
    seal(output,dict(arm=arm,initialization_sha256=expected,restore_policy_sha256=sha(ROOT/"scripts/tco_two_track_hybrid.py"),
         initialization_seconds=audit.initialization_seconds,factors=len(digest_by_name),named_verified_sha256=digest_by_name,
         completed_adam_updates=audit.updates,states_before_scene=len(optimizer.state),source_images=0,
         gpu=torch.cuda.get_device_name(),process_seconds=time.perf_counter()-start,passed=True,rental_compute_usd=0))


def launch(arm):
    from run_profiled_d0_local import DOCKER,docker_json,ensure_stopped
    from run_profiled_d1_local import host_lock
    from run_tco_courtyard_camera_local import check_device_available
    from run_tco_final_ln_local import docker_base,mount
    from run_tco_two_track_a_local import PREP, RAW, STUDY, POLICY, committed
    index=ROOT/"results"/STUDY/"a_trained_initializations.json"
    committed(Path(__file__), POLICY, index)
    expected=read(index)["initializations"][arm]
    artifact=PREP/"initializations"/(expected+".safetensors")
    if sha(artifact)!=expected:
        raise ValueError("Training file has changed")
    case=RAW/"restore_audits"/expected
    case.mkdir(parents=True,exist_ok=True)
    receipt=case/"proof.json"
    with host_lock():
        if not receipt.exists():
            check_device_available()
            name="tco-restore-"+arm.lower()+"-"+str(time.time_ns())
            cmd=docker_base(name,True)
            cmd[cmd.index("--label")+1]="tco.study="+STUDY
            cmd+=mount(Path("E:/ResearchData/TCO/checkpoints/VGGT-1B"),"/data/checkpoints/VGGT-1B")
            cmd+=mount(artifact,"/inputs/initial.safetensors")+mount(case,"/case",True)
            cmd+=[IMAGE,"python","-B","-u","scripts/verify_tco_two_track_initialization.py",arm,"--inside",
                  "--artifact","/inputs/initial.safetensors","--sha256",expected,"--output","/case/proof.json"]
            cid=subprocess.check_output(cmd,text=True,timeout=45).strip()
            try:
                info, = docker_json(["inspect",cid])
                if (info["Image"]!=IMAGE or info["HostConfig"]["NetworkMode"]!="none" or
                      not info["HostConfig"].get("DeviceRequests") or info["Config"]["Labels"].get("tco.study")!=STUDY):
                    raise RuntimeError("Wrong local restore worker")
                with (case/(name+".log")).open("xb") as log:
                    subprocess.run(DOCKER+["start","--attach",cid],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            finally:
                stopped,issues=ensure_stopped(cid)
                if not stopped or stopped["State"]["Running"]:
                    raise RuntimeError("Could not stop restore worker "+str(issues))
                subprocess.run(DOCKER+["rm",cid],check=True,capture_output=True,timeout=25)
        proof=read(receipt)
        if (not proof["passed"] or proof["arm"]!=arm or proof["initialization_sha256"]!=expected or
                 proof["restore_policy_sha256"]!=sha(POLICY) or proof["factors"]!=192):
            raise ValueError("Stale or incomplete restore audit")
        summary={k:v for k,v in proof.items() if k!="named_verified_sha256"}
        seal(ROOT/"results"/STUDY/("a_restore_"+arm+".json"),dict(summary,complete_sha256=sha(receipt)))
    print(arm, "192 trained factors restored exactly; Adam starts empty",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm",choices=("Distilled","OfflineOriginal"))
    ap.add_argument("--inside",action="store_true")
    ap.add_argument("--artifact",type=Path)
    ap.add_argument("--sha256")
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.inside:
        inside(args.arm,args.artifact,args.sha256,args.output)
    else:
        launch(args.arm)
