"""Register, run and replay a separate post-core test of the ordinary TCO observer harness."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from unittest.mock import patch

from tco_final_ln_protocol import read, seal, sha
from tco_two_track_protocol import MANIFEST, PRIVATE, PUBLIC, ROOT, STUDY


PLAN=MANIFEST.with_name(STUDY+"_native_bridge.json")
CORE=MANIFEST.with_name(STUDY+"_development.json")
OLDER=ROOT/"manifests/tco_one_percent_v1.json"
RAW=PRIVATE/"native_harness_bridge"
CURRENT="scripts/run_tco_two_track_a.py"
LEAN="scripts/run_tco_native_lifetime_bridge.py"
OLD_OBSERVER="scripts/run_tco_one_percent.py"
SHARED=("scripts/tco_frozen_prefix.py","configs/7scenes_baseline.yaml","scripts/run_experiment.py",
    "third_party/TCO/tco_vggt_lora.py","third_party/TCO/tco_loss.py","third_party/TCO/objective.py",
    "third_party/TCO/utils/interfaces.py","third_party/TCO/utils/geometry.py","third_party/TCO/peft/lora.py")
FILES=(CURRENT,LEAN,OLD_OBSERVER,"scripts/tco_two_track_hybrid.py","scripts/run_tco_final_ln_model.py",
       "scripts/run_tco_two_track_a_local.py",Path(__file__).resolve().relative_to(ROOT).as_posix(),*SHARED)
HEAD="development_heads_seq-01_42_Full10"
FIRE="development_fire_seq-03_44_Full40"


def file_lf(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def history_audit():
    """Bind published/archive controls and hash current source files; no model or reference is opened."""
    previous=ROOT/"results/tco_one_percent_v1/evidence.json"
    archive=read(previous)["rows"]
    old=next(r for r in archive if r["id"]=="heads_seq01_four_10_42")
    old_case=ROOT/"reports/tco_one_percent_v1/cases"/old["id"]/old["attempt"]/"complete.json"
    archived=read(old_case)
    folder=PRIVATE/"cases"/HEAD/"attempt01"
    recent=read(folder/"model/complete.json")
    started=read(folder/"start.json")
    pilot_file=PUBLIC/"a_resource_pilot.json"
    pilot=next(r for r in read(pilot_file)["cases"] if r["id"]==HEAD)
    if (any(old[k]!=recent[k] for k in ("seed","steps","input_sha256","reference_sha256","frames","raster_shape"))
            or old["scene"]!=recent["room"] or old["output_sha256"]!=archived["output_sha256"]
            or sha(folder/"model/complete.json")!=pilot["complete_sha256"]
            or not all(old["runtime"][k]==recent["runtime"][k] for k in ("torch","cuda","gpu","allocator","gsplat_sha256"))
            or read(OLDER)["docker_image"]!=started["image"]):
        raise ValueError("Invalid historical baseline pairing")
    checkpoint=Path("E:/ResearchData/TCO/checkpoints/VGGT-1B/model.safetensors")
    with checkpoint.open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        if n>16*1024**2:
            raise ValueError("Unexpected checkpoint header")
        header=json.loads(f.read(n))
    specs=[v["shape"] for k,v in header.items() if re.fullmatch(
        r"aggregator\.(?:frame|global)_blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.fc[12])\.weight",k)]
    if len(specs)!=192 or any(len(s)!=2 for s in specs):
        raise ValueError("Different full adapter inventory")
    numel=sum(4*sum(s) for s in specs)
    log_file=PRIVATE/"cases"/FIRE/"attempt01/stdout.log"
    held=log_file.read_text(errors="replace")
    steps=[int(x) for x in re.findall(r"^\s*(\d+)\s*\|",held,re.MULTILINE)]
    keys=("scene_seconds","placement_seconds","cold_seconds","peak_allocated","peak_reserved","cuda_free_before",
          "cuda_free_after","host_peak_bytes")
    summary=dict(historical={k:old[k] for k in keys},current={k:recent[k] for k in keys},
        current_minus_historical_live_bytes=recent["peak_allocated"]-old["peak_allocated"],
        current_minus_historical_reserved_bytes=recent["peak_reserved"]-old["peak_reserved"],
        current_over_historical_seconds=recent["scene_seconds"]/old["scene_seconds"],
        historical_cap_bytes=int(old["cuda_device_total"]*old["runtime"]["gpu_memory_fraction"]),current_cap_bytes=recent["cap_bytes"],
        historical_first_last=old["losses"],current_first_last=[recent["losses"][0],recent["losses"][-1]],
        current_allocator_retries=recent["allocation_retries"],historical_allocator_retries=None,
        native_file_pins_verified={k:file_lf(ROOT/k)==read(OLDER)["source_sha256_lf"][k] for k in (OLD_OBSERVER,*SHARED)},
        potential_retained_adam=dict(rank=4,target_linear_layers=len(specs),lora_parameters=numel,
              two_fp32_moments_bytes=2*numel*4,applies_after_model_return=True),
        logical_observer_bytes=dict(current_per_update_assuming_fp32_scalar_and_int64_order=44,
             historical_postloop_two_fp32_scalars=8,ten_extra=432,forty_extra=1752),
        historical_evidence_sha256=sha(previous),historical_receipt_sha256=sha(old_case),current_pilot_sha256=sha(pilot_file),
        current_receipt_sha256=sha(folder/"model/complete.json"),pilot_worker_revision=started["revision"],
        pilot_worker_sha256=started["worker_sha256"],bridge_current_worker_file_sha256=sha(ROOT/CURRENT),
        fire_first_attempt=dict(terminal=read(log_file.parent/"terminal.json")["status"],log_sha256=sha(log_file),
             last_printed_pre_backward_loss_step=max(steps,default=None),autograd_engine_stack_mentions=held.count("_engine_run_backward"),
             checkpoint_recompute_stack_mentions=held.count("in recompute_fn"),
             historical_same_seed_seconds=next(r["scene_seconds"] for r in archive if r["id"]=="fire_seq03_four_40_44")),
        scope="CPU metadata and code audit only. Event counters and stream timing do not identify an allocation request.")
    if not all(summary["native_file_pins_verified"].values()):
        raise ValueError("Historical engine source no longer matches pinned protocol")
    seal(PUBLIC/"a_native_harness_provenance.json",summary)


def registration():
    original=read(OLDER)
    pins={f:file_lf(ROOT/f) for f in FILES}
    for f in (OLD_OBSERVER,*SHARED):
        if pins[f]!=original["source_sha256_lf"][f]:
            raise ValueError("Historical clean or shared native engine changed: "+f)
    core={j["id"]:j for j in read(CORE)["jobs"]}
    jobs=[]
    for ref,variant in ((HEAD,"legacy_lifetimes"),(HEAD,"current"),(FIRE,"legacy_lifetimes")):
        name="heads" if ref==HEAD else "fire"
        j=dict(core[ref],id="nativebridge_"+name+"_"+variant,harness=variant,original_case_id=ref)
        jobs.append(j)
    seal(PLAN,dict(study=STUDY,role="Ancillary harness investigation outside all model-selection and primary rows",
        version=1,source_sha256_lf=pins,core_sha256=sha(CORE),registration_sha256=sha(MANIFEST),
        historical_sha256=sha(OLDER),jobs=jobs,order="Heads lean first, unchanged current second, fresh processes under one host lock",
        common="Production config and renderer, same GPU, absolute 8960 MiB cap, warm-up, CPU threads, live stdout, seed, input, loss, ten/forty real updates, FP32 delivered surface and timer boundary",
        lean="Use archived install_variant(model,'four',...,audit=None) and observe_losses(...,None,False), historical non-owning update counter, first/last CUDA scalar only. No retained permutation or Adam owner beyond model return.",
        arithmetic="Both complete expected updates. First losses must match within 1e-5 absolute plus 1e-5 relative. Exact final global CUDA state on lean must match replay of N native randperm(5) calls, and inferred orders equal current directly observed schedule. Report terminal loss and native prediction field differences descriptively.",
        fire_gate="Only when both registered Fire44 Full40 primary attempts ended in watchdog timeout, Heads completions pass arithmetic and schedule checks, and current Heads takes at least 2x lean AND >=30 s longer AND has more allocator retries. Then one lean Fire44 40-step case. No rerun of primary failures, no substitutions, no GPU retry for this diagnostic.",
        safety="Deadline 1200 s each; no scorer while the pair runs, no dense reference mount, no rental. Infrastructure failure or reverse/inconclusive contrast stays recorded. No timing-gate changes."))


def check_core():
    for job in read(CORE)["jobs"]:
        attempts=sorted((PRIVATE/"cases"/job["id"]).glob("attempt[0-9][0-9]"))
        if not attempts or not (attempts[-1]/"terminal.json").is_file():
            raise ValueError("Core still active: "+job["id"])
        status=read(attempts[-1]/"terminal.json")["status"]
        if status in ("interrupted","provisioning","timeout") and len(attempts)<2:
            raise ValueError("Registered primary retry is still pending: "+job["id"])


def original_fire_status():
    return [read(p)["status"] for p in sorted((PRIVATE/"cases"/FIRE).glob("attempt[0-9][0-9]/terminal.json"))]


def attempt(job):
    folder=RAW/"cases"/job["id"]/"attempt01"
    if not (folder/"terminal.json").is_file():
        raise ValueError("Missing terminal record "+job["id"])
    terminal=read(folder/"terminal.json")
    if read(folder/"job.json")!=job or terminal["id"]!=job["id"] or not terminal["stopped"]:
        raise ValueError("Native bridge attempt identity differs")
    if terminal["status"]!="completed":
        return folder,terminal,None
    receipt=folder/"model/complete.json"
    result=read(receipt)
    if (sha(receipt)!=terminal["complete_sha256"] or sha(folder/"model/prediction.npz")!=result["predictions_sha256"]
            or result["completed_adam_updates"]!=job["steps"] or result["cap_bytes"]!=8960*1024**2
            or any(result.get(k)!=v for k,v in job.items())):
        raise ValueError("Invalid bridge provenance")
    return folder,terminal,result


def public_case(job,folder,term,rec):
    out=dict(id=job["id"],harness=job["harness"],status=term["status"],terminal_sha256=sha(folder/"terminal.json"),
             elapsed_wall_seconds=term["wall_seconds"],start_sha256=sha(folder/"start.json"))
    if rec is None:
        return out
    return dict(**out,**{k:rec[k] for k in ("scene_seconds","cold_seconds","placement_seconds","peak_allocated",
         "peak_reserved","allocation_retries","oom_count","input_sha256","predictions_sha256","completed_adam_updates",
         "cuda_free_before","cuda_free_after","host_peak_bytes")}, scene_retries_if_measured=rec.get("scene_retries"),
         model_receipt_sha256=sha(folder/"model/complete.json"))


def prediction_delta(left,right):
    """Native tensor differences only; no ground truth or surface alignment."""
    import numpy as np
    metrics={}
    with np.load(left,allow_pickle=False) as lean,np.load(right,allow_pickle=False) as curr:
        if set(lean.files)!=set(curr.files) or set(curr.files)!={"canonical","depth","depth_conf","extrinsics","intrinsics"}:
            raise ValueError("Expected identical captured fields")
        for k in sorted(lean.files):
            a,b=lean[k],curr[k]
            if a.shape!=b.shape or a.dtype!=b.dtype or not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError("Prediction arithmetic shape/dtype/finite check failed")
            diff=a.astype(float)-b
            metrics[k]=dict(shape=list(a.shape),dtype=str(a.dtype),bitwise_equal=np.array_equal(a,b),
                            max_abs=float(np.abs(diff).max()),rms=float(np.sqrt(np.mean(diff**2))))
    return metrics


def fire_rule(first,second,arithmetic,fire):
    if first is None or second is None or not arithmetic or fire!=["timeout","timeout"]:
        return False
    return bool(second["scene_seconds"]>=2.*first["scene_seconds"]
                and second["scene_seconds"]-first["scene_seconds"]>=30.
                and second["allocation_retries"]>first["allocation_retries"])


def summarize_heads():
    ljob,cjob,_=read(PLAN)["jobs"]
    lf,lt,left=attempt(ljob)
    cf,ct,right=attempt(cjob)
    both=left is not None and right is not None
    paired={}
    ok=False
    if both:
        if any(left["runtime"][k]!=right["runtime"][k] for k in ("fraction","allocator","torch","cuda","gpu","gsplat_sha256")):
            raise ValueError("Clean and lean runtime differ")
        endpoints=left["loss_endpoints"],[right["losses"][0],right["losses"][-1]]
        loss_equal=abs(endpoints[0][0]-endpoints[1][0])<=1e-5+1e-5*abs(endpoints[1][0])
        same_order=left["rng_replay_end_state_equal"] and left["replayed_permutations"]==right["view_permutations"]
        ok=loss_equal and same_order
        paired=dict(first_last_losses=list(endpoints), first_loss_within_tolerance=loss_equal,
            lean_cuda_end_state_matches_replay=left["rng_replay_end_state_equal"],lean_rng_sha256=left["rng_end_sha256"],
            orders_match=bool(same_order),orders=right["view_permutations"],
            current_over_lean_time=right["scene_seconds"]/left["scene_seconds"],
            current_minus_lean_seconds=right["scene_seconds"]-left["scene_seconds"],
            native_output_differences=prediction_delta(lf/"model/prediction.npz",cf/"model/prediction.npz"))
    fire=original_fire_status()
    report=dict(plan_sha256=sha(PLAN),first_lean=public_case(ljob,lf,lt,left),second_current=public_case(cjob,cf,ct,right),
         paired=paired,fire_primary_statuses=fire,allow_one_lean_fire=fire_rule(left,right,ok,fire),rental_compute_usd=0,
         interpretation="One fixed-order diagnostic pair. No new estimate of general speed or geometry; retains all primary denominators.")
    seal(PUBLIC/"a_native_harness_heads.json",report)
    return report


def execute(phase):
    import run_tco_two_track_a_local as launcher
    from run_profiled_d1_local import host_lock
    jobs=read(PLAN)["jobs"]
    launcher.committed(PLAN,CORE,*[ROOT/f for f in FILES])
    if read(PLAN)["source_sha256_lf"]!={f:file_lf(ROOT/f) for f in FILES}:
        raise ValueError("Harness code differs from registered bridge")
    check_core()
    if phase=="heads":
        chosen=jobs[:2]
    else:
        r=summarize_heads()
        if not r["allow_one_lean_fire"]:
            print("Conditional Fire diagnostic is not earned.")
            return
        chosen=jobs[2:]
    with host_lock():
        for j in chosen:
            existing=RAW/"cases"/j["id"]/"attempt01"
            if existing.exists():
                attempt(j) # no replacement or retry of an interrupted comparison
                continue
            lean=j["harness"]=="legacy_lifetimes"
            worker=ROOT/(LEAN if lean else CURRENT)
            policy=ROOT/(OLD_OBSERVER if lean else "scripts/tco_two_track_hybrid.py")
            with patch.multiple(launcher,RAW=RAW,WORKER=worker,POLICY=policy,MASTER=PLAN):
                launcher.launch(j,PLAN)
    if phase=="heads":
        summarize_heads()
    else:
        folder,term,result=attempt(chosen[0])
        seal(PUBLIC/"a_native_harness_fire.json",dict(result=public_case(chosen[0],folder,term,result),
             monitored_primary=original_fire_status(),gate_sha256=sha(PUBLIC/"a_native_harness_heads.json"),
             interpretation="One ancillary lean run against previously censored primary cases; no finite paired speed ratio."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("audit","register","heads","fire","replay-heads"))
    a=p.parse_args().action
    if a=="audit":
        history_audit()
    elif a=="register":
        registration()
    elif a=="replay-heads":
        summarize_heads()
    else:
        execute(a)
