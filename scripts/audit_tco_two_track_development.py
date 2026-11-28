"""Independent local receipt audit for Track A development. No model, geometry scorer or framework imports."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, median
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_two_track_v1"
RAW = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
MANIFEST = ROOT / "manifests" / (STUDY + ".json")
METRICS = ("accuracy_mm", "completeness_mm", "scene_seconds", "peak_allocated_bytes", "peak_reserved_bytes")
SIG = ("dataset", "frames", "input_sha256", "reference_sha256", "input_file_sha256", "reference_file_sha256")


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def arm_accounting(r):
    """Count the original forward, gradient forwards and final readout explicitly."""
    arm, n = r["arm"], r["steps"]
    cached_suffix = arm in ("Cache10", "OfflineOriginal", "Distilled")
    narrow = arm.startswith("Narrow")
    hybrid = narrow or arm.startswith("Refresh")
    full = re.fullmatch(r"Full(\d+)", arm)
    check(hybrid or cached_suffix or (full and int(full[1]) == n), f"Unsupported step/arm: {arm}/{n}")
    calls = n+2
    refresh = [step for step in ((1,6) if arm=="Refresh2" else (1,)) if step<=n] if hybrid else []
    prefix = 16 if narrow else 192
    if hybrid:
        active = [192+prefix if i in refresh else 192 for i in range(1,n+1)]
        states = {f"prefix_{len(refresh)}":prefix, f"suffix_{n}":192}
    elif cached_suffix:
        active, states = [192]*n, {f"suffix_{n}":192}
    else:
        active, states = [384]*n, {f"prefix_{n}":192,f"suffix_{n}":192}
    check(r["completed_adam_updates"]==n and r["active_counts"]==active and r["adam_state_counts"]==states,
          "Incorrect factor scope or Adam moments: "+r["id"])
    c=r["cache"]
    b,h=c["builds"],c["hits"]
    if hybrid:
        call_generation=[0]
        generation=0
        for step in range(1,n+1):
            call_generation.append(generation)
            generation+=int(step in refresh)
        call_generation.append(generation)
        check(c["generation"]==len(refresh) and c["calls_by_generation"]==dict(Counter(map(str,call_generation))),
              "Invalid cache generation changes: "+r["id"])
        wide=1+len(refresh)
        cheap=calls-wide
        uncached=arm.endswith("Uncached")
        if uncached:
            check(b==h==c["bytes"]==0 and c["prefix_stages"]==12*cheap+(11*wide if narrow else 0),
                  "Uncached twin unexpectedly reused work: "+r["id"])
            if narrow:
                check(c["stable_builds"]==c["stable_hits"]==c["stable_bytes"]==0, "Uncached narrow stable boundary")
        else:
            check(len(refresh)<=b and b+h==cheap and h>=0 and (n<=2 or h>0) and c["bytes"]>0,
                  "Missing or extra cached suffix forward: "+r["id"])
            stages=12*b
            if narrow:
                stable=c["stable_builds"]
                check(stable>0 and c["stable_bytes"]>0 and stable+c["stable_hits"]==wide+b, "Stable boundary reuse")
                stages=11*stable+b
            check(c["prefix_stages"]==stages,"Prefix computation count mismatch: "+r["id"])
    elif cached_suffix:
        check(b>0 and h>0 and b+h==calls and c["prefix_stages"]==12*b and c["bytes"]>0 and c["generation"]==0,
              "Wrong fixed cache: "+r["id"])
    else:
        check(b==h==c["bytes"]==c["prefix_stages"]==c["generation"]==0,"Ordinary TCO reused a prefix")
    orders=r["view_permutations"]
    check(len(orders)==len(r["losses"])==n and all(sorted(o)==list(range(5)) for o in orders)
          and all(math.isfinite(x) for x in r["losses"]), "Loss/permutation ledger incomplete")


def fixed_twins(include_narrow):
    """Read original probe receipts and archive member names; never infer per-run hashes for clean timings."""
    output=[]
    for q in sorted(RAW.glob("*qualification_pass.json")):
        if q.name.startswith("narrow") and not include_narrow:
            continue
        audit=read(q)
        check(audit["passed"] and audit["fixed"]["passed"] and audit["fixed"]["sign_negative_control"],"Failed signed probe")
        twins={}
        for case in audit["source_cases"]:
            fs=list((RAW/"cases"/case["id"]).glob("attempt*/model/complete.json"))
            f,=[p for p in fs if sha(p)==case["complete_sha256"]]
            rec=read(f)
            if rec["profile"]!="fixed":
                continue
            arm_accounting(rec)
            initial=rec["lora_initial_sha256"]
            narrow=rec["arm"].startswith("Narrow")
            expected={f"aggregator.{stream}_blocks.{stage}.{module}.lora_{factor}"
                      for stream in ("frame","global") for stage in range(11 if narrow else 0,24)
                      for module in ("attn.qkv","attn.proj","mlp.fc1","mlp.fc2") for factor in ("A","B")}
            check(set(initial)==expected,"Named seed inventory differs")
            for name, digest in initial.items():
                check(re.fullmatch("[0-9a-f]{64}",digest),"Invalid seed digest in "+name)
            with ZipFile(f.parent/"fixed_signed_gradients.npz") as z:
                keys={name[:-4] for name in z.namelist() if name.endswith(".npy")}
                expected={"final/"+p for p in initial}|{"grad/1/"+p for p in initial}
                expected.update("grad/2/"+p for p in initial if int(p.split(".")[2])>=12)
                check(keys==expected,"Incomplete signed archive")
            twins[rec["arm"].endswith("Uncached")]=rec
        check(set(twins)=={True,False},"Missing cached or uncached engineering twin")
        left,right=twins[False],twins[True]
        check(all(left[k]==right[k] for k in ("seed","frames","input_sha256","view_permutations","lora_initial_sha256")),
              "Fixed seeds, factors or render permutations differ")
        output.append(dict(arm=left["arm"],named_initial_factors=len(left["lora_initial_sha256"]),
                           signed_arrays_present=len(expected),qualification_sha256=sha(q)))
    check(output,"Missing qualification")
    return output


def scorer_intervals():
    folders=list((RAW/"fusion/jobs").glob("*"))+list((RAW/"score_batches").glob("tco-*/"))+list((RAW/"zero_controls").glob("*/tco-*/"))
    intervals=[]
    now=datetime.now().astimezone()
    for folder in folders:
        start,end=folder/"start.json",folder/"terminal.json"
        if not start.exists() and not end.exists():
            continue
        a=read(start) if start.exists() else {}
        b=read(end) if end.exists() else {}
        left=a.get("started") or a.get("utc") or b.get("started")
        right=b.get("utc")
        check(left and (right or not end.exists()), "CPU run missing UTC boundaries: "+folder.name)
        first=datetime.fromisoformat(left)
        last=datetime.fromisoformat(right) if right else now
        check(first<=last,"Inverted interval")
        intervals.append((first,last,folder.name,a.get("stage")))
    return intervals


def source_values(job, registry, input_plan):
    src=input_plan[job["room"]]
    signed={k:src[k] for k in SIG}
    check(all(job[k]==val for k,val in signed.items()) and job["frames"]==[0,200,400,600,800],
          "Study inputs changed: "+job["id"])
    packet,=[p for p in registry["design"]["rooms_dev"] if p["scene"]==job["room"]]
    check(all(packet[k]==src[k] for k in ("frames","dataset","input_sha256","reference_sha256")),"Unsigned room")


def timings_complete(r, start, end, intervals):
    check(0<r["placement_seconds"]<=r["scene_seconds"]<=r["cold_seconds"]<=end["wall_seconds"]+1
          and r["checkpoint_seconds"]>0 and r["peak_allocated"]<=r["peak_reserved"] and
          0<r["peak_allocated"]<=r["cap_bytes"]==8960*2**20 and r["oom_count"]==0,"Wrong time/cap accounting")
    a=datetime.fromisoformat(start["started"])
    b=datetime.fromisoformat(end["utc"])
    check(a<b and all(b<=first or a>=last for first,last,name,stage in intervals),"Clean launch overlaps CPU job: "+r["id"])
    return b


def shared_ratios(raw, order):
    rooms=sorted({r["room"] for r in raw.values()})
    metrics={}
    for arm in ("Refresh1","Refresh2","Narrow","Distilled"):
        for control in ("Full10","Full40","Cache10","Refresh1Uncached","OfflineOriginal"):
            by_room={}
            for room in rooms:
                slots=[[raw.get((room,seed,a)) for seed in order] for a in (arm,control)]
                if any(not case or not all(k in case for k in METRICS) for block in slots for case in block):
                    continue
                by_room[room]={k:median(v[k] for v in slots[0])/median(v[k] for v in slots[1]) for k in METRICS}
            if by_room:
                metrics[arm+"/"+control]=dict(rooms=by_room,point={k:mean(row[k] for row in by_room.values()) for k in METRICS})
    return metrics


def audit(stage, partial):
    registry=read(MANIFEST)
    kh=sha(MANIFEST)
    input_plan={r["room"]:r for r in read(PUBLIC/"development_inputs.json")["rooms"]}
    selected=[stage] if stage in ("pilot","development") else ["development",stage]
    if stage=="students" and (PUBLIC/"a_alternative_gate.json").exists() and read(PUBLIC/"a_alternative_gate.json")["candidate"]:
        selected.insert(1,"alternative")
    approved={}
    for p in (ROOT/"manifests").glob(STUDY+"_*.json"):
        data=read(p)
        if data.get("registration_sha256")==kh and "jobs" in data:
            approved[sha(p)]={x["id"]:x for x in data["jobs"]}
    intervals=scorer_intervals()
    include_narrow=any(j["arm"]=="Narrow" for cohort in selected for j in
                       read(ROOT/"manifests"/(STUDY+"_"+cohort+".json"))["jobs"])
    qualified=fixed_twins(include_narrow)
    historical=read(ROOT/"results/tco_one_percent_v1/evidence.json")["rows"]
    origins={(r["scene"],r["seed"],r["arm"],r["steps"]):r for r in historical}
    raw,orders={},defaultdict(list)
    historic, first_loss, counts, waiting={},defaultdict(list),Counter(),[]
    public_scores=0
    case_evidence,ends=[],[]
    for cohort in selected:
        ledger=read(ROOT/"manifests"/(STUDY+"_"+cohort+".json"))
        check(ledger["registration_sha256"]==kh,"Unregistered ledger")
        score_path=PUBLIC/("a_"+cohort+"_rows.json")
        public={r["id"]:r for r in read(score_path)["rows"]} if stage!="pilot" and score_path.exists() else {}
        if public:
            check(set(public)=={job["id"] for job in ledger["jobs"]},"Incomplete published cohort")
        for j in ledger["jobs"]:
            uid=j["id"]
            source_values(j,registry,input_plan)
            attempts=sorted((RAW/"cases"/uid).glob("attempt[0-9][0-9]"))
            if not attempts or not (attempts[-1]/"terminal.json").exists():
                waiting.append(uid)
                continue
            finals=[read(a/"terminal.json") for a in attempts]
            check(len(finals)<=2 and all(t["status"] in ("interrupted","timeout","provisioning") for t in finals[:-1]),"Invalid retries")
            recdir=attempts[-1]
            end=finals[-1]
            counts[end["status"]]+=1
            if public:
                check(public[uid]["status"]==end["status"],"Changed terminal status")
            if end["status"]!="completed":
                continue
            start=read(recdir/"start.json")
            f=recdir/"model/complete.json"
            r=read(f)
            check(read(recdir/"job.json")==j and all(r[k]==v for k,v in j.items())
                  and approved.get(start["ledger_sha256"],{}).get(uid)==j and start["registration_sha256"]==kh
                  and start["input_file_sha256"]==j["input_file_sha256"] and end["complete_sha256"]==sha(f)
                  and end["stopped"] and end["exit_code"]==0 and r["rental_compute_usd"]==0
                  and r["profile"]=="clean" and r["runtime"]["gpu"]==registry["runtime"]["gpu"]
                  and not r["lora_initial_sha256"] and r["fixed_signed_gradients_sha256"] is None,
                  "Contaminated or unmatched clean receipt: "+uid)
            arm_accounting(r)
            ends.append(timings_complete(r,start,end,intervals))
            key=(j["room"],j["seed"],j["arm"])
            check(key not in raw,"Duplicate clean case")
            orders[key[:2]].append(r["view_permutations"])
            row={k:r[k] for k in ("room","arm","seed","scene_seconds")}
            row.update(peak_allocated_bytes=r["peak_allocated"],peak_reserved_bytes=r["peak_reserved"])
            first_loss[key[:2]].append((j["arm"],r["losses"][0]))
            private=RAW/"scores/first"/uid/"score.json"
            if stage!="pilot" and private.exists():
                score=read(private)
                check(all(score[k]==j[k] for k in ("id","room","seed","arm","input_sha256","reference_sha256"))
                      and score["predictions_sha256"]==r["predictions_sha256"] and score["model_receipt_sha256"]==sha(f)
                      and score["model_file_sha256"]==j["input_file_sha256"]
                      and score["reference_file_sha256"]==j["reference_file_sha256"] and score["scoring_cpu"]
                      and score["readout"]=="canonical" and score["evaluator_sha256"]["scripts/run_experiment.py"]==
                      registry["pins"]["scripts/run_experiment.py"],"Scorer provenance")
                row.update({k:score[k] for k in ("accuracy_mm","completeness_mm","valid_points")})
            if uid in public:
                check(all(public[uid][k]==row[k] for k in METRICS) and public[uid]["valid_points"]==row["valid_points"]
                      and public[uid]["score_sha256"]==sha(private),"Published score/model differs")
                public_scores+=1
            raw[key]=row
            old_arm="four" if j["arm"] in ("Full10","Full40") else "cached12" if j["arm"]=="Cache10" else None
            old=origins.get((j["room"],j["seed"],old_arm,j["steps"]))
            if old:
                check(old["input_sha256"]==j["input_sha256"] and old["reference_sha256"]==j["reference_sha256"]
                      and old["frames"]==j["frames"] and old["raster_shape"]==j["raster_shape"],"Historical packet mismatch")
                previous=dict(scene_seconds=old["scene_seconds"],peak_allocated_bytes=old["peak_allocated"],
                              peak_reserved_bytes=old["peak_reserved"],accuracy_mm=1000*old["metrics"]["accuracy_mean_m"],
                              completeness_mm=1000*old["metrics"]["completeness_mean_m"])
                historic[uid]=dict(first_loss_difference=r["losses"][0]-old["losses"]["first_loss"],
                                   current_over_historical={k:row[k]/v for k,v in previous.items() if k in row})
            case_evidence.append(dict(id=uid,complete_sha256=sha(f),terminal_sha256=sha(recdir/"terminal.json"),
                                      policy_sha256=start["policy_sha256"],worker_sha256=start["worker_sha256"]))
    for key, seqs in orders.items():
        longest=max(seqs,key=len)
        check(all(s==longest[:len(s)] for s in seqs),"Mismatched actual permutations for "+str(key))
    for losses in first_loss.values():
        standard=[x for arm,x in losses if arm not in ("Distilled","OfflineOriginal")]
        check(not standard or max(standard)-min(standard)<1e-6,"Shared initial loss differs")
    for room in input_plan:
        valid={v["valid_points"] for key,v in raw.items() if key[0]==room and "valid_points" in v}
        check(len(valid)<=1,"Changed evaluation mask size")
    seed_order=registry["design"]["dev_seeds"]
    ratios=shared_ratios(raw,seed_order)
    verified=[]
    decisions={"pilot":[], "development":["a_alternative_gate.json"],"alternative":["a_hybrid_stress_entry.json"],
               "students":["a_student_stress_entry.json"]}
    for source in decisions[stage]:
        f=PUBLIC/source
        if not f.exists():
            continue
        record=read(f)
        analysis=record["analysis"]
        for arm,entry in analysis["candidates"].items():
            for control,comp in entry["comparisons"].items():
                key=arm+"/"+control
                if key not in ratios:
                    continue
                local=ratios[key]
                check(set(local["rooms"])==set(comp["complete_rooms"]),"Different complete room set")
                for k,s in comp["metrics"].items():
                    if k in METRICS:
                        check(math.isclose(local["point"][k],s["ratio"],rel_tol=1e-12) and all(
                            math.isclose(local["rooms"][room][k],v,rel_tol=1e-12) for room,v in s["room_ratios"].items()),
                            "Independent ratio mismatch: "+key+"/"+k)
                verified.append(source+"/"+key)
    if not partial:
        check(not waiting,"Pending cases")
        if stage!="pilot":
            check(public_scores==counts["completed"] and bool(verified),"Pending geometry rows or published decision replay")
    return dict(study=STUDY,stage=stage,registration_sha256=kh,completed_or_failed=dict(counts),waiting=waiting,
                case_receipts=case_evidence,geometry_rows_matched=public_scores,qualified_fixed_twins=qualified,
                compared_renderer_groups=len(orders),cpu_windows_checked=len([x for x in intervals if
                    (ends and x[0]<=max(ends)) or x[3] in selected]),historical=historic,
                equal_room_ratios=ratios,primary_ratios_verified=verified,rental_compute_usd=0,
                scope="Receipt-only audit. Model and CPU worker wall-clock intervals are disjoint. Fixed twins expose named factor hashes and signed archive inventory; production timing intentionally omits per-factor hashes. Historical performance ratios are descriptive.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("pilot","development","alternative","students"))
    parser.add_argument("--partial",action="store_true",help="Inspect available receipts without publishing.")
    parser.add_argument("--save",action="store_true",help="Publish immutable complete audit.")
    args=parser.parse_args()
    check(not (args.partial and args.save),"Partial audits cannot be sealed")
    result=audit(args.stage,args.partial)
    if args.save:
        f=PUBLIC/("a_"+args.stage+"_receipt_audit.json")
        payload=(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
        if f.exists():
            check(read(f)==result,"Sealed audit changed")
        else:
            f.write_bytes(payload)
    print(json.dumps({k:result[k] for k in ("stage","completed_or_failed","geometry_rows_matched","qualified_fixed_twins",
                      "compared_renderer_groups","cpu_windows_checked","primary_ratios_verified")},indent=2))
