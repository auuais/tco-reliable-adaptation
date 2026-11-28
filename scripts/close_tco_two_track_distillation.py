"""Close only a verified, locally infeasible distillation branch; replay the original hybrid gates."""
import argparse
from pathlib import Path

from plan_tco_two_track_a_followup import STRESS_CONTROLS, analyze, hybrids
from replay_tco_two_track import choose_a
from score_tco_two_track_a import locate
from tco_final_ln_protocol import IMAGE, read, seal, sha
from tco_two_track_protocol import MANIFEST, NEW_SEEDS, PRIVATE, PUBLIC, SEEDS, STUDY

PREP=Path("E:/ResearchData/TCO/preparation")/STUDY
TRIGGER=PUBLIC/"a_distillation_trigger.json"
TEACHERS=MANIFEST.with_name(STUDY+"_distillation_teachers.json")
ARMS=("Distilled","OfflineOriginal")
RETRIABLE={"timeout","provisioning","interrupted"}


def require(test,message):
    if not test:
        raise ValueError(message)


def final_state(cases,deadline):
    """A missing slot, arbitrary exception, or single ambiguous watchdog cannot close the branch."""
    require(1<=len(cases)<=2,"Expected one or two actual attempts for every required slot")
    first=cases[0]["status"]
    require(len(cases)==1 or first in RETRIABLE,"Unexpected extra attempt after a terminal outcome")
    require(all(c["stopped"] and c["rental_compute_usd"]==0 for c in cases),"Open worker or rental")
    for c in cases:
        if c["status"]=="timeout":
            require(c["wall_seconds"]>=deadline-2 and c.get("error") and "timed out" in c["error"].lower(),
                    "Watchdog receipt lacks the full registered time allowance")
    last=cases[-1]["status"]
    if last=="completed":
        return last
    if last=="oom":
        require(cases[-1]["verified_cuda_oom"],"An explicit CUDA allocation failure is required")
        return last
    require(last=="timeout" and len(cases)==2,"Remain pending: safe retry, repair or definitive local evidence is required")
    return "timeout"


def attempt_log(case,terminal,deadline):
    t=dict(terminal)
    log=case/"stdout.log"
    data=log.read_bytes() if log.is_file() else b""
    t["verified_cuda_oom"]=bool(t["status"]=="oom" and t.get("exit_code",t.get("code")) not in (None,0)
            and (b"CUDA out of memory" in data or b"torch.OutOfMemoryError" in data)
            and (case/"container.json").is_file())
    if t["status"] in ("timeout","oom"):
        require((case/"container.json").is_file() and log.is_file() and t["complete_sha256"] is None
                and not (case/"model/complete.json").exists(),"Ambiguous failed or unstarted native attempt")
    audit=dict(attempt=case.name,status=t["status"],deadline_seconds=deadline,wall_seconds=t["wall_seconds"],
               start_sha256=sha(case/"start.json"),terminal_sha256=sha(case/"terminal.json"),
               stdout_sha256=sha(log) if log.is_file() else None,verified_cuda_oom=t["verified_cuda_oom"])
    return t,audit


def model_rows(ledger):
    plan=read(ledger)
    require(plan["registration_sha256"]==sha(MANIFEST),"Wrong stage registration")
    results=[]
    for j in plan["jobs"]:
        cases,terms,success=locate(j)
        require([p.name for p in cases]==[f"attempt{i+1:02d}" for i in range(len(cases))],"Nonsequential attempts")
        binds=[]
        history=[]
        logs=[]
        for case,term in zip(cases,terms):
            start=read(case/"start.json")
            basis=(start["ledger_sha256"],start["registration_sha256"],start["input_file_sha256"],start["worker_sha256"],
                   start["policy_sha256"],start["image"],start["deadline_seconds"])
            require(basis[0]==sha(ledger) and basis[1]==sha(MANIFEST) and basis[2]==j["input_file_sha256"]
                    and basis[-2:]==(IMAGE,1200) and "RTX 3080" in start["device"],"Different input/runtime/deadline")
            binds.append(basis)
            t,a=attempt_log(case,term,1200)
            history.append(t);logs.append(a)
        require(len(set(binds))==1,"Retry changed input, model, allocator policy or deadline")
        state=final_state(history,1200)
        result=dict(id=j["id"],room=j["room"],seed=j["seed"],arm=j["arm"],steps=j["steps"],status=state,
                    input_sha256=j["input_sha256"],attempts=logs)
        if success:
            model=success[0]/"model"
            r=read(model/"complete.json")
            require(sha(model/"prediction.npz")==r["predictions_sha256"],"Completed prediction bytes differ")
            result.update(completed_adam_updates=r["completed_adam_updates"],model_receipt_sha256=sha(model/"complete.json"),
                          predictions_sha256=r["predictions_sha256"])
        results.append(result)
    return results


def intake_proof():
    gate=read(TRIGGER)
    ledger=read(TEACHERS)
    audit=PUBLIC/"arkit_teacher_intake.json"
    packet=PUBLIC/"arkit_teacher_inputs.json"
    enrollment,prepared=read(audit),read(packet)
    require(gate["proceed"] and ledger["trigger_sha256"]==enrollment["trigger_sha256"]==sha(TRIGGER)
            and prepared["intake_sha256"]==sha(audit) and ledger["packet_receipt_sha256"]==sha(packet)
            and prepared["registration_sha256"]==sha(MANIFEST) and enrollment["split"]==prepared["split"]=="Training"
            and enrollment["source_revision"]==read(MANIFEST)["fresh"]["primary"]["source_revision"],
            "Unmatched scientific trigger, split, or preparation receipts")
    rooms=prepared["rooms"]
    require(rooms==enrollment["selected"] and len(rooms)==len(ledger["jobs"])==6
            and all(len({r[k] for r in rooms})==6 for k in ("visit","room","site_group","input_sha256")),
            "Six registered independent Training visits are required")
    by_room={r["room"]:r for r in rooms}
    require({j["room"] for j in ledger["jobs"]}==set(by_room),"Different teacher roster")
    certificates=[]
    for j in ledger["jobs"]:
        r=by_room[j["room"]]
        require(all(j[k]==value for k,value in r.items()) and j["profile"]=="clean" and j["arm"]=="Full40"
                and j["steps"]==40 and j["seed"]==42 and j["split"]=="Training" and len(set(j["frames"]))==5,
                "Changed teacher definition or source packet")
        detail_file=PUBLIC/f"arkit_teacher_{r['visit']}.json"
        detail=read(detail_file)
        require(detail["frames"]==r["frames"] and detail["input_sha256"]==r["input_sha256"]
                and detail["reference_sha256"]==r["reference_sha256"] and detail["model_npz_sha256"]==r["input_file_sha256"]
                and detail["reference_npz_sha256"]==r["reference_file_sha256"],"Changed underlying room preparation")
        certificates.append(dict(visit=r["visit"],source_detail_sha256=sha(detail_file)))
        for role,key in (("model","input_file_sha256"),("reference","reference_file_sha256")):
            require(sha(PREP/role/(r["slug"]+".npz"))==r[key],"Frozen public packet differs")
    return dict(intake_sha256=sha(audit),packets_sha256=sha(packet),teachers_sha256=sha(TEACHERS),visits=certificates)


def offline_rows():
    from train_tco_two_track_suffix_local import plan_for
    rows=[]
    for arm in ARMS:
        expected,_=plan_for(arm)
        paths=sorted((PRIVATE/"offline_training"/arm).glob("attempt[0-9][0-9]"))
        require([p.name for p in paths]==[f"attempt{i+1:02d}" for i in range(len(paths))],"Nonsequential offline attempts")
        history,logs,binds=[],[],[]
        for p in paths:
            start=read(p/"start.json")
            term=read(p/"terminal.json")
            require(read(p/"plan.json")==expected and start["plan_sha256"]==sha(p/"plan.json")
                    and start["image"]==IMAGE and "RTX 3080" in start["gpu"],"Offline input/job binding failed")
            binds.append((start["plan_sha256"],start["image"],start["worker_sha256"],start["math_sha256"]))
            t,a=attempt_log(p,term,5400)
            a["completed_update_receipts"]=[sha(f) for f in sorted((p/"model").glob("update_*.json"))]
            history.append(t);logs.append(a)
        state=final_state(history,5400)
        require(len(set(binds))==1,"Changed offline optimizer, code or inputs between retries")
        row=dict(arm=arm,status=state,attempts=logs,updates_required=24)
        if state=="completed":
            dest=paths[-1]/"model"
            receipt=read(dest/"complete.json")
            pub=PUBLIC/f"a_training_{arm}.json"
            require(term["complete_sha256"]==sha(dest/"complete.json") and read(pub)["receipt"]==receipt
                    and receipt["optimizer_steps"]==24 and receipt["arm"]==arm and receipt["inputs_sha256"]==sha(paths[-1]/"plan.json")
                    and sha(dest/"initialization.safetensors")==receipt["initialization_sha256"]
                    and sha(PREP/"initializations"/(receipt["initialization_sha256"]+".safetensors"))==receipt["initialization_sha256"],
                    "Incorrect completed offline student")
            row.update(initialization_sha256=receipt["initialization_sha256"],release_sha256=sha(pub))
        rows.append(row)
    return rows


def deployment(stage):
    ledger=MANIFEST.with_name(STUDY+"_"+stage+".json")
    jobs=read(ledger)["jobs"]
    source=PUBLIC/("a_"+stage+"_rows.json")
    released=read(source)
    seeds=SEEDS if stage=="students" else NEW_SEEDS
    key="rooms_dev" if stage=="students" else "rooms_stress"
    wanted={(r["scene"],s,a) for r in read(MANIFEST)["design"][key] for s in seeds for a in ARMS}
    actual=[(j["room"],j["seed"],j["arm"]) for j in jobs if j["arm"] in ARMS]
    require(len(actual)==len(wanted) and set(actual)==wanted,"Changed paired deployment roster")
    init=read(PUBLIC/"a_trained_initializations.json")["initializations"]
    require(set(init)==set(ARMS) and all(j["initialization_sha256"]==init[j["arm"]] and j["steps"]==10
                     and j["profile"]=="clean" for j in jobs if j["arm"] in ARMS),"Changed deployment initialization")
    require(released["plan_sha256"]==sha(ledger) and released["registration_sha256"]==sha(MANIFEST),
            "Missing public terminal rows")
    rr=model_rows(ledger)
    by_id={r["id"]:r for r in released["rows"]}
    require(len(rr)==len(released["rows"])==len(by_id),"Missing deployment slots")
    for r in rr:
        public=by_id[r["id"]]
        require(public["status"]==r["status"] and [x["terminal_sha256"] for x in public["attempts"]]
                ==[x["terminal_sha256"] for x in r["attempts"]],"Released status differs from raw receipts")
        if r["status"]=="completed":
            require(public["predictions_sha256"]==r["predictions_sha256"] and public["model_receipt_sha256"]==r["model_receipt_sha256"]
                    and sha(PRIVATE/"scores/first"/r["id"]/"score.json")==public["score_sha256"],"Successful score is unbound")
        else:
            require(not ({"accuracy_mm","completeness_mm","predictions_sha256"} & public.keys()),"Imputed failed endpoint")
    return rr,dict(stage=stage,ledger_sha256=sha(ledger),scored_rows_sha256=sha(source))


def close(stage):
    require(TRIGGER.is_file(),"Distillation has not been scientifically triggered")
    gate=read(TRIGGER)
    prep=intake_proof()
    teachers=model_rows(TEACHERS)
    students=[]
    deployed=[]
    sources=[]
    failed=lambda rows:[r for r in rows if r["status"]!="completed"]
    if stage=="teachers":
        decisive=failed(teachers)
    else:
        require(not failed(teachers),"Incomplete six-teacher prerequisite")
        students=offline_rows()
        if stage=="offline":
            decisive=failed(students)
        else:
            require(not failed(students),"Incomplete offline training prerequisite")
            init_path=PUBLIC/"a_trained_initializations.json"
            init=read(init_path)
            require(init["initializations"]=={r["arm"]:r["initialization_sha256"] for r in students}
                    and init["training_receipts_sha256"]=={r["arm"]:r["release_sha256"] for r in students},
                    "Matched initialization release differs")
            dev,where=deployment("students")
            deployed=dev
            sources.append(where)
            if stage=="stress_student":
                require(not failed(dev) and read(PUBLIC/"a_student_stress_entry.json")["proceed"],"Student did not earn stress")
                stress,where=deployment(stage)
                deployed=dev+stress
                sources.append(where)
                decisive=failed([r for r in stress if r["arm"] in ARMS])
            else:
                decisive=failed([r for r in dev if r["arm"] in ARMS])
    require(decisive,"All targeted slots finished; use normal science gates")
    stages,candidates,dev=hybrids()
    chosen=choose_a(dev)
    entry=PUBLIC/"a_hybrid_stress_entry.json"
    require(chosen==gate["hybrid"]==read(entry)["candidate"] and gate["development_sha256"]==sha(entry),
            "Hybrid nomination entry differs from independent development replay")
    stress_stages=["stress_hybrid"] if chosen else []
    stress=analyze(stress_stages,[chosen],STRESS_CONTROLS,True) if chosen else None
    can_nominate=choose_a(dev,stress) if stress else None
    require(bool(can_nominate)==gate["stress_pass"],"Hybrid stress gate differs from original trigger")
    path=PUBLIC/"a_distillation_local_inconclusive.json"
    closure=dict(stage=stage,verdict="inconclusive_local_resource_limit",trigger_sha256=sha(TRIGGER),preparation=prep,
                 teacher_rows=teachers,offline_student_rows=students,deployment_rows=deployed,scored_deployment_sources=sources,
                 decisive_slots=[r.get("id",r["arm"]) for r in decisive],eligible_hybrid=can_nominate,
                 downstream="Subsequent distillation stages not earned. No missing output or geometric score was imputed.",
                 registration_sha256=sha(MANIFEST),rental_compute_usd=0)
    seal(path,closure)
    seal(PUBLIC/"a_nomination.json",dict(candidate=can_nominate,development=dev,stress=stress,stages=stages,
         stress_stages=stress_stages,registration_sha256=sha(MANIFEST),rental_compute_usd=0,
         distillation_resource_closure_sha256=sha(path),partial_student_sources=sources))
    print("Verified distillation resource closure:",stage,"Hybrid nominee:",can_nominate)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("teachers","offline","students","stress_student"))
    close(p.parse_args().stage)
