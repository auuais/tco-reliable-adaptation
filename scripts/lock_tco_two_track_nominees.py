"""Freeze the combined public selection decision before any held-out enrollment."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import math
import re
import subprocess

from plan_tco_two_track_comparators import POLICY, select_count
from replay_tco_two_track import choose_a
from score_tco_final_ln import REFERENCE_FILES
from tco_two_track_protocol import DEV, MANIFEST, NEW_SEEDS, PUBLIC, ROOT, SEEDS, STUDY, digest, read, seal

CODE=(
    "configs/7scenes_baseline.yaml","scripts/tco_two_track_protocol.py","scripts/tco_two_track_hybrid.py",
    "scripts/tco_frozen_prefix.py","scripts/run_tco_two_track_a.py","scripts/run_tco_two_track_a_local.py",
    "scripts/run_tco_final_ln_model.py","scripts/lock_tco_two_track_arkit.py","scripts/lock_tco_final_ln_replication.py",
    "scripts/prepare_tco_final_ln_arkit.py","scripts/plan_tco_two_track_comparators.py","scripts/score_tco_two_track_a.py",
    "scripts/replay_tco_two_track.py","scripts/tco_two_track_zero.py","scripts/summarize_tco_two_track_controls.py",
    "third_party/TCO/tco_vggt_lora.py","third_party/TCO/tco_loss.py","third_party/TCO/objective.py",
    "third_party/TCO/utils/interfaces.py","third_party/TCO/utils/geometry.py","third_party/TCO/peft/lora.py",*REFERENCE_FILES)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def select(a,b,independent,registration):
    require(a["registration_sha256"]==b["registration_sha256"]==registration,"Mismatched scientific registration")
    stress=a["stress"]
    if stress is None:
        require(choose_a(a["development"]) is None,"An eligible development arm is missing its stress gate")
    nominee=choose_a(a["development"],stress) if stress is not None else None
    require(a["candidate"]==nominee,"Frozen A nomination differs from paired development/stress ranking")
    check_b_outcome(b,independent)
    require(not b["passed"],"A passing B pilot needs its own conditional selection and comparable-compute lock")
    return nominee


def check_b_outcome(b,independent):
    coverage=independent["coverage"]
    require(b["gates"]["coverage_complete"] and coverage["expected"]==coverage["scored"]==24
            and not coverage["failed"] and not coverage["missing"]
            and b["passed"]==all(b["gates"].values())==independent["eligible"],"Unresolved Track B pilot")


def b_evidence(reg,registration):
    files=[PUBLIC/f for f in ("b_development_decision.json","b_development_independent.json","b_development_rows.json")]
    decision,independent,packet=map(read,files)
    require(all(v["study"]==reg["study"]==STUDY for v in (decision,independent,packet))
            and decision["registration_sha256"]==packet["registration_sha256"]==registration
            and decision["source_lock_sha256"]==packet["source_lock_sha256"]
            and decision["rows_sha256"]==digest(files[2]),"B evidence provenance changed")
    check_b_outcome(decision,independent)
    slots=set()
    for room in reg["b"]["dev_triplets"]:
        for arm,seeds in (("native",room["seeds"]),("reprojected",room["seeds"]),
                          ("triplicate_anchor",[room["anchor"]]),("fusion",[None])):
            slots.update((room["scene"],arm,s) for s in seeds)
    measured=packet["rows"]
    require(len(measured)==len(slots)==24 and slots=={(r["room"],r["arm"],r["seed"]) for r in measured}
            and all(r.get("status","completed")=="completed"
                    and all(math.isfinite(r[k]) and r[k]>0 for k in ("accuracy_mm","completeness_mm"))
                    for r in measured),"B pilot lacks its registered 24 measured rows")
    return decision,independent,files


def verify_time(lock,arm,nom_sha,policy_sha):
    require(lock["candidate"]==arm and lock["nomination_sha256"]==nom_sha and lock["comparator_policy_sha256"]==policy_sha,
            "Wrong nominee or comparable-time policy")
    cases=lock["timing_receipts"]
    times={(r["arm"],r["room"],r["seed"]):r["scene_seconds"] for r in cases}
    require(len(times)==len(cases) and len({r["id"] for r in cases})==len(cases)
            and all(re.fullmatch(r"[0-9a-f]{64}",r["model_receipt_sha256"]) for r in cases),"Ambiguous timing evidence")
    counts=[4,5,6,10]
    _,base=select_count(times,DEV,SEEDS,arm,counts)
    if base[4]["ordinary_over_candidate"]>1:
        counts=[1,2,3]+counts
    require(set(times)=={(a,r,s) for a in [arm]+[f"Full{k}" for k in counts] for r in DEV for s in SEEDS},
            "Timing grid does not match the registered development-only roster")
    k,ratios=select_count(times,DEV,SEEDS,arm,counts)
    fastest=min(ratios,key=lambda i:(ratios[i]["ordinary_over_candidate"],i))
    comparator=k if k is not None else fastest
    require(lock["ratios"]=={str(i):v for i,v in ratios.items()} and lock["selected_k"]==k and
            lock["affordable"]==(k is not None) and lock["comparator_k"]==comparator,"Inconsistent ordinary step selection")
    return dict(arm=f"Full{comparator}",steps=comparator,affordable=k is not None)


def clean_tracked(owner,names,exceptions=()):
    if names:
        subprocess.run(["git","ls-files","--error-unmatch","--",*sorted(names)],cwd=owner,check=True,capture_output=True)
    clean=sorted(set(names)-set(exceptions))
    if clean:
        subprocess.run(["git","diff","--exit-code","HEAD","--",*clean],cwd=owner,check=True,capture_output=True)


def lf_sha(path):
    return sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()


def committed(paths):
    runtime=PUBLIC/"runtime.json"
    historical=ROOT/"manifests/tco_one_percent_v1.json"
    vendor=(ROOT/"third_party/TCO").resolve()
    names={p.resolve() for p in [*paths,runtime,historical,MANIFEST]}
    outer={p.relative_to(ROOT).as_posix() for p in names if vendor not in p.parents}
    nested={p.relative_to(vendor).as_posix() for p in names if vendor in p.parents}
    clean_tracked(ROOT,outer)
    rt,old=read(runtime),read(historical)
    revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=vendor,text=True).strip()
    require(revision==rt["upstream_commit"]==old["upstream"]==read(MANIFEST)["runtime"]["upstream"],
            "Different registered upstream checkout")
    modified=rt["upstream_modified_files"]
    clean_tracked(vendor,nested|set(modified),modified)
    for name,expected in modified.items():
        require(digest(vendor/name)==expected,"Local compatibility patch changed: "+name)
    for name in nested:
        key=(vendor/name).relative_to(ROOT).as_posix()
        require(old["source_sha256_lf"].get(key)==lf_sha(vendor/name),"Different registered TCO source: "+key)
    return dict(upstream_revision=revision,upstream_code_files=len(nested),outer_files=len(outer),
                compatibility_raw_sha256=modified,runtime_receipt_sha256=digest(runtime),
                historical_sources_receipt_sha256=digest(historical))


def registry_fields(reg):
    primary=reg["fresh"]["primary"]
    runtime=reg["runtime"]
    require(primary["seeds"]==NEW_SEEDS and primary["visits"]==6 and re.fullmatch("[0-9a-f]{40}",primary["source_revision"])
            and re.fullmatch("[0-9a-f]{40}",runtime["upstream"]) and re.fullmatch("[0-9a-f]{64}",runtime["checkpoint"])
            and re.fullmatch("sha256:[0-9a-f]{64}",runtime["image"]),"Malformed master runtime or fresh declaration")


def preflight():
    reg=read(MANIFEST)
    registry_fields(reg)
    b,independent,files=b_evidence(reg,digest(MANIFEST))
    sources={ROOT/f for f in CODE}
    audit=committed([*files,POLICY,*sources])
    return dict(source_audit=audit,code_files=len(sources),reference_code_files=len(set(REFERENCE_FILES)),
                b_slots=independent["coverage"]["scored"],b_development_passed=b["passed"])


def write():
    nom=PUBLIC/"a_nomination.json"
    a,reg=read(nom),read(MANIFEST)
    registry_fields(reg)
    registration=digest(MANIFEST)
    b,independent,bfiles=b_evidence(reg,registration)
    arm=select(a,b,independent,registration)
    locks=[nom,*bfiles,MANIFEST,POLICY]
    for stage in a["stages"]+a["stress_stages"]:
        require(re.fullmatch(r"[a-z_]+",stage),"Invalid source stage")
        rows=PUBLIC/("a_"+stage+"_rows.json")
        ledger=MANIFEST.with_name(STUDY+"_"+stage+".json")
        stage_rows=read(rows)
        require(stage_rows["plan_sha256"]==digest(ledger) and stage_rows["registration_sha256"]==registration,
                "A nomination source rows or ledger changed")
        locks.extend([rows,ledger])
    if a.get("distillation_resource_closure_sha256"):
        closure=PUBLIC/"a_distillation_local_inconclusive.json"
        require(digest(closure)==a["distillation_resource_closure_sha256"],"Changed resource closure")
        locks.append(closure)
    match=None
    initialization=None
    if arm:
        m=PUBLIC/"a_time_match.json"
        match=verify_time(read(m),arm,digest(nom),digest(POLICY))
        locks.append(m)
        if arm=="Distilled":
            trained=PUBLIC/"a_trained_initializations.json"
            initialization=read(trained)["initializations"][arm]
            require(re.fullmatch(r"[0-9a-f]{64}",initialization),"Unbound trained initialization")
            locks.append(trained)
    code=sorted({ROOT/f for f in CODE})
    source_verification=committed(locks+code+[Path(__file__).resolve()])
    result=dict(study=STUDY,a=arm,b=None,decision="enroll_arkitscenes_validation" if arm else "no_fresh_evaluation",
        enrollment_authorized=bool(arm),fresh_scoring_requires_locked_model_and_reference_rosters=True,
        comparable_ordinary=match,initialization_sha256=initialization,primary_source_revision=reg["fresh"]["primary"]["source_revision"],
        visits=reg["fresh"]["primary"]["visits"],seeds=NEW_SEEDS,controls=(
            list(dict.fromkeys(["Full10","Full40","Cache10",match["arm"],"ZERO","ZERO_supplied"])) if arm else []),
        checkpoint=reg["runtime"]["checkpoint"],docker_image=reg["runtime"]["image"],tco_revision=reg["runtime"]["upstream"],
        source_pins_lf_sha256={p.relative_to(ROOT).as_posix():lf_sha(p) for p in code},source_verification=source_verification,
        decision_inputs_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in locks},registration_sha256=registration,
        rental_compute_usd=0)
    seal(PUBLIC/"nominees.json",result)
    if not arm:
        note=("# Held-out evaluation decision\n\n"
          "Neither direction earned blind enrollment. The fresh ARKitScenes and Hypersim cohorts therefore have no measured accuracy, completeness, time or memory result in this study. [Combined decision](nominees.json)\n\n"
          "Track A required a candidate to improve scene time and stay within the 1% accuracy and 3% completeness margins against ordinary ten-update TCO on development and stress data. Its recorded nomination and any resource-limited branch closure describe which comparisons were possible. The study did not establish held-out superiority to VGGT, ordinary ten-update TCO, an ordinary time-matched control, or full forty-update TCO. [A nomination](a_nomination.json)\n\n"
          "Earlier five-view measurements compare cached and ordinary TCO within the archived study. They provide historical context for speed and geometry and cannot supply missing endpoints for the methods tested here. [Archived comparator tables](a_historical_baselines.json)\n\n"
          "The saved three-run fusion missed its development gate. Native and reprojected singles were measured on three previously exposed rooms. The longer ordinary-TCO comparator and fresh Track B evaluation were conditional on passing that pilot. [B development](b_development_decision.json) · [Independent replay](b_development_independent.json)\n")
        target=PUBLIC/"no_fresh_evaluation.md"
        if target.exists():
            require(target.read_bytes()==note.encode(),"Immutable no-enrollment note differs")
        else:
            with target.open("xb") as f:
                f.write(note.encode())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true",help="Read only registered code and public B receipts; write nothing")
    if parser.parse_args().preflight:
        print(json.dumps(preflight(),sort_keys=True))
    else:
        write()
