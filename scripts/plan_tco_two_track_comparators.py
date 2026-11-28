"""Time-only comparator lock and a conditional fresh roster for the selected Track A model."""
import argparse
from itertools import product
import math
import random
from statistics import mean, median
from pathlib import Path

from lock_tco_two_track_jobs import attach, write_roster
from tco_two_track_protocol import (ARMS, DEV, MANIFEST, NEW_SEEDS, ORDER_SEED, PUBLIC, ROOT, SEEDS,
                                    STUDY, digest, read, seal)

PREP = Path("E:/ResearchData/TCO/preparation") / STUDY
POLICY = ROOT / "manifests" / (STUDY + "_comparator_policy.json")
NOM = PUBLIC / "a_nomination.json"
MATCH = PUBLIC / "a_time_match.json"


def jobs_for(stage, packets, seeds, arms, salt):
    rng = random.Random(ORDER_SEED + salt)
    blocks = list(product(sorted(packets, key=lambda p:p["room"]), seeds))
    rng.shuffle(blocks)
    items = []
    for packet, seed in blocks:
        shuffled = list(arms)
        rng.shuffle(shuffled)
        for arm in shuffled:
            if arm == "ZERO" and seed != seeds[0]:
                continue
            count = 0 if arm == "ZERO" else int(arm[4:]) if arm.startswith("Full") else ARMS[arm]
            items.append(dict(room=packet["room"], id=f"{stage}_{packet['slug']}_{seed}_{arm}",
                              arm=arm, seed=seed, steps=count, profile="clean"))
    return items


def select_count(times, rooms, seeds, nominee, counts):
    """Input is exclusively ordered model time by (arm,room,seed), free of geometry."""
    ratios = {}
    for k in counts:
        per_room = {}
        for room in rooms:
            den = [times[nominee,room,s] for s in seeds]
            num = [times[f"Full{k}",room,s] for s in seeds]
            if not all(math.isfinite(x) and x > 0 for x in den+num):
                raise ValueError("Missing, nonfinite or zero model duration")
            per_room[room] = median(num) / median(den)
        ratios[k] = dict(room_ratios=per_room, ordinary_over_candidate=mean(per_room.values()))
    eligible = [k for k in counts if ratios[k]["ordinary_over_candidate"] <= 1.]
    return (max(eligible) if eligible else None), ratios


def require_nominee():
    nominee = read(NOM)["candidate"]
    if nominee not in ("Refresh1","Refresh2","Narrow","Distilled"):
        raise ValueError("No eligible Track A nominee for timing or fresh confirmation")
    return nominee


def time_roster(low=False):
    require_nominee()
    stage = "time_short" if low else "time_grid"
    trigger = PUBLIC/"a_time_short_trigger.json" if low else NOM
    if low and not read(trigger)["proceed"]:
        raise ValueError("Short grid not indicated")
    src = PUBLIC/"development_inputs.json"
    packets = read(src)["rooms"]
    arms = ["Full1","Full2","Full3"] if low else ["Full4","Full5","Full6"]
    jobs = jobs_for(stage,packets,SEEDS,arms,82 if low else 81)
    write_roster(stage,attach(jobs,packets,PREP),src,
                 dict(selection_sha256=digest(trigger),comparator_policy_sha256=digest(POLICY),timing_only=True))


def find_times(stages, arms):
    from score_tco_two_track_a import locate
    times, receipts = {}, []
    for stage in stages:
        path = MANIFEST.with_name(STUDY + "_" + stage + ".json")
        if read(path)["registration_sha256"] != digest(MANIFEST):
            raise ValueError("Unmatched timing registration")
        for job in read(path)["jobs"]:
            if job["arm"] not in arms:
                continue
            _, terms, done = locate(job)
            if not done:
                raise ValueError(f"Unmeasurable time for {job['id']} ({terms[-1]['status']})")
            record = done[0] / "model/complete.json"
            rec = read(record)
            if rec["profile"] != "clean" or rec["steps"] != job["steps"] or not 0 < rec["scene_seconds"]:
                raise ValueError("Expected current uninstrumented scene measurement")
            key = job["arm"],job["room"],job["seed"]
            if key in times:
                raise ValueError("Duplicate source for timing comparison")
            times[key] = rec["scene_seconds"]
            receipts.append(dict(id=job["id"], arm=job["arm"], room=job["room"], seed=job["seed"],
                                 scene_seconds=rec["scene_seconds"], model_receipt_sha256=digest(record)))
    return times, receipts


def choose_time():
    arm = require_nominee()
    stages = read(NOM)["stages"]+["time_grid"]
    counts = [4,5,6,10]
    times, receipts = find_times(stages, {arm,*(f"Full{k}" for k in counts)})
    picked, ratios = select_count(times, DEV, SEEDS, arm, counts)
    low = ratios[4]["ordinary_over_candidate"] > 1.
    seal(PUBLIC/"a_time_short_trigger.json",dict(proceed=low, nomination_sha256=digest(NOM),
         full4_over_candidate=ratios[4]["ordinary_over_candidate"],comparator_policy_sha256=digest(POLICY)))
    if low:
        if not MANIFEST.with_name(STUDY+"_time_short.json").is_file():
            time_roster(True)
            print("Short grid required; run its model ledger before repeating choose-time.")
            return
        extra, extra_receipts = find_times(["time_short"],{"Full1","Full2","Full3"})
        receipts += extra_receipts
        times.update(extra)
        picked, ratios=select_count(times, DEV, SEEDS, arm, [1,2,3]+counts)
    fastest = min(ratios, key=lambda k:(ratios[k]["ordinary_over_candidate"], k))
    seal(MATCH,dict(candidate=arm, selected_k=picked, comparator_k=picked if picked is not None else fastest,
         affordable=picked is not None, ratios={str(k):v for k,v in ratios.items()}, timing_receipts=receipts,
         nomination_sha256=digest(NOM), comparator_policy_sha256=digest(POLICY), rental_compute_usd=0))
    print("Largest affordable ordinary budget", picked)


def fresh_roster(dataset, source):
    arm = require_nominee()
    if dataset=="hypersim":
        primary=read(PUBLIC/"a_arkitscenes_fresh_comparator_decision.json")["analysis"]
        if not primary["candidates"][arm]["equal_step"]["passed"]:
            raise ValueError("Cross-dataset replication requires a supported primary result")
    match = read(MATCH)
    if match["nomination_sha256"] != digest(NOM) or arm != match["candidate"]:
        raise ValueError("Candidate changed after timing lock")
    packets = read(source)["rooms"]
    if len(packets)!=6 or any(len({r[k] for r in packets}) != 6 for k in ("room","input_sha256","reference_sha256","slug")):
        raise ValueError("Expected six independently locked public windows")
    stage = dataset+"_fresh"
    target = "Full"+str(match["comparator_k"])
    compare = list(dict.fromkeys(["Full10","Full40","Cache10",target]))
    jobs = jobs_for(stage,packets,NEW_SEEDS,compare+[arm,"ZERO"],84 if dataset=="hypersim" else 83)
    if arm=="Distilled":
        init=read(PUBLIC/"a_trained_initializations.json")["initializations"][arm]
        for job in jobs:
            if job["arm"]==arm:
                job["initialization_sha256"]=init
    model_ledger=write_roster(stage,attach(jobs,packets,PREP),source,dict(
        dataset=dataset, candidate=arm, comparable_time_sha256=digest(MATCH),
        comparator_policy_sha256=digest(POLICY)))
    cohort=dict(study=STUDY,dataset=dataset,candidate=arm,controls=compare+["ZERO","ZERO_supplied"],
        singleton_controls=["ZERO","ZERO_supplied"],time_control=target,seeds=NEW_SEEDS,
        rooms=[{**p,"scene":p["room"]} for p in packets],model_ledger_sha256=digest(model_ledger),
        eligibility_sha256=digest(source),timing_lock_sha256=digest(MATCH),
        independent_zero_room=min(p["room"] for p in packets),
        model_jobs=len(jobs),zero_forward_count=6,expected_supplied_readouts=6)
    seal(MANIFEST.with_name(STUDY+"_"+stage+"_cohort.json"),cohort)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("time-roster","choose-time","fresh-roster"))
    parser.add_argument("--dataset",choices=("arkitscenes","hypersim"))
    parser.add_argument("--inputs",type=Path)
    args=parser.parse_args()
    if args.action=="time-roster":
        time_roster()
    elif args.action=="choose-time":
        choose_time()
    elif args.dataset and args.inputs:
        fresh_roster(args.dataset,args.inputs)
    else:
        parser.error("fresh-roster requires --dataset and --inputs")
