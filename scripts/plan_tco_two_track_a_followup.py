"""Apply the registered development and stress gates to immutable rows and schedule the next local branch."""
import argparse
from pathlib import Path

from lock_tco_two_track_jobs import attach, write_roster
from replay_tco_two_track import A_CANDIDATES, a_analysis, choose_a, next_hybrid
from tco_two_track_protocol import MANIFEST, NEW_SEEDS, PUBLIC, SEEDS, STUDY, digest, read, schedule, seal

PREP = Path("E:/ResearchData/TCO/preparation") / STUDY
CORE = ["Full10","Full40","Cache10","Refresh1Uncached"]
STRESS_CONTROLS = ["Full10","Cache10"]


def rows(stage):
    return read(PUBLIC/f"a_{stage}_rows.json")["rows"]


def analyze(stages, candidates, controls, stress=False):
    registry = read(MANIFEST)
    packets = registry["design"]["rooms_stress" if stress else "rooms_dev"]
    selected = set(candidates+controls)
    cases = [r for s in stages for r in rows(s) if r["arm"] in selected]
    return a_analysis(cases, packets, NEW_SEEDS if stress else SEEDS, candidates, controls)


def ledger(stage, arms, selection, stress=False, init=None):
    source = PUBLIC/("stress_inputs.json" if stress else "development_inputs.json")
    rooms = read(source)["rooms"]
    jobs = schedule(stage,[r["room"] for r in rooms],NEW_SEEDS if stress else SEEDS,arms,2 if stress else 1)
    for job in jobs:
        if init and job["arm"] in init:
            job["initialization_sha256"] = init[job["arm"]]
    write_roster(stage,attach(jobs,rooms,PREP),source,dict(selection_sha256=digest(selection)))


def outcome(stage, data):
    path = PUBLIC/f"a_{stage}.json"
    seal(path,data)
    return path


def alternative():
    result=analyze(["development"],["Refresh1"],CORE)
    arm=next_hybrid(result)
    path=outcome("alternative_gate",dict(candidate=arm,analysis=result,source_sha256=digest(PUBLIC/"a_development_rows.json")))
    if arm:
        ledger("alternative",[arm],path)
    print("Alternative",arm)


def hybrids():
    arm=read(PUBLIC/"a_alternative_gate.json")["candidate"]
    stages=["development"]+(["alternative"] if arm else [])
    candidates=["Refresh1"]+([arm] if arm else [])
    return stages,candidates,analyze(stages,candidates,CORE)


def stress_hybrid():
    stages,candidates,dev=hybrids()
    choice=choose_a(dev)
    path=outcome("hybrid_stress_entry",dict(candidate=choice,candidates=candidates,analysis=dev,stages=stages))
    if choice:
        ledger("stress_hybrid",[choice,*STRESS_CONTROLS],path,stress=True)
    print("Hybrid for stress",choice)


def trigger():
    stages,candidates,dev=hybrids()
    chosen=read(PUBLIC/"a_hybrid_stress_entry.json")["candidate"]
    stress=(analyze(["stress_hybrid"],[chosen],STRESS_CONTROLS,True) if chosen else None)
    passed=bool(chosen and stress["candidates"][chosen]["equal_step"]["passed"])
    stretch=bool(passed and dev["candidates"][chosen]["stretch_point"]["passed"])
    path=outcome("distillation_trigger",dict(proceed=not stretch,hybrid=chosen,stress_pass=passed,
                 retained_stretch=stretch,development_stages=stages,stress_analysis=stress,
                 development_sha256=digest(PUBLIC/"a_hybrid_stress_entry.json"),rental_compute_usd=0))
    print(path, "run distillation" if not stretch else "advance hybrid")


def student_ledger():
    gate=PUBLIC/"a_distillation_trigger.json"
    if not read(gate)["proceed"]:
        raise ValueError("Distillation scientific gate was not triggered")
    init=read(PUBLIC/"a_trained_initializations.json")["initializations"]
    if set(init)!={"Distilled","OfflineOriginal"}:
        raise ValueError("Both offline controls required")
    ledger("students",["Distilled","OfflineOriginal"],gate,init=init)


def stress_student():
    stages,candidates,_=hybrids()
    result=analyze(stages+["students"],candidates+["Distilled"],CORE+["OfflineOriginal"])
    passed=result["candidates"]["Distilled"]["equal_step"]["passed"]
    path=outcome("student_stress_entry",dict(proceed=passed,analysis=result))
    if passed:
        arms=["Distilled","OfflineOriginal"]
        if not read(PUBLIC/"a_hybrid_stress_entry.json")["candidate"]:
            arms+=STRESS_CONTROLS
        init=read(PUBLIC/"a_trained_initializations.json")["initializations"]
        ledger("stress_student",arms,path,stress=True,init=init)
    print("Distilled meets development entry", passed)


def nominate():
    stages,candidates,_=hybrids()
    gate=read(PUBLIC/"a_distillation_trigger.json")
    stress_stages=[]
    stress_arms=[]
    if gate["hybrid"]:
        stress_stages.append("stress_hybrid")
        stress_arms.append(gate["hybrid"])
    controls=list(CORE)
    if gate["proceed"]:
        stages+=["students"]
        candidates+=["Distilled"]
        controls+=["OfflineOriginal"]
        if read(PUBLIC/"a_student_stress_entry.json")["proceed"]:
            stress_stages.append("stress_student")
            stress_arms.append("Distilled")
    dev=analyze(stages,candidates,controls)
    comp=STRESS_CONTROLS+(["OfflineOriginal"] if "Distilled" in stress_arms else [])
    stress=(analyze(stress_stages,stress_arms,comp,True) if stress_arms else None)
    choice=choose_a(dev,stress) if stress else None
    outcome("nomination",dict(candidate=choice,development=dev,stress=stress,registration_sha256=digest(MANIFEST),
                              stages=stages,stress_stages=stress_stages,rental_compute_usd=0))
    print("A nominee",choice)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    actions={"alternative":alternative,"stress-hybrid":stress_hybrid,"distillation-gate":trigger,
             "schedule-students":student_ledger,"stress-student":stress_student,"nominate":nominate}
    p.add_argument("action",choices=actions)
    actions[p.parse_args().action]()
