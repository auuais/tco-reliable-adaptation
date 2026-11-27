"""Independent numpy replay of the two Stanford comparisons from public per-seed scores."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np

from tco_final_ln_protocol import PUBLIC, read, seal, sha


FIELDS = ("accuracy_mm", "completeness_mm", "scene_seconds")
SEEDS = [42,43,44]


def audit(folder, stage):
    files = [folder/f"{stage}_{suffix}.json" for suffix in ("rows","analysis_plan","decision")]
    artifact, plan, official = [read(p) for p in files]
    candidate, short, full = official["arms"]
    permitted = {candidate,short,full}
    selected = [r for r in artifact["rows"] if r["arm"] in permitted and r["status"]=="completed"
                and r["role"] not in {"duplicate","engineering","anchor","diagnostics"}]
    keys = Counter((r["room"],r["arm"],r["seed"]) for r in selected)
    assert all(v==1 for v in keys.values()), "Multiple scored endpoints for a paired slot"
    entries={(r["room"],r["arm"],r["seed"]):r for r in selected}
    rooms = sorted(r["room"] for r in plan["rooms"] if
                   all((r["room"],a,s) in entries for a in permitted for s in SEEDS))
    assert rooms==official["complete_rooms"] and len(selected)==official["scored"], "Different coverage"
    if not rooms:
        return dict(coverage=False,judgment="no_complete_rooms",files={p.name:sha(p) for p in files})
    groups = {}
    environments = {r["room"]:r["environment"] for r in plan["rooms"]}
    for index, room in enumerate(rooms):
        key=environments[room] if plan["dataset"]=="stanford12" else "all_rooms"
        groups.setdefault(key,[]).append(index)
    n, draws=len(rooms),10_000
    # Generate paired room/seed draw indices independently; fold the two inner orderings into a base-three code.
    rng=random.Random(20261128)
    sampled_rooms=np.empty((draws,n),dtype=np.int64)
    codes=np.empty_like(sampled_rooms)
    for d in range(draws):
        col=0
        for group in groups.values():
            for _ in group:
                sampled_rooms[d,col]=group[rng.randrange(len(group))]
                codes[d,col]=rng.randrange(27)
                col+=1
    seed_draws=np.stack((codes//9,(codes//3)%3,codes%3),axis=-1)
    def cube(arm):
        return np.array([[[entries[room,arm,seed][k] for k in FIELDS] for seed in SEEDS] for room in rooms])
    def replicate(values):
        return np.median(values[sampled_rooms[:,:,None],seed_draws],axis=2)
    numerator=cube(candidate)
    boot_num=replicate(numerator)
    statistics={}
    largest=0.
    for ref in (short,full):
        den=cube(ref)
        boot=(boot_num/replicate(den)).mean(axis=1)
        q=np.quantile(boot,[.05,.95],axis=0)
        point=(np.median(numerator,axis=1)/np.median(den,axis=1)).mean(axis=0)
        statistics[ref]={}
        for i,metric in enumerate(FIELDS):
            got=np.array([point[i],q[0,i],q[1,i]])
            expected=np.array([official["comparisons"][ref][metric][k] for k in ("ratio","lower_95","upper_95")])
            err=float(np.max(np.abs(got-expected)))
            if err>1e-11:
                raise ValueError(f"Independent paired bootstrap differs for {ref} {metric} by {err}")
            largest=max(largest,err)
            statistics[ref][metric]=dict(estimate=float(got[0]),lower=float(got[1]),upper=float(got[2]))
    req=((short,"accuracy_mm",1.,True),(short,"completeness_mm",1.03,False),(full,"accuracy_mm",1.01,False),
         (full,"completeness_mm",1.03,False),(full,"scene_seconds",1.,True))
    passed=[]
    for ref,metric,limit,strict in req:
        st=statistics[ref][metric]
        value=max(st["estimate"],st["upper"])
        passed.append(value<limit if strict else value<=limit)
    coverage=len(rooms)==len(plan["rooms"]) and len(rooms)>=6 and plan["dataset"]!="development"
    assert official["formal_coverage"] == coverage
    if coverage:
        assert (official["joint_verdict"]=="pass") == all(passed)
    return dict(dataset=plan["dataset"],scored=len(selected),expected=official["expected"],
                complete_rooms=rooms,formal_coverage=coverage,max_absolute_ratio_difference=largest,
                numerical_tolerance=1e-11,surface_and_time_stats=statistics,pass_indicators=passed,
                official_verdict=official["joint_verdict"],verified=True,
                files={p.name:sha(p) for p in files})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",default="primary",choices=("primary","development","followup","replication"))
    parser.add_argument("--folder",type=Path,default=PUBLIC)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    result=audit(args.folder,args.stage)
    if args.out:
        seal(args.out,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
