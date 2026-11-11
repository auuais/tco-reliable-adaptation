"""Bound any whole-scene selector restricted to the tested camera-prior interventions."""
import argparse
from itertools import product
from statistics import mean

import camera_tangent_depth as study
from evaluation_stability import dump,sha
from prepare_source_spacing import PUBLIC,SCENES
from score_saved_geometry import read

RESULT=PUBLIC/"camera_prior_selection_bound.json"
OPTIONS=("direct","joint","prior_adaptive","equal","full_prior",*("gain_"+a for a in study.controls.ARMS),
         "surface_equal","surface_full_prior","guarded")


def calculate():
    parent=read(study.RESULT)["results"]
    reg=parent["references"]["registered_rgb"]
    released=parent["references"]["released"]
    best,hist=parent["local_comparator"],parent["historical_comparator"]
    b=reg["overall"][best]
    threshold=.95*min(b["score"],reg["overall"][hist]["score"])
    winning=None
    safe_winning=None
    passed=0
    count=0
    for assignment in product(OPTIONS,repeat=len(SCENES)):
        chosen=[reg["scenes"][s][a] for s,a in zip(SCENES,assignment)]
        score=mean(v["score"] for v in chosen)
        released_score=mean(released["scenes"][s][a]["score"] for s,a in zip(SCENES,assignment))
        safe=(all(v["score"]<=reg["scenes"][s][best]["score"] for s,v in zip(SCENES,chosen))
              and all(mean(v[m] for v in chosen)<=1.02*b[m] for m in ("relative_accuracy","relative_completeness"))
              and released_score<=released["overall"][best]["score"])
        item=dict(choices=dict(zip(SCENES,assignment)),score=score,released_score=released_score)
        if winning is None or score<winning["score"]: winning=item
        if safe and (safe_winning is None or score<safe_winning["score"]): safe_winning=item
        passed+=int(safe and score<=threshold)
        count+=1
    return dict(local=best,historical=hist,required_score=threshold,assignments=count,passing=passed,
                best_ignoring_safety=winning,best_with_safety=safe_winning)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("measure","verify"))
    deps=sha(study.RESULT)
    if p.parse_args().action=="measure":
        if RESULT.exists(): raise FileExistsError(RESULT)
        dump(RESULT,dict(source=deps,options=OPTIONS,results=calculate()))
    report=read(RESULT)
    if deps!=report["source"] or list(OPTIONS)!=report["options"] or calculate()!=report["results"]:
        raise ValueError("Selection upper bound differs")
    print(report["results"])
