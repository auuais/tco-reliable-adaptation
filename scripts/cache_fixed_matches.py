"""Prepare fixed image-only support locally, before v6 outcomes are inspected."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from evaluation_stability import dump, sha
from fixed_match_validation import build_support

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"reports/fixed_matches_v6")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    dump(args.output/"plan.json",dict(sources={name:sha(ROOT/"scripts"/name) for name in ["cache_fixed_matches.py","fixed_match_validation.py"]},
        input_scope="Only resized RGB images from the three already-used development scenes; no reference masks, cameras or depths",
        eligibility="Each of two component-disjoint folds must connect all five frames and contain >=100 matches; >=10 per eligible pair",
        score="Equal-pair mean of median matched-3D distance, divided by current pointmap median pairwise anchor distance",
        planned_analysis="All 40 v6 predictions; report coverage/abstention, within-scene rank association and harmful-update detection; no fit or threshold sweep",
        provisional_acceptance="Both fold scores improve >=2% against fresh zero; exploratory offline diagnostic, not a deployed controller",
        warning="Cross-view consistency cannot certify shape accuracy. Collapsed scale and insufficient support abstain. No metric-scale claim."))
    summaries=[]
    for scene in ["heads/seq-01","chess/seq-03","fire/seq-03"]:
        key=scene.replace("/","_")
        record=json.loads((args.evidence_root/"runs"/("prior5_"+key+"_zero_seed42")/"run.json").read_text())
        with np.load(args.evidence_root/record["evaluation_inputs_relative"],allow_pickle=False) as source:
            images=source["images"]
        started=time.perf_counter();pairs,metadata=build_support(images)
        entries=[]
        for index,pair in enumerate(pairs):
            name=f"{key}_pair{index}.npz"
            np.savez_compressed(args.output/name,xy_i=pair["xy_i"],xy_j=pair["xy_j"],fold=pair["fold"])
            entries.append(dict(i=pair["i"],j=pair["j"],file=name,sha256=sha(args.output/name),
                matches=len(pair["fold"]),fold_counts=[int(np.sum(pair["fold"]==i)) for i in [0,1]]))
        metadata.update(scene=scene,image_shape=list(images.shape),matching_seconds=time.perf_counter()-started,pairs=entries)
        dump(args.output/(key+".json"),metadata);summaries.append(metadata)
        print(json.dumps(dict(scene=scene,pairs=len(pairs),matches=sum(e["matches"] for e in entries),seconds=metadata["matching_seconds"])),flush=True)
    dump(ROOT/"results/fixed_match_v6_preflight/support.json",dict(scenes=summaries,source_plan_sha256=sha(args.output/"plan.json"),
        scope="Exploratory preparation only; no reference evaluation labels or adaptation outcomes inspected for support selection"))


if __name__=="__main__":main()
