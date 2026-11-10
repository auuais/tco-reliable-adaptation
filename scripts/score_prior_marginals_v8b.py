"""Exploratory whole-trajectory prior contrasts; not causal one-step proposals."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump,sha
from evaluate_fixed_matches import rank_correlation
from fixed_match_validation import image_digest,score_pointmaps

ROOT=Path(__file__).resolve().parents[1]
STUDY="prior_marginal_proxy_v8b"
FAMILIES={"add_pose_to_visual":("visual","pose_only"),"add_intrinsics_to_visual":("visual","intrinsics_only"),
          "add_pose_to_intrinsics":("intrinsics_only","both"),"add_intrinsics_to_pose":("pose_only","both")}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--evidence-root",type=Path);args=parser.parse_args()
    manifest=ROOT/"manifests"/(STUDY+".json")
    plan=dict(study_id=STUDY,sources={name:sha(ROOT/name) for name in ["scripts/score_prior_marginals_v8b.py","scripts/fixed_match_validation.py","scripts/evaluate_fixed_matches.py"]},
              prior_manifest_sha256=sha(ROOT/"manifests/prior_ablation_v5.json"),support_manifest_sha256=sha(ROOT/"manifests/learned_support_v8.json"),
              support_summary_sha256=sha(ROOT/"results/learned_support_v8/support.json"),families=FAMILIES,planned_predictions=39,planned_contrasts=36,
              hypothesis="Compare the fixed visual proxy's whole-trajectory marginal ranking with offline reference differences",
              selection_rule="Favor addition only if both eligible fold scores are at least 2% lower than the without-group endpoint",
              labels="Report signed normalized-score difference and >1% accuracy OR completeness regression relative to without-group; do not tune thresholds",
              limitations="v5 labels already inspected; matched scene/seed but separate stochastic trajectories, not shared-gradient one-step causality. Report all four families separately. No held-out claim or paid-stage release.")
    # JSON turns tuples into lists, so normalize before exact comparison.
    plan=json.loads(json.dumps(plan))
    if args.freeze:
        assert not manifest.exists();dump(manifest,plan)
    assert json.loads(manifest.read_text())==plan
    if args.freeze or args.dry_run:print(json.dumps(dict(study=STUDY,manifest_sha256=sha(manifest),predictions=39,contrasts=36)));return
    assert args.evidence_root is not None
    out=ROOT/"results"/STUDY;out.mkdir(parents=True,exist_ok=False)
    jobs=json.loads((ROOT/"manifests/prior_ablation_v5.json").read_text())["jobs"]
    support_dir=ROOT/"reports/learned_support_v8";support={};metadata={};rows=[]
    for job in jobs:
        scene=job["scene"]
        if scene not in support:
            meta=json.loads((support_dir/(scene.replace("/","_")+".json")).read_text());pairs=[]
            for item in meta["pairs"]:
                assert sha(support_dir/item["file"])==item["sha256"]
                pairs.append(dict(i=item["i"],j=item["j"],**dict(np.load(support_dir/item["file"],allow_pickle=False))))
            support[scene]=pairs;metadata[scene]=meta
        folder=args.evidence_root/"runs"/job["run_id"];record=json.loads((folder/"run.json").read_text())
        assert record["status"]=="complete"
        with np.load(args.evidence_root/record["evaluation_inputs_relative"],allow_pickle=False) as inputs:
            assert image_digest(inputs["images"])==metadata[scene]["image_sha256"]
        prediction=np.load(folder/"predictions.npy",allow_pickle=False)
        rows.append(dict(run_id=job["run_id"],scene=scene,condition=job["condition"],seed=job["seed"],
                         proxy=score_pointmaps(prediction,support[scene]),prediction_sha256=sha(folder/"predictions.npy")))
    # Freeze every proxy outcome before any reference-metric join.
    dump(out/"proxy_scores.json",dict(rows=rows,manifest_sha256=sha(manifest),reference_used_for_proxy=False))
    lookup={}
    for row in rows:
        metrics=json.loads((args.evidence_root/"runs"/row["run_id"]/"metrics.json").read_text())
        row["metrics"]={k:metrics[k] for k in ["accuracy_mean_m","completeness_mean_m"]}
        lookup[(row["scene"],row["seed"],row["condition"])]=row
    for row in rows:
        base=lookup[(row["scene"],42,"zero")]["metrics"]
        row["reference_score"]=float(np.mean([row["metrics"][k]/base[k] for k in base]))
    contrasts=[]
    for family,(without,with_group) in FAMILIES.items():
        for scene in support:
            for seed in [42,123,2026]:
                a,b=lookup[(scene,seed,without)],lookup[(scene,seed,with_group)]
                eligible=a["proxy"]["eligible"] and b["proxy"]["eligible"]
                proxy_delta=(b["proxy"]["score"]/a["proxy"]["score"]-1) if eligible and a["proxy"]["score"]>0 else None
                favor=eligible and all(old["score"]>0 and new["score"]<=.98*old["score"] for old,new in zip(a["proxy"]["folds"],b["proxy"]["folds"]))
                reference_delta=b["reference_score"]-a["reference_score"]
                harmful=any(b["metrics"][k]>1.01*a["metrics"][k] for k in a["metrics"])
                contrasts.append(dict(family=family,scene=scene,seed=seed,without_run=a["run_id"],with_run=b["run_id"],eligible=eligible,
                    proxy_relative_change=proxy_delta,reference_score_change=reference_delta,reference_relative_change=b["reference_score"]/a["reference_score"]-1,
                    would_favor_addition=favor,with_group_harmful_vs_without=harmful,
                    strict_sign_agreement=(bool(np.sign(proxy_delta)==np.sign(reference_delta)) if proxy_delta is not None else None)))
    summaries=[]
    for family in FAMILIES:
        group=[r for r in contrasts if r["family"]==family];eligible=[r for r in group if r["eligible"]];favored=[r for r in group if r["would_favor_addition"]]
        summaries.append(dict(family=family,contrasts=len(group),eligible=len(eligible),strict_sign_agreement=sum(r["strict_sign_agreement"] for r in eligible),
            favored=len(favored),harmful_among_favored=sum(r["with_group_harmful_vs_without"] for r in favored),
            harmful_if_always_add=sum(r["with_group_harmful_vs_without"] for r in group),
            mean_reference_score_change_if_rule=float(np.mean([r["reference_score_change"] if r["would_favor_addition"] else 0 for r in group])),
            mean_reference_score_change_if_always_add=float(np.mean([r["reference_score_change"] for r in group])),
            rank_association=rank_correlation([r["proxy_relative_change"] for r in eligible],[r["reference_relative_change"] for r in eligible])))
    dump(out/"summary.json",dict(study_id=STUDY,manifest_sha256=sha(manifest),proxy_file_sha256=sha(out/"proxy_scores.json"),
        planned_predictions=39,completed_predictions=len(rows),planned_contrasts=36,completed_contrasts=len(contrasts),families=summaries,contrasts=contrasts,rows=rows,
        interpretation=plan["limitations"],later_allocations_released=False))
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":main()
