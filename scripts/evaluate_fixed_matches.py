"""Offline proxy validity screen; references enter only after proxy computation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from evaluation_stability import dump,sha
from fixed_match_validation import image_digest,score_pointmaps

ROOT=Path(__file__).resolve().parents[1]


def ranks(values):
    values=np.asarray(values);result=np.empty(len(values))
    for value in np.unique(values):
        mask=values==value
        result[mask]=np.sum(values<value)+(np.sum(mask)-1)/2
    return result


def rank_correlation(a,b):
    if len(a)<3:return None
    a,b=ranks(a),ranks(b)
    if min(np.std(a),np.std(b))==0:return None
    return float(np.corrcoef(a,b)[0,1])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--evidence-root",type=Path,required=True)
    parser.add_argument("--support",type=Path,default=ROOT/"reports/fixed_matches_v6")
    parser.add_argument("--results",type=Path,default=ROOT/"results/repeat_calibration_v6")
    args=parser.parse_args()
    # Run identity/order only; no reference metrics are needed to score predictions.
    manifest=json.loads((ROOT/"manifests/repeat_calibration_v6.json").read_text())
    supports={};metadata={};rows=[]
    for job in manifest["jobs"]:
        scene=job["scene"]
        if scene not in supports:
            meta=json.loads((args.support/(scene.replace("/","_")+".json")).read_text())
            pairs=[]
            for pair in meta["pairs"]:
                assert sha(args.support/pair["file"])==pair["sha256"]
                pairs.append(dict(i=pair["i"],j=pair["j"],**dict(np.load(args.support/pair["file"],allow_pickle=False))))
            supports[scene]=pairs;metadata[scene]=meta
        path=args.evidence_root/"runs"/job["run_id"]
        record=json.loads((path/"run.json").read_text());assert record["status"]=="complete"
        with np.load(args.evidence_root/record["evaluation_inputs_relative"],allow_pickle=False) as source:
            assert image_digest(source["images"])==metadata[scene]["image_sha256"]
        started=time.perf_counter()
        score=score_pointmaps(np.load(path/"predictions.npy",allow_pickle=False),supports[scene])
        rows.append(dict(run_id=job["run_id"],scene=scene,condition=job["condition"],calibration=job["calibration"],
                         seed=job["seed"],execution_repeat=job["execution_repeat"],proxy=score,
                         scoring_seconds=time.perf_counter()-started,prediction_sha256=sha(path/"predictions.npy")))
    # Freeze all proxy scores before joining any evaluation label.
    raw=args.results/"fixed_match_proxy_scores.json"
    assert not raw.exists(),"Never overwrite the initial proxy screen"
    dump(raw,dict(rows=rows,support_plan_sha256=sha(args.support/"plan.json"),
                  scorer_sha256=sha(ROOT/"scripts/fixed_match_validation.py"),reference_used_for_proxy=False))
    primary=json.loads((args.results/"summary.json").read_text());assert primary["all_evidence_validated"]
    labels={r["run_id"]:r for r in primary["rows"]}
    zero={r["scene"]:r for r in rows if r["condition"]=="zero"}
    summaries=[]
    for scene in supports:
        selected=[r for r in rows if r["scene"]==scene and r["condition"]!="zero"]
        for row in selected:
            base=zero[scene]["proxy"]
            row["would_accept"]=row["proxy"]["eligible"] and base["eligible"] and all(
                old["score"]>0 and new["score"]<=.98*old["score"] for new,old in zip(row["proxy"]["folds"],base["folds"]))
            row["reference_score"]=labels[row["run_id"]]["score"]
            row["reference_harmful"]=labels[row["run_id"]]["harmful"]
        eligible=[r for r in selected if r["proxy"]["eligible"]]
        accepted=[r for r in selected if r["would_accept"]]
        summaries.append(dict(scene=scene,adapted_runs=len(selected),eligible=len(eligible),abstentions=len(selected)-len(eligible),
            would_accept=len(accepted),harmful_among_accepted=sum(r["reference_harmful"] for r in accepted),
            harmful_among_all=sum(r["reference_harmful"] for r in selected),
            rank_correlation_with_reference_error=rank_correlation([r["proxy"]["score"] for r in eligible],[r["reference_score"] for r in eligible])))
    dump(args.results/"fixed_match_proxy_validity.json",dict(scenes=summaries,rows=rows,
        interpretation="Exploratory conventional consistency baseline; correlated development executions, no significance or generalization claim. Eligibility/2% rule were fixed before v6 outcomes. Abstention is not evidence of a useful controller.",
        original_proxy_file_sha256=sha(raw),held_out_scenes_used=False))
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":main()
