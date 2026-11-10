"""Apply the unchanged v6 proxy rule to frozen learned support, then join labels."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from evaluation_stability import dump,sha
import evaluate_fixed_matches as original

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--evidence-root",type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((ROOT/"manifests/learned_support_v8.json").read_text())
    for name,digest in manifest["sources"].items():assert sha(ROOT/name)==digest,name
    public=ROOT/"results/learned_support_v8";private=ROOT/"reports/learned_support_v8_scoring";private.mkdir(parents=True,exist_ok=False)
    dump(private/"plan.json",dict(wrapper_sha256=sha(Path(__file__)),scorer_sha256=sha(ROOT/"scripts/evaluate_fixed_matches.py"),
        support_manifest_sha256=sha(ROOT/"manifests/learned_support_v8.json"),support_summary_sha256=sha(public/"support.json"),
        analysis="All 40 saved v6 predictions; unchanged eligibility and two-fold 2% rule; exploratory post-v6 redesign",
        no_outcome_threshold_selection=True))
    shutil.copyfile(ROOT/"results/repeat_calibration_v6/summary.json",private/"summary.json")
    sys.argv=[sys.argv[0],"--evidence-root",str(args.evidence_root),"--support",str(ROOT/"reports/learned_support_v8"),"--results",str(private)]
    original.main()
    # Preserve the original wrapper outputs privately. Public interpretation identifies the later support study.
    for name in ["fixed_match_proxy_scores.json","fixed_match_proxy_validity.json"]:
        assert not (public/name).exists()
        content=json.loads((private/name).read_text())
        if name.endswith("validity.json"):
            content["interpretation"]="Exploratory learned-matcher baseline designed after v6 labels were inspected; same earlier eligibility/2% rule, new frozen image-only support. No held-out validity, significance, novel method or automatic spending release."
            content["analysis_plan_sha256"]=sha(private/"plan.json")
        if name.endswith("scores.json"):
            # Keep exact bytes so the validity output's frozen-proxy hash remains valid.
            shutil.copyfile(private/name,public/name)
        else:dump(public/name,content)


if __name__=="__main__":main()
