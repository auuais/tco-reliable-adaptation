"""Recompute the clean development decision and check every published metric row."""
import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
NAME = "sensor_extrinsic_v15"
PUBLIC = ROOT/"results"/NAME
PLAN = ROOT/"manifests"/(NAME+".json")
DISTANCES = ("accuracy_mean_m","completeness_mean_m")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    plan = read(PLAN)
    seal = read(PUBLIC/"seal.json")
    seal_hash = sha(PUBLIC/"seal.json")
    if seal["manifest_sha256"] != sha(PLAN):
        raise ValueError("Seal and protocol disagree")
    for src, digest in plan["sources"].items():
        if sha(ROOT/src) != digest:
            raise ValueError(f"Frozen source changed: {src}")
    predictions = {(r["scene"],r["method"],r["seed"]): r["sha256"] for r in seal["rows"]}
    if len(predictions)!=36:
        raise ValueError("Expected 36 predictions")
    expected = {(*key,ref) for key in predictions for ref in plan["references"]}
    observed = set()
    references = {}
    for r in rows:
        key = r["scene"],r["method"],r["seed"]
        cell = (*key,r["reference"])
        if (cell not in expected or cell in observed or r["prediction_sha256"] != predictions[key]
                or r["seal_sha256"] != seal_hash or not all(isfinite(v) for v in r["metrics"].values())):
            raise ValueError("Unknown, duplicate, changed or nonfinite evaluation")
        ref = references.setdefault((r["scene"],r["reference"]),r["reference_sha256"])
        if ref != r["reference_sha256"]:
            raise ValueError("Reference differs between methods")
        observed.add(cell)
    if observed != expected or len(set(references.values())) != 9:
        raise ValueError("Missing evaluations or indistinguishable reference conventions")
    tables, decisions = {}, {}
    released_controls = read(ROOT/"results/camera_depth_swap_v12/summary.json")["rows"]
    differences = []
    for ref in plan["references"]:
        table = {}
        for method in plan["methods"]:
            scenes = {}
            for scene in plan["scenes"]:
                zero = next(r["metrics"] for r in rows if r["reference"]==ref and r["scene"]==scene and r["method"]=="initial")
                group = [r["metrics"] for r in rows if r["reference"]==ref and r["scene"]==scene and r["method"]==method]
                metrics = {k:mean(r[k] for r in group) for k in zero}
                if ref=="released" and method in ("initial","direct"):
                    mode = "initial_initial" if method=="initial" else "prior_initial"
                    old = next(r["metrics"] for r in released_controls if r["scene"]==scene and r["mode"]==mode)
                    differences.extend(abs(metrics[k]/old[k]-1) for k in DISTANCES)
                ratios = [metrics[k]/zero[k] for k in DISTANCES]
                scenes[scene] = dict(metrics=metrics,score=mean(ratios),harmful=max(ratios)>1.01)
            table[method] = dict(score=mean(v["score"] for v in scenes.values()),
                                 harmful=sum(v["harmful"] for v in scenes.values()),scenes=scenes)
        best = min((m for m in table if m!="fitted_rgb"),key=lambda m:table[m]["score"])
        gain = 1-table["fitted_rgb"]["score"]/table[best]["score"]
        max_loss = max(table["fitted_rgb"]["scenes"][s]["score"]/table[best]["scenes"][s]["score"]-1 for s in plan["scenes"])
        harm = table["fitted_rgb"]["harmful"]<=table[best]["harmful"]
        decisions[ref] = dict(best_fixed=best,gain=gain,max_scene_relative_loss=max_loss,
                              harm_pass=harm,passed=gain>=.05 and max_loss<=.01 and harm)
        tables[ref] = table
    controls = max(differences)
    return dict(seal_sha256=seal_hash,rows=rows,tables=tables,decisions=decisions,
                max_released_control_relative_error=controls,control_pass=controls<=.005,
                advance_to_noisy_controls=controls<=.005 and all(decisions[r]["passed"] for r in ("released","registered_rgb")),
                additional_cloud_usd=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    path = PUBLIC/"summary.json"
    if args.collect:
        files = sorted((ROOT/"reports"/NAME/"scores").rglob("evaluation.json"))
        result = summarize([read(f) for f in files])
        with path.open("x",encoding="utf-8",newline="\n") as f:
            f.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    else:
        result = read(path)
        if summarize(result["rows"]) != result:
            raise ValueError("Published decision cannot be reproduced")
    print(len(result["rows"]),"evaluations, control difference",result["max_released_control_relative_error"],
          "advance",result["advance_to_noisy_controls"])


if __name__=="__main__":
    main()
