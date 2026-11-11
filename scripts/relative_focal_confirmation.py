"""Test camera-relative agreement with a fixed geometric focal correction."""
import argparse
import hashlib
import json
from statistics import mean

import numpy as np

from focal_metadata_control import focal_proposal
from focal_zoom_observations import ROOT, PUBLIC as PARENT
from measure_focal_crop_response import focal_xy
from score_saved_geometry import read
from visual_focal_baseline import PUBLIC as SOLVER


OUT = ROOT / "results/relative_focal_confirmation"
PROTOCOL = ROOT / "docs/RELATIVE_FOCAL_CONFIRMATION.md"


def confirmation(supplied, model, matched):
    if supplied.shape != (5, 2) or model.shape != supplied.shape or not np.isfinite([supplied, model]).all():
        raise ValueError("Expected five finite pairs of focal coordinates")
    if np.min([supplied, model]) <= 0:
        raise ValueError("Focal lengths must be positive")
    k = np.repeat(np.eye(3)[None], len(supplied), axis=0)
    k[:, [0, 1], [0, 1]] = supplied
    view, proposed = focal_proposal(k)
    if view is None:
        return None, dict(raw=False, relative=False, raw_with_folds=False, relative_with_folds=False)
    others = np.arange(len(k)) != view
    bias = np.median(np.log(model[others] / supplied[others]), axis=0)
    observed = np.log(supplied[view])
    replacement = np.log(proposed[[0, 1], [0, 1]])
    def choose(prediction):
        return bool(np.abs(prediction - replacement).mean() < np.abs(prediction - observed).mean())
    predicted = np.log(model[view])
    raw, relative = choose(predicted), choose(predicted - bias)
    return view, dict(raw=raw, relative=relative, raw_with_folds=raw and matched,
                      relative_with_folds=relative and matched)


def aggregate(rows):
    def reduce(group):
        states = {}
        for method in group[0]["edited"]:
            errors = [r["errors"][method] for r in group]
            states[method] = dict(
                focal_gradient=mean(e["focal_gradient_relative"] for e in errors),
                focal_log_error=mean(e["log_focal_error"] for e in errors),
                faults_corrected=sum(bool(r["edited"][method]) for r in group if r["case"] == "metadata_fault"),
                valid_edits=sum(bool(r["edited"][method]) for r in group if r["case"] == "valid_zoom"))
        return dict(n=len(group), methods=states,
                    classical_gradients={m:mean(r["classical_gradients"][m] for r in group)
                                         for m in group[0]["classical_gradients"]})
    visible = [r for r in rows if r["case"] in ("valid_zoom", "metadata_fault")]
    mild = [r for r in visible if r["noise"] == "mild"]
    if len(rows) != 465 or len(mild) != 120:
        raise ValueError("Missing paired evidence")
    primary = reduce(mild)
    scenes = {s:reduce([r for r in mild if r["scene"] == s]) for s in sorted({r["scene"] for r in rows})}
    def gain(group):
        data = group["methods"]
        base = min(data[m]["focal_gradient"] for m in ("retain", "consensus"))
        return 1 - data["relative_with_folds"]["focal_gradient"] / base
    m = primary["methods"]
    score = m["relative_with_folds"]["focal_gradient"]
    checks = dict(valid_edits=m["relative_with_folds"]["valid_edits"] <= 3,
                  gain=gain(primary) >= .2, scenes=all(gain(g) >= 0 for g in scenes.values()),
                  raw_head=score <= m["raw"]["focal_gradient"],
                  raw_with_folds=score <= m["raw_with_folds"]["focal_gradient"])
    return dict(primary=primary, gain=gain(primary), checks=checks, supports_new_windows=all(checks.values()),
                by_scene=scenes, by_noise={n:reduce([r for r in visible if r["noise"] == n])
                                           for n in ("clean", "mild", "strong")},
                controls={n:reduce([r for r in rows if r["case"] == n]) for n in ("hidden_zoom", "uniform_clean")})


def analyze():
    parent = read(PARENT / "summary.json")
    classical = read(SOLVER / "summary.json")
    observations = read(PARENT / "observations.json")
    images = {(o["camera"]["scene"], o["camera"]["batch"]):
              (o["camera"]["rgb_sha256"], focal_xy(o["camera"])) for o in observations}
    rows = []
    keys = ("scene", "batch", "case", "view", "direction", "noise", "seed")
    for r, c in zip(parent["rows"], classical["rows"]):
        if any(r[k] != c[k] for k in keys):
            raise ValueError("Unpaired solver comparison")
        image_digest, model = images[r["scene"], r["batch"]]
        if image_digest != r["rgb_sha256"]:
            raise ValueError("Camera output provenance mismatch")
        # The source rigs have four identical neighbor focals and at most one supplied outlier.
        supplied = np.tile(r["consensus_focal"], (5, 1))
        if r["view"] is not None:
            supplied[r["view"]] = r["supplied_focal"]
        view, edits = confirmation(supplied, model, r["edits"]["both_folds"])
        if (view is not None and view != r["view"]) or edits["raw"] != r["edits"]["model_nearest"]:
            raise ValueError("Could not reconstruct the original raw-head comparator")
        edits.update({m:r["edits"][m] for m in ("retain", "consensus", "both_folds")})
        metrics = {m:r["errors"]["consensus" if edit else "retain"] for m, edit in edits.items()}
        rows.append(dict(**{k:r[k] for k in keys}, edited=edits, errors=metrics,
                         classical_gradients={m:c["errors"][m]["focal_gradient_relative"]
                                              for m in ("iterative_continuous", "iterative_nearest")}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = analyze()
    result = aggregate(rows)
    parents = [PROTOCOL, PARENT/"summary.json", PARENT/"observations.json", SOLVER/"summary.json"]
    # The exact gradient arrays and per-choice errors already live in the parent evidence.
    decisions = [{k:v for k,v in row.items() if k not in ("errors", "classical_gradients")} for row in rows]
    evidence = dict(inputs={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in parents},
                    rows=decisions, results=result, additional_cloud_usd=0)
    path = OUT / "summary.json"
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(evidence, f, indent=2, allow_nan=False)
            f.write("\n")
    elif read(path) != evidence:
        raise ValueError("Relative focal results failed replay")
    print(json.dumps(dict(gain=result["gain"], checks=result["checks"], primary=result["primary"])))


if __name__ == "__main__":
    main()
