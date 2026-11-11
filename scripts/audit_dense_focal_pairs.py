"""Measure same-target repeatability in the 24 adaptive/control pairs after scoring."""
import argparse
from statistics import mean, median

import numpy as np

from evaluation_stability import dump, sha
from run_dense_focal_pilot import PLAN, RUNS
from score_dense_focal_pilot import METRICS, PUBLIC, REFS
from score_saved_geometry import read


def pair_key(job):
    return job["input_key"], job["prior_key"], job["seed"]


def inspect(row, runs):
    folder = runs / row["tag"]
    if sha(folder / "run.json") != row["run_sha256"]:
        raise ValueError(f"Run record differs for {folder.name}")
    run = read(folder / "run.json")
    if sha(folder / "capture.json") != run["capture_sha256"]:
        raise ValueError(f"Capture differs for {folder.name}")
    captures = read(folder / "capture.json")["records"]
    return run, captures


def compare(adaptive, matched, runs=RUNS):
    a, a_capture = inspect(adaptive, runs)
    b, b_capture = inspect(matched, runs)
    for field in ("used_focal", "supplied_focal"):
        np.testing.assert_array_equal(a["decision"][field], b["decision"][field])
    if a["configuration"] != b["configuration"] or a["checkpoint_sha256"] != b["checkpoint_sha256"]:
        raise ValueError("Matched executions used different configurations or checkpoints")
    if set(a_capture) != set(b_capture):
        raise ValueError("Matched evidence has different capture labels")
    same = {label: a[label]["tensor_sha256"] == b[label]["tensor_sha256"] for label in ("initial", "final")}
    for label in a_capture:
        digest = "tensor_sha256" if "tensor_sha256" in a_capture[label] else "sha256"
        same[label] = a_capture[label][digest] == b_capture[label][digest]
    signed = {}
    for ref in REFS:
        a_score = adaptive["scores"][ref]
        b_score = matched["scores"][ref]
        signed[ref] = {m: a_score["final"][m] - b_score["final"][m] for m in METRICS}
        signed[ref]["normalized_score"] = mean(
            a_score["final"][m] / a_score["initial"][m] - b_score["final"][m] / b_score["initial"][m]
            for m in METRICS[:2]
        )
    return dict(adaptive_tag=adaptive["tag"], matched_tag=matched["tag"],
                matched_arm=matched["job"]["arm"], identical=same, adaptive_minus_matched=signed)


def audit(runs=RUNS):
    summary_path = PUBLIC / "summary.json"
    rows = read(summary_path)["rows"]
    if [r["job"] for r in rows] != read(PLAN)["jobs"]:
        raise ValueError("Unexpected dense grid")
    arms = {(pair_key(r["job"]), r["job"]["arm"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["job"]["arm"] == "adaptive":
            arm = "consensus" if row["edited"] else "retain"
            pairs.append(compare(row, arms[pair_key(row["job"]), arm], runs))
    if len(pairs) != 24 or len({p["matched_tag"] for p in pairs}) != 24:
        raise ValueError("Expected 24 distinct paired executions")
    repeatability = {}
    for ref in REFS:
        deltas = [p["adaptive_minus_matched"][ref]["normalized_score"] for p in pairs]
        repeatability[ref] = dict(mean_signed=mean(deltas), median_absolute=median(map(abs, deltas)),
                                  largest_absolute=max(map(abs, deltas)))
    return dict(summary_sha256=sha(summary_path), manifest_sha256=sha(PLAN),
                identical_pairs={k: sum(p["identical"][k] for p in pairs) for k in pairs[0]["identical"]},
                normalized_score_difference=repeatability, pairs=pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "verify"))
    args = parser.parse_args()
    result = audit()
    path = PUBLIC / "same_target_pairs.json"
    if args.action == "verify":
        if read(path) != result:
            raise ValueError("Published same-target audit differs from the run records")
    elif path.exists():
        raise FileExistsError(path)
    else:
        dump(path, result)
    print("Verified 24 matched focal-target pairs against their unconditioned control runs.")
