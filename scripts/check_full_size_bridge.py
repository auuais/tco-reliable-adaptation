"""Compare the only full-length original and staged executions on an identical window."""
import argparse
from pathlib import Path

import numpy as np

import evaluate_sensor_tco as scorer
from evaluation_stability import dump, sha
from score_saved_geometry import read, store_map


ROOT = Path(__file__).resolve().parents[1]
NAME = "staged_heads_bridge"
PUBLIC, RAW = ROOT / "results" / NAME, ROOT / "reports" / NAME
PLAN = ROOT / "manifests" / (NAME + ".json")
SCENE = "heads/seq-01"
ARMS = {"original": ["sensor_tco_v17", "sensor17_heads_seq-01_released_a1"],
        "staged": ["staged_sensor_factorial", "sensor17p_heads_seq-01_released_a1"]}


def protocol():
    return dict(scene=SCENE, arms=ARMS, references=list(scorer.REFS), updates=40, seed=42,
                sources={s:sha(ROOT/s) for s in ("scripts/check_full_size_bridge.py", "scripts/evaluate_sensor_tco.py",
                         "manifests/sensor_tco_v17.json", "manifests/staged_sensor_factorial.json")},
                interpretation="Compare the one full original trajectory that finished on this window with its prespecified staged counterpart. Compare instrumented adaptation time, peak allocations, and both distance errors on the released and registered references. Report normal consistency as well. This single paired execution under WSL cannot establish a general speedup, noninferiority, or a memory-only causal effect. Other original cells are unavailable. No advancement decision.")


def prepare():
    entries, maps, evidence = {}, {}, {}
    shared = None
    destination = RAW / "sealed"
    destination.mkdir(parents=True, exist_ok=False)
    for arm, (study, tag) in ARMS.items():
        folder = ROOT / "runs" / tag
        record = read(folder / "run.json")
        completed = read(ROOT / "reports" / study / (tag + ".json"))
        if (completed["status"] != "complete" or completed["record_sha256"] != sha(folder / "run.json")
                or record["manifest_sha256"] != sha(ROOT / "manifests" / (study + ".json"))
                or record["arguments"]["steps"] != 40 or record["arguments"]["seed"] != 42
                or record["arguments"]["scene"] != SCENE):
            raise ValueError("Missing matched 40-step trajectory")
        cap = read(folder / "capture.json")["records"]
        check = dict(inputs=record["evaluation_inputs_relative"], reference=record["evaluation_tensor_sha256"],
                     heads=cap["initial_predictions"]["tensor_sha256"], adapters=cap["initial_adapters"]["tensor_sha256"],
                     rng=cap["after_adapter_initialization_rng"]["sha256"], supplied_prior=record["prior_metadata"],
                     first_loss=read(folder / "losses.json")[0]["total"])
        if shared is not None and shared != check:
            raise ValueError("Different starting state or reference")
        shared = check
        points = np.load(folder / "predictions.npy", allow_pickle=False)
        maps[arm] = store_map(destination, points, SCENE, tag, entries)
        evidence[arm] = {key:record[key] for key in ("inference_and_adaptation_seconds", "model_load_seconds",
                              "peak_allocated_bytes", "peak_reserved_bytes", "research_commit")}
    dump(PUBLIC / "seal.json", dict(common={SCENE:shared}, maps={SCENE:maps}, entries=entries, arms=evidence,
                                    manifest_sha256=sha(PLAN), evaluation_source_sha256=sha(Path(scorer.__file__))))


def collect():
    seal = read(PUBLIC / "seal.json")
    rows, relative = [], {}
    for ref in scorer.REFS:
        pair = {}
        for arm in ARMS:
            row = read(RAW / "scores" / SCENE.replace("/", "_") / ref / arm / "evaluation.json")
            key = seal["maps"][SCENE][arm]
            if row["prediction_sha256"] != seal["entries"][key]["sha256"]:
                raise ValueError("Scored a different prediction")
            pair[arm] = row
            rows.append(row)
        if pair["original"]["reference_sha256"] != pair["staged"]["reference_sha256"]:
            raise ValueError("Scored different reference geometry")
        relative[ref] = {k:pair["staged"]["metrics"][k]/pair["original"]["metrics"][k]-1 for k in scorer.METRICS}
    dump(PUBLIC / "summary.json", dict(seal_sha256=sha(PUBLIC / "seal.json"), rows=rows, relative_change=relative,
                                      arms=seal["arms"], paired_executions=1, additional_cloud_usd=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "prepare", "score", "collect"))
    parser.add_argument("--reference", choices=scorer.REFS)
    args = parser.parse_args()
    if args.action == "freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN, protocol())
    elif read(PLAN) != protocol():
        raise ValueError("Matched-pair protocol changed")
    elif args.action == "prepare":
        prepare()
    elif args.action == "score":
        if args.reference is None:
            parser.error("--reference is required for score")
        scorer.PUBLIC, scorer.RAW, scorer.MANIFEST = PUBLIC, RAW, PLAN
        scorer.score(SCENE, args.reference)
    else:
        if (PUBLIC / "summary.json").exists():
            raise FileExistsError("Existing summary")
        collect()


if __name__ == "__main__":
    main()
