"""Validate paired evidence before producing a fixed-subset summary.

Never replace missing results with zero or silently average a smaller subset.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ["accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean"]


def validated_pairs(runs, manifest, prefix="cloud"):
    pairs, missing = [], []
    for seq, frames in manifest.items():
        records = []
        for steps in [0, 40]:
            run_id = f"{prefix}_{seq.replace('/', '_')}_518_s{steps}_seed42"
            path = runs / run_id / "run.json"
            record = json.loads(path.read_text()) if path.exists() else None
            if record is None or record.get("status") != "complete":
                missing.append({"run_id": run_id, "status": record.get("status") if record else "missing"})
                continue
            for key, expected in {"scene": seq, "frames": frames, "size": 518, "steps": steps, "seed": 42}.items():
                if record["arguments"][key] != expected:
                    raise ValueError(f"Protocol mismatch: {run_id}: {key}")
            if record["base_parameters_modified"]:
                raise ValueError(f"Base weights changed: {run_id}")
            for key in METRICS:
                value = record["metrics"][key]
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Missing/nonfinite metric: {run_id}: {key}")
            records.append(record)
        if len(records) != 2:
            continue
        baseline, adapted = records
        for key in ["input_sha256", "reference_sha256", "upstream_commit", "upstream_method_sha256",
                    "gpu", "versions", "prior_provenance", "evaluation_only"]:
            if baseline[key] != adapted[key]:
                raise ValueError(f"Unmatched pair for {seq}: {key}")
        if baseline["checkpoint_download"]["sha256"] != adapted["checkpoint_download"]["sha256"]:
            raise ValueError(f"Checkpoint mismatch: {seq}")
        if baseline.get("gsplat_backend") != adapted.get("gsplat_backend"):
            raise ValueError(f"Renderer binary mismatch: {seq}")
        if baseline["metrics"]["protocol"] != adapted["metrics"]["protocol"]:
            raise ValueError(f"Evaluation protocol mismatch: {seq}")
        cfg0, cfg1 = [dict(r["configuration"]) for r in records]
        cfg0.pop("tco_steps")
        cfg1.pop("tco_steps")
        if cfg0 != cfg1:
            raise ValueError(f"Configuration mismatch: {seq}")
        pairs.append((seq, baseline, adapted))
    return pairs, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=pathlib.Path, default=ROOT / "runs")
    parser.add_argument("--manifest", type=pathlib.Path, default=ROOT / "manifests/midterm_3scene_kf200.json")
    parser.add_argument("--prefix", default="cloud")
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "reports/cloud_comparison")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    pairs, missing = validated_pairs(args.runs, manifest, args.prefix)
    result = {"status": "incomplete" if missing else "complete", "planned_pairs": len(manifest),
              "completed_pairs": len(pairs), "missing_or_failed": missing,
              "scope": "Fixed five-sequence public subset; not the full paper benchmark",
              "aggregation": "Equal weight per sequence; no partial-subset mean"}
    rows = []
    for seq, baseline, adapted in pairs:
        for label, record in [("VGGT", baseline), ("TCO", adapted)]:
            rows.append({"sequence": seq, "method": label, "run_id": record["run_id"],
                         **{key: record["metrics"][key] for key in METRICS},
                         "seconds": record["inference_and_adaptation_seconds"],
                         "peak_allocated_GiB": record["peak_allocated_bytes"] / 2**30})
    if not missing:
        if len({r["gpu"] for _, b, a in pairs for r in [b, a]}) != 1:
            raise ValueError("Mixed hardware in the fixed experiment")
        result["mean_metrics"] = {method: {key: statistics.mean(row[key] for row in rows if row["method"] == method)
                                          for key in METRICS} for method in ["VGGT", "TCO"]}
        result["gpu"] = pairs[0][1]["gpu"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    fields = ["sequence", "method", "run_id", *METRICS, "seconds", "peak_allocated_GiB"]
    with (args.output / "per_sequence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
