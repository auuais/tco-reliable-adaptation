"""Export a complete local run ledger; never infer missing experiment metrics."""
import csv
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
rows = []
for path in sorted((root / "runs").glob("*/run.json")):
    r = json.loads(path.read_text())
    args = r["arguments"]
    metrics = r.get("metrics", {})
    rows.append({"run_id": r["run_id"], "status": r["status"], "scene": args["scene"],
                 "size": args["size"], "views": len(args["frames"]), "steps": args["steps"],
                 "accuracy_cm": metrics.get("accuracy_mean_m", 0) * 100 if metrics else "",
                 "completeness_cm": metrics.get("completeness_mean_m", 0) * 100 if metrics else "",
                 "normal_consistency": metrics.get("normal_consistency_mean", ""),
                 "inference_adaptation_seconds": r.get("inference_and_adaptation_seconds", ""),
                 "peak_allocated_GiB": r["peak_allocated_bytes"] / 2**30 if "peak_allocated_bytes" in r else "",
                 "source": path.relative_to(root).as_posix()})
out = root / "reports/run_ledger.csv"
with out.open("w", newline="") as stream:
    fields = list(rows[0]) if rows else ["run_id", "status", "scene", "size", "views", "steps", "accuracy_cm", "completeness_cm", "normal_consistency", "inference_adaptation_seconds", "peak_allocated_GiB", "source"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
