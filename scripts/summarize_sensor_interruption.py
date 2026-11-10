"""Export the full planned denominator and partial observations from an interrupted factorial."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    if not path.is_file():
        return None
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def partial_comparison(original, replication):
    if not all((p / "capture.json").exists() for p in (original, replication)):
        return None
    captures = [read(p / "capture.json")["records"] for p in (original, replication)]
    initial = {}
    for key, field in (("initial_predictions", "tensor_sha256"), ("initial_adapters", "tensor_sha256"),
                       ("after_adapter_initialization_rng", "sha256")):
        if all(key in capture for capture in captures):
            initial[key] = [capture[key][field] for capture in captures]
    records = [read(p / "run.json") for p in (original, replication)]
    traces = [read(p / "losses.json") if (p / "losses.json").is_file() else [] for p in (original, replication)]
    return dict(order=["original", "selected_readouts"], initial=initial,
                evaluation_inputs=[v.get("evaluation_tensor_sha256") for v in records],
                paired_observed_losses=[[u["total"], v["total"]] for u, v in zip(*traces)])


def summarize(study):
    manifest = ROOT / "manifests" / (study + ".json")
    plan = read(manifest)
    rows = []
    for job in plan["jobs"]:
        name = job["run_id"] + "_a1"
        folder = ROOT / "runs" / name
        supervisor = ROOT / "reports" / study / (name + ".json")
        trace = folder / "run.json"
        if not supervisor.exists() and folder.exists():
            raise RuntimeError(f"Missing terminal supervisor record for {name}")
        status = read(supervisor)["status"] if supervisor.exists() else "not_started"
        record = read(trace) if trace.exists() else {}
        losses, gradients = [read(path) if path.exists() else [] for path in
                             (folder / "losses.json", folder / "gradients.json")]
        for series in (losses, gradients):
            if [r["step"] for r in series] != list(range(len(series))):
                raise ValueError(f"Unexpected step indices in {name}")
        if not len(gradients) <= len(losses) <= plan["steps"]:
            raise ValueError(f"Unexpected number of losses or gradient records in {name}")
        if status == "complete":
            if (record.get("status") != "complete" or read(supervisor)["record_sha256"] != digest(trace)
                    or len(gradients) != plan["steps"] or not (folder / "predictions.npy").is_file()):
                raise ValueError(f"Missing completion evidence for {name}")
        rows.append(dict(**job, status=status, loss_rows=len(losses), gradient_rows=len(gradients),
                         final_prediction_sha256=digest(folder / "predictions.npy"),
                         peak_allocated_bytes=record.get("peak_allocated_bytes"),
                         peak_reserved_bytes=record.get("peak_reserved_bytes"),
                         inference_and_adaptation_seconds=record.get("inference_and_adaptation_seconds"),
                         source_run_record_sha256=digest(trace), supervisor_sha256=digest(supervisor)))
        if study == "sensor_tco_readout_replication" and folder.exists():
            original = folder.with_name(name.replace("sensor17r_", "sensor17_", 1))
            rows[-1]["partial_comparison"] = partial_comparison(original, folder)
    if not any(r["status"] == "interrupted" for r in rows):
        raise ValueError("No interrupted attempt was found")
    return dict(study=study, manifest_sha256=digest(manifest), rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["sensor_tco_v17", "sensor_tco_readout_replication"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.study), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
