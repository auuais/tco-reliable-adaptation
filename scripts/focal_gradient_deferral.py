"""Measure guaranteed shared update prefixes using recorded full-model focal traces."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import tarfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "results/canonical_sensor_factorial"
OUT = ROOT / "results/focal_gradient_deferral"
PROTOCOL = ROOT / "docs/FOCAL_GRADIENT_DEFERRAL.md"
SCALES = (1.05, 1.12, 1.25)
FIELDS = ("predicted_focal_xy_pixels", "target_focal_xy_pixels", "residual_xy_pixels",
          "smooth_l1_derivative_before_reduction")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def read(path):
    return json.loads(path.read_bytes())


def extract(archives):
    runs = read(PARENT / "seal.json")["runs"]
    hashes = read(PARENT / "session.json")["raw_archives"]
    traces, provenance = [], []
    for run in runs:
        attempt, = run["attempts"]
        if attempt["status"] != "complete":
            raise ValueError(run["run_id"])
        tag = f'{run["run_id"]}_a{attempt["attempt"]}'
        archive = archives / (tag + ".tar.gz")
        checksum = digest(archive)
        if checksum != hashes[archive.name]["sha256"]:
            raise ValueError("Source archive mismatch: " + tag)
        with tarfile.open(archive) as src:
            prefix = f"runs/{tag}/"
            run_bytes = src.extractfile(prefix + "run.json").read()
            data = src.extractfile(prefix + "calibration_residuals.json").read()
        if hashlib.sha256(run_bytes).hexdigest() != attempt["record_sha256"]:
            raise ValueError("Sealed run record mismatch: " + tag)
        records = json.loads(data)
        if [r["step"] for r in records] != list(range(40)) or not all(r["active"] for r in records):
            raise ValueError("Expected 40 active focal updates")
        values = np.asarray([[r[key] for key in FIELDS] for r in records], dtype=np.float32)
        if values.shape != (40, 4, 1, 5, 2) or not np.isfinite(values).all():
            raise ValueError("Invalid archived focal coordinates")
        traces.append(values[:, :, 0])
        provenance.append(dict(tag=tag, scene=run["scene"], condition=run["condition"],
                               archive_sha256=checksum, log_sha256=hashlib.sha256(data).hexdigest()))
    return np.stack(traces), provenance


def first_difference(predicted, original, alternative):
    first = np.clip(predicted - original, -1, 1)
    second = np.clip(predicted - alternative, -1, 1)
    changed = np.flatnonzero(np.any(first != second, axis=-1))
    return int(changed[0]) if changed.size else None


def counts(rows):
    steps = [r["first_difference"] for r in rows]
    histogram = Counter(s for s in steps if s is not None)
    return dict(candidates=len(rows), never_called=steps.count(None), immediately_required=steps.count(0),
                first_required_later=sum(n for step, n in histogram.items() if step > 0),
                first_required_update={str(step): histogram[step] for step in sorted(histogram)})


def score(traces, sources):
    if traces.shape != (12, 40, 4, 5, 2) or traces.dtype != np.float32 or len(sources) != 12:
        raise ValueError("Unexpected factorial shape or dtype")
    if len({(s["scene"], s["condition"]) for s in sources}) != 12:
        raise ValueError("Duplicate source")
    rows = []
    for trace, source in zip(traces, sources):
        predicted, target, residual, slope = np.moveaxis(trace, 1, 0)
        np.testing.assert_array_equal(predicted - target, residual)
        np.testing.assert_array_equal(np.clip(residual, -1, 1), slope)
        np.testing.assert_array_equal(target, np.broadcast_to(target[0], target.shape))
        if (target <= 0).any() or (predicted <= 0).any():
            raise ValueError("Invalid focal length")
        for view in range(5):
            focal = target[:, view]
            for scale in SCALES:
                for direction in (-1, 1):
                    candidate = (focal.astype(np.float64) * scale ** direction).astype(np.float32)
                    step = first_difference(predicted[:, view], focal, candidate)
                    rows.append(dict(tag=source["tag"], scene=source["scene"], condition=source["condition"],
                                     view=view, scale=scale, direction=direction, first_difference=step))
    summary = counts(rows)
    breakdowns = {}
    for key in ("scene", "condition", "scale", "direction"):
        breakdowns[key] = {str(v): counts([r for r in rows if r[key] == v])
                           for v in sorted({r[key] for r in rows})}
    return dict(primary=summary, groups=breakdowns, supports_timing_as_standalone=
                summary["never_called"] / summary["candidates"] >= .30, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract", "verify"))
    parser.add_argument("--archives", type=Path)
    args = parser.parse_args()
    trace_file = OUT / "focal_traces.zip"
    result_file = OUT / "summary.json"
    if args.action == "extract":
        if not args.archives or trace_file.exists() or result_file.exists():
            parser.error("Supply raw archives and an empty output location")
        traces, sources = extract(args.archives)
        result = score(traces, sources)
        OUT.mkdir(parents=True, exist_ok=True)
        with trace_file.open("xb") as stream:
            np.savez_compressed(stream, focal=traces)
        save(result_file, dict(protocol_sha256=digest(PROTOCOL), trace_sha256=digest(trace_file),
                               sources=sources, results=result, additional_cloud_usd=0))
    record = read(result_file)
    if record["trace_sha256"] != digest(trace_file) or record["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Public data or registered question changed")
    archived = read(PARENT / "session.json")["raw_archives"]
    for source in record["sources"]:
        if source["archive_sha256"] != archived[source["tag"] + ".tar.gz"]["sha256"]:
            raise ValueError("Source archive identity mismatch")
    with np.load(trace_file, allow_pickle=False) as data:
        result = score(data["focal"], record["sources"])
    if result != record["results"]:
        raise ValueError("Public focal replay differs")
    print(json.dumps(result["primary"]))


if __name__ == "__main__":
    main()
