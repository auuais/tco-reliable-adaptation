"""Rescore completed native development predictions in a separate CPU process."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import run_tco_render_lifetime as original
from run_profiled_d0 import read, sha, write_new
from run_tco_five_view_replay_cache import source_directory
from run_tco_one_percent import DATA_ROOT, load_case
from tco_five_view_replay_cache_protocol import PRIVATE, PUBLIC, REGISTRATIONS, STUDY
from tco_one_percent_protocol import check_case_inputs


def main():
    if torch.cuda.is_available():
        raise ValueError("Use the pinned environment with no GPU devices")
    evidence = PUBLIC / "development_evidence.json"
    manifest = REGISTRATIONS["development"]
    plan = read(manifest)
    jobs = {row["id"]: row for row in plan["jobs"]}
    ledger = read(evidence)
    if ledger["registration_sha256"] != sha(manifest, text=True):
        raise ValueError("Development evidence does not match the registration")
    results = {}
    for row in ledger["runs"]:
        identifier = row["id"]
        job = jobs[identifier]
        if job["policy"] != "native" or not job["score"] or not row["result"]:
            continue
        source = source_directory(identifier)
        check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
        with TemporaryDirectory(prefix="tco-five-view-rescore-") as folder:
            out = Path(folder)
            _, reference = load_case(job, out, job["width"])
            with patch.object(original, "PRIVATE", PRIVATE):
                original.rescore(dict(job, source_job=identifier), out, reference)
            rescore = read(out / "complete.json")
        if rescore["original_metrics"] != row["result"]["metrics"]:
            raise ValueError(f"Rescore loaded different native metrics for {identifier}")
        deltas = {key: 1000 * (rescore["metrics"][key] - rescore["original_metrics"][key])
                  for key in ("accuracy_mean_m", "completeness_mean_m")}
        within = max(map(abs, deltas.values())) <= 0.01
        results[identifier] = dict(deltas_mm=deltas, reproducible=within,
                                   source_completion_sha256=sha(source / "complete.json"),
                                   source_prediction_sha256=rescore["source_prediction_sha256"],
                                   reference_sha256=rescore["reference_sha256"])
        print(f"{identifier}: maximum score change {max(map(abs, deltas.values())):.3g} mm", flush=True)
    if len(results) == 0:
        raise ValueError("No saved native predictions eligible for independent scoring")
    path = PUBLIC / "native_cpu_rescore.json"
    write_new(path, dict(study=STUDY, input_evidence_sha256_lf=sha(evidence, text=True),
                         scored=len(results), tolerance_mm=0.01, pinned_torch=torch.__version__,
                         results=results, pass_all=all(item["reproducible"] for item in results.values()),
                         rental_compute_usd=0))
    if not all(item["reproducible"] for item in results.values()):
        raise ValueError("Native score replay exceeded the registered tolerance")


if __name__ == "__main__":
    main()
