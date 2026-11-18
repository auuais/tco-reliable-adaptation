"""Seal all 36 profiled-LoRA predictions without decoding any study arrays.

Every discovered attempt is audited. UTC intervals establish chronological
selection; filesystem modification times are never used. Partial or unreported
attempts, mixed registries, and modified artifacts prevent a global seal.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from profiled_artifacts import logical_inventory, resolve_logical


ROOT = Path(__file__).resolve().parents[1]
SCENES = ("heads", "chess", "fire")
SIDES = ("earlier", "later")
ARMS = tuple(f"P{i}" for i in range(6))
RULES = "manifests/lora_profiled_evaluation_v2.json"
FAILURE_POLICY = {
    "automatic_numerical_error_types": ["FloatingPointError"],
    "unclassified_errors": "block all study-score release pending engineering review",
    "infrastructure_errors": "no automatic negative scientific verdict; any disposition must be frozen before scores",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(root, name):
    return resolve_logical(root, name)


def dump_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def timestamp(value):
    require(isinstance(value, str), "Attempt lacks an explicit UTC timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset().total_seconds() == 0,
            "Attempt timestamps must use UTC")
    return parsed.astimezone(timezone.utc)


def inventory(directory, exclude=()):
    return {name: sha(path) for name, path in logical_inventory(directory).items() if name not in exclude}


def check_map(root, mapping, text=False):
    require(isinstance(mapping, dict), "Expected a path/hash mapping")
    for name, expected in mapping.items():
        require(sha(relative(root, name), text=text) == expected, f"Changed referenced artifact: {name}")


def expected_jobs(registry):
    jobs = registry.get("jobs", [])
    expected = {f"{scene}_{side}_{arm}" for scene in SCENES for side in SIDES for arm in ARMS}
    require(len(jobs) == 36 and {job.get("run_id") for job in jobs} == expected,
            "Expected exactly 36 distinct registered run IDs")
    for job in jobs:
        require(job["run_id"] == f'{job["scene"]}_{job["adaptation_source"]}_{job["arm"]}',
                "Registered run ID and factors differ")
        require(job["adapt_lora"] == (job["arm"] != "P0") and
                job["optimizer_steps"] == (0 if job["arm"] == "P0" else 8),
                "Registered arm/step contract differs")
    return jobs


def check_identity(root, record, job, registry, manifest_name, manifest_hash, attempt):
    require(record.get("run_id") == job["run_id"] and record.get("attempt") == attempt and
            record.get("job") == job, "Attempt identity or registered job differs")
    require(record.get("manifest") == manifest_name and record.get("manifest_sha256") == manifest_hash,
            "Mixed scientific registries")
    for key in ("source_sha256_lf", "artifact_sha256", "checkpoint_sha256", "docker_image", "runtime_policy"):
        require(record.get(key) == registry.get(key), f"Attempt {key} differs from the common registry")
    inputs = record.get("input_sha256", {})
    require(all(inputs.get(name) == value for name, value in registry["input_sha256"].items()),
            "Attempt omits a registered input hash")
    require(all(registry["artifact_sha256"].get(name) == value for name, value in inputs.items()),
            "Attempt includes an input not bound by the registry")
    check_map(root, inputs)
    require(record.get("accuracy_scoring_performed") is False and
            record.get("held_source_development_arrays_loaded") is False and
            record.get("target_depth_arrays_loaded") is False, "Attempt violated prediction-before-scoring scope")


def audit_attempt(root, job, registry, manifest_name, manifest_hash, attempt):
    public = relative(root, job["public_attempt_parent"]) / attempt
    raw = relative(root, job["raw_attempt_parent"]) / attempt
    require(public.is_dir() and raw.is_dir(), "Partial attempt directory pair")
    complete, failure = public / "complete.json", public / "failure.json"
    require(complete.is_file() != failure.is_file(), "Partial attempt or contradictory completion/failure")
    path = complete if complete.is_file() else failure
    record = read(path)
    check_identity(root, record, job, registry, manifest_name, manifest_hash, attempt)
    require(record.get("public_attempt") == public.relative_to(root).as_posix() and
            record.get("raw_attempt") == raw.relative_to(root).as_posix(), "Attempt output roots differ")
    started = read(public / "started.json")
    check_identity(root, started, job, registry, manifest_name, manifest_hash, attempt)
    require(record.get("started_at_utc") == started.get("started_at_utc"), "Attempt start records differ")
    start = timestamp(record.get("started_at_utc"))
    finish_key = "completed_at_utc" if complete.is_file() else "failed_at_utc"
    finish = timestamp(record.get(finish_key))
    require(start <= finish, "Attempt finishes before it starts")
    result = dict(attempt=attempt, status="complete" if complete.is_file() else "failed",
        started_at_utc=record["started_at_utc"], finished_at_utc=record[finish_key],
        report=path.relative_to(root).as_posix(), report_sha256=sha(path),
        public_files=inventory(public), raw_files=inventory(raw), input_sha256=record["input_sha256"])
    if failure.is_file():
        require(record.get("pass_all") is False and record.get("error_type") and record.get("message"),
                "Unreported or malformed attempt failure")
        result.update(error_type=record["error_type"], message=record["message"],
                      failure_classification="numerical" if record["error_type"] in
                      FAILURE_POLICY["automatic_numerical_error_types"] else "unclassified")
        return result
    require(record.get("pass_all") is True and record.get("original_state_unchanged") is True and
            record.get("status") == "predictions_sealed_without_accuracy_scoring", "Attempt is not a valid completion")
    require(record.get("original_state_sha256") and
            record["original_state_sha256"] == record.get("pre_endpoint_original_state_sha256") ==
            record.get("final_original_state_sha256"), "Original model state identity differs")
    steps = record.get("steps", [])
    require(record.get("optimizer_updates") == job["optimizer_steps"] and len(steps) == job["optimizer_steps"],
            "Unexpected optimizer or trace count")
    require(all(row["step"] == index and row["optimizer_steps"] == 1 and
                row["optimizer_updates_cumulative"] == index+1 and
                row["graph_free_decoder_calls"] == row["differentiable_decoder_calls"] == 20
                for index, row in enumerate(steps)), "Incomplete registered macro-step trace")
    require(all(read(public / f"step_{index:02d}.json") == row for index, row in enumerate(steps)),
            "Per-step trace files differ from completion record")
    require(record.get("raw_files") == result["raw_files"] and
            record.get("public_files") == inventory(public, exclude=("complete.json",)),
            "Changed bytes or file inventory after completion")
    endpoint = record["endpoint"]
    for kind in ("native", "final"):
        expected = (raw / "endpoint" / f"{kind}.npz").relative_to(root).as_posix()
        require(endpoint.get(kind+"_file") == expected and
                sha(relative(root, expected)) == endpoint.get(kind+"_sha256"), "Endpoint identity or bytes differ")
    capture = endpoint["capture"]
    require(capture["prior_cache_files"] == 0 and capture["pairs"] == 20 and
            capture["coarse_iterations"] == capture["fine_iterations"] == 300 and
            capture["native_size"] == [384, 512], "Fresh native endpoint contract differs")
    duration = record.get("seconds")
    require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0,
            "Missing finite measured runtime")
    if job["adapt_lora"]:
        require("adapters.npz" in record["raw_files"] and record.get("adapter_final_sha256") and
                record.get("adapter_module_count") == 144 and record.get("trainable_parameters") == 1032192,
                "Adapter artifact or model identity is missing")
        adapter = dict(file=(raw / "adapters.npz").relative_to(root).as_posix(),
                       sha256=record["raw_files"]["adapters.npz"], state_sha256=record["adapter_final_sha256"])
    else:
        require("adapters.npz" not in record["raw_files"] and record.get("adapter_final_sha256") is None and
                record.get("adapter_module_count") == record.get("trainable_parameters") == 0,
                "Frozen P0 contains an adapter")
        adapter = dict(frozen_P0=True, original_state_sha256=record["original_state_sha256"])
    result.update(native_file=endpoint["native_file"], native_sha256=endpoint["native_sha256"],
                  final_file=endpoint["final_file"], final_sha256=endpoint["final_sha256"],
                  adapter=adapter, seconds=float(duration), original_state_sha256=record["original_state_sha256"])
    return result


def build_seal(root, manifest_name, evaluation_identity, rules_name=RULES):
    """Pure file/provenance audit. Does not import NumPy or deserialize predictions."""
    root = Path(root).resolve()
    manifest = relative(root, manifest_name)
    registry, manifest_hash = read(manifest), sha(manifest)
    require(registry.get("phase") == "scientific" and registry.get("scientific_ready_at_registration") is True,
            "A ready scientific registry is required")
    jobs = expected_jobs(registry)
    check_map(root, registry["source_sha256_lf"], text=True)
    check_map(root, registry["artifact_sha256"])
    require(registry["artifact_sha256"].get(rules_name) == sha(relative(root, rules_name)),
            "Scientific registry does not bind these evaluation rules")
    require(sha(relative(root, evaluation_identity["file"])) == evaluation_identity["sha256"],
            "Evaluation execution registration changed")
    rows = []
    for job in jobs:
        public_parent = relative(root, job["public_attempt_parent"])
        raw_parent = relative(root, job["raw_attempt_parent"])
        public_ids = {p.name for p in public_parent.iterdir() if p.is_dir()} if public_parent.is_dir() else set()
        raw_ids = {p.name for p in raw_parent.iterdir() if p.is_dir()} if raw_parent.is_dir() else set()
        require(public_ids and public_ids == raw_ids, f"Missing or partial attempts for {job['run_id']}")
        attempts = [audit_attempt(root, job, registry, manifest_name, manifest_hash, name) for name in sorted(public_ids)]
        by_start = sorted(attempts, key=lambda item: (timestamp(item["started_at_utc"]), item["attempt"]))
        require(all(timestamp(first["finished_at_utc"]) <= timestamp(second["started_at_utc"])
                    for first, second in zip(by_start, by_start[1:])), "Overlapping attempt intervals in one run ID")
        valid = [item for item in attempts if item["status"] == "complete"]
        require(valid, f"No valid completed attempt for {job['run_id']}")
        selected = min(valid, key=lambda item: (timestamp(item["finished_at_utc"]), item["attempt"]))
        rows.append(dict(run_id=job["run_id"], scene=job["scene"], adaptation_source=job["adaptation_source"],
                         arm=job["arm"], selected_attempt=selected["attempt"], selected=selected,
                         failed_attempt_count=sum(item["status"] == "failed" for item in attempts),
                         numerical_failure_count=sum(item.get("failure_classification") == "numerical" for item in attempts),
                         unclassified_failure_count=sum(item.get("failure_classification") == "unclassified" for item in attempts),
                         attempts=attempts))
    require(len({row["selected"]["original_state_sha256"] for row in rows}) == 1,
            "Completed jobs do not share one original model state")
    require(len({json.dumps(attempt["input_sha256"], sort_keys=True)
                 for row in rows for attempt in row["attempts"]}) == 1,
            "Attempts do not share one permitted input registration")
    return dict(schema="profiled_global_prediction_seal_v1", common_scientific_registry=manifest_name,
                common_scientific_registry_sha256=manifest_hash, evaluation_execution=evaluation_identity,
                evaluation_rules=rules_name, evaluation_rules_sha256=sha(relative(root, rules_name)),
                cases=rows, required_states=36, completed_states=len(rows),
                failure_policy=FAILURE_POLICY,
                failure_review_required=any(row["unclassified_failure_count"] for row in rows),
                attempt_selection="earliest valid completed_at_utc, then attempt ID; all discovered attempts audited",
                held_source_arrays_loaded=False, target_depth_arrays_loaded=False,
                prediction_arrays_deserialized=False, pass_all=True)


def verify_seal(root, seal_name, evaluation_identity):
    saved = read(relative(root, seal_name))
    require(saved.get("evaluation_execution") == evaluation_identity, "Seal uses another evaluation registration")
    current = build_seal(root, saved["common_scientific_registry"], evaluation_identity, saved["evaluation_rules"])
    require(current == saved, "Global prediction seal differs from current attempts or artifacts")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--manifest", default="manifests/lora_profiled_execution_v2.json")
    parser.add_argument("--evaluation", default="manifests/lora_profiled_evaluation_execution_v2.json")
    parser.add_argument("--seal", default="results/lora_profiled_v1/prediction_seal.json")
    args = parser.parse_args()
    from register_profiled_execution import verify_registry
    from score_profiled_lora import verify_evaluation_registration
    verify_registry(relative(ROOT, args.manifest), "scientific")
    evaluation = verify_evaluation_registration(ROOT, args.evaluation)
    if args.action == "verify":
        report = verify_seal(ROOT, args.seal, evaluation)
        require(report["common_scientific_registry"] == args.manifest, "Requested registry differs from sealed registry")
    else:
        report = build_seal(ROOT, args.manifest, evaluation)
        dump_new(relative(ROOT, args.seal), report)
    print(json.dumps(dict(pass_all=True, states=report["completed_states"],
                         failure_review_required=report["failure_review_required"], prediction_arrays_deserialized=False)))


if __name__ == "__main__":
    main()
