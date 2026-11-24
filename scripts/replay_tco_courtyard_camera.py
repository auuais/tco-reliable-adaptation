"""Verify public receipts and reproduce the Courtyard and camera study decisions."""

import argparse
import hashlib
import json

from replay_tco_full40_recompute import (
    analyze_five_view,
    analyze_ten_view,
    check_attempts,
    check_result,
    reproducible_oom,
    require,
    valid_sha,
)
from run_profiled_d0 import read, sha, write_new
from tco_courtyard_camera_protocol import (
    CAMERA,
    CONFIRM_SEEDS,
    DEPTH_ONLY,
    INITIAL,
    LOCKED,
    MARGIN,
    PUBLIC,
    REFERENCE,
    ROOT,
    STUDY,
    TRACE_MILESTONES,
    initial_jobs,
    locked_jobs,
    runtime_pins,
    window_key,
)

ENDPOINTS = ("accuracy_mean_m", "completeness_mean_m")
ALIGNMENTS = ("similarity_only", "similarity_and_icp")


def payload_digest(payload):
    return hashlib.sha256((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def verify_roster(plan, expected):
    require(plan["study"] == STUDY and plan["jobs"] == expected, "Registration roster differs")
    require(all(plan[key] == value for key, value in runtime_pins().items()), "Wrong runtime pins")
    require(plan["margin"] == MARGIN and plan["bootstrap_seed"] == 20261125
            and plan["bootstrap_samples"] == 10_000, "Wrong statistical protocol")
    for name, digest in plan["source_sha256_lf"].items():
        path = ROOT / name
        # CI does not check out the upstream clone. Its byte pins remain in the manifests.
        if name.startswith("third_party/") and not path.is_file():
            continue
        require(sha(path, text=True) == digest, f"Registered source changed: {name}")
    require(all(valid_sha(digest) for digest in plan["input_source_sha256"].values()), "Invalid public raw input digest")


def verify_cases(plan, evidence, authority_sha):
    require(evidence["study"] == STUDY and evidence["registration_sha256"] == authority_sha
            and evidence["rental_compute_usd"] == 0, "Detached or invalid evidence")
    jobs = {job["id"]: job for job in plan["jobs"]}
    records = {run["id"]: run for run in evidence["runs"]}
    require(records.keys() == jobs.keys() and len(evidence["runs"]) == len(records), "Evidence roster differs")
    expected_signatures = plan.get("expected_confirmation_signatures", [])
    signatures = {window_key(item): item for item in expected_signatures}
    scored_windows = {window_key(job) for job in jobs.values() if job["cohort"] in ("confirmation", "revisit")}
    require(len(signatures) == len(expected_signatures) and signatures.keys() == scored_windows,
            "Missing or duplicate decoded window signatures")
    seen = {}
    results = {}
    for identifier, job in jobs.items():
        row = records[identifier]
        check_attempts(row, job)
        require(len(row["attempts"]) <= plan["max_attempts"], f"Unexpected retries for {identifier}")
        if row["verdict"] != "completed":
            require(row["result"] is None, f"Unsuccessful case has results: {identifier}")
            continue
        result = row["result"]
        require(payload_digest(result) == row["attempts"][-1]["complete_sha256"]
                and all(result.get(key) == value for key, value in job.items())
                and result["rental_compute_usd"] == 0, f"Completion receipt differs: {identifier}")
        results[identifier] = result
        profile = job["profile"]
        if profile in ("fixture", "archived", "diagnosis"):
            key = "fixture" if profile == "fixture" else profile
            require(bool(result[key]), f"Empty CPU result: {identifier}")
        elif profile == "readout":
            require(result["updates"] == 0 and valid_sha(result["readout_sha256"]),
                    f"Readout executed adaptation or lacks provenance: {identifier}")
        else:
            check_result(result, job, scored=job["score"])
            runtime = result["runtime"]
            require(all(runtime[field] == plan[field] for field in ("gpu", "gpu_memory_fraction", "allocator"))
                    and runtime["torch"] == "2.7.1+cu128" and runtime["cuda"] == "12.8"
                    and result["peak_allocated"] <= result["cuda_device_total"] * plan["gpu_memory_fraction"],
                    f"Runtime does not match the registered local resources: {identifier}")
            views = len(job["frames"])
            groups = [1] * views if job["policy"] in ("c1_on", "c1_off") else ([8, 2] if views == 10 else [5])
            require(result["native_depth_batches"] == [groups] and result["camera_checkpoint"] == (job["policy"] == CAMERA),
                    f"Depth batches or checkpoint assignment changed: {identifier}")
            window = window_key(job)
            decoded = (result["input_sha256"], result["reference_sha256"])
            if job.get("probe") != "leak":
                require(seen.setdefault(window, decoded) == decoded, f"Mismatched input pair: {identifier}")
            if window in signatures:
                signature = signatures[window]
                require(decoded == (signature["input_sha256"], signature["reference_sha256"])
                        and result["raster_shape"] == signature["raster_shape"]
                        and result["metrics"]["valid_points"] == signature["valid_points"],
                        f"Decoded inputs, raster or mask count differ: {identifier}")

    for job in jobs.values():
        if job["profile"] == "readout" and job["id"] in results:
            require(results[job["id"]]["source_trace_sha256"] == results[job["trace_job"]]["trace_sha256"],
                    "Common readout used a different trace")
        if job["cohort"] == "ten" and plan.get("five_only"):
            require(records[job["id"]]["verdict"] == "unattempted", "Attempted a closed ten-view branch")
        if job.get("first_gate") and records[job["id"]]["verdict"] != "unattempted":
            require(records[job["first_gate"]]["verdict"] == "completed", "Native seed gate was bypassed")
    return records, results


def screen_verdict(cases):
    clean, pilot = cases["gate_camera"], cases["promote_camera"]
    if clean["verdict"] == pilot["verdict"] == "completed":
        return "promote"
    for phase, row in (("two_update", clean), ("full_length", pilot)):
        if row["verdict"] == "oom" and sum(item["verdict"] == "oom" for item in row["attempts"]) >= 2:
            return f"repeatable_{phase}_oom"
    return "ten_view_fit_unverified"


def verify_diagnosis(diagnosis, results):
    archive, traces = diagnosis["archived"], diagnosis["traces"]
    require(archive["passed"] and archive["atol_m"] == 1e-5 and len(archive["rows"]) == 6,
            "Historical rescoring failed or lacked coverage")
    previous = read(ROOT / "results/tco_full40_recompute_v1/evidence.json")
    old_rows = {row["id"]: row for row in previous["runs"]}
    historical = {f"conf_eth3d_courtyard_{policy}_{seed}"
                  for seed in CONFIRM_SEEDS for policy in (REFERENCE, "c1_on")}
    require({row["id"] for row in archive["rows"]} == historical, "Historical Courtyard roster differs")
    for row in archive["rows"]:
        old = old_rows[row["id"]]
        original = old["result"]["metrics"]
        require(row["parity"] and row["receipt_sha256"] == old["attempts"][-1]["complete_sha256"]
                and row["output_sha256"] == old["result"]["output_sha256"]
                and all(row["original"][key] == original[key] for key in (*ENDPOINTS, "valid_points"))
                and all(abs(delta) <= 1e-5 for fields in row["deltas"].values() for delta in fields.values()),
                f"Invalid archived parity: {row['id']}")

    require(traces["matched_render_rng"] and set(traces["seeds"]) == set(map(str, CONFIRM_SEEDS)),
            "Trace coverage or matched RNG checks failed")
    stages = (REFERENCE, *DEPTH_ONLY, "native_repeat")
    for seed, value in traces["seeds"].items():
        require(set(value["milestones"]) == set(map(str, TRACE_MILESTONES)), "Missing trajectory stages")
        require(set(value["source_hashes"]) == set(stages), "Missing trace control arm")
        draws = []
        for policy in stages:
            trace, readout = results[f"trace_{policy}_{seed}"], results[f"readout_{policy}_{seed}"]
            receipts = value["source_hashes"][policy]
            require(receipts["trace"] == trace["trace_sha256"]
                    and receipts["readout"] == readout["readout_sha256"]
                    and all(row["relative_l2"] <= 1e-5 for row in receipts["step40_parity"].values()),
                    f"Trace artifacts differ for {policy}, seed {seed}")
            draws.append([(row["render_permutation"], row["renderer_rng_before"], row["renderer_rng_after"])
                          for row in value["training"][policy]])
        require(all(len(arm) == 40 and arm == draws[0] for arm in draws)
                and len({row["init_factors"] for row in value["source_hashes"].values()}) == 1,
                f"Trace optimization inputs differ at seed {seed}")
        for label in ("chunk_shape", "same_chunk_checkpoint"):
            for stage in ALIGNMENTS:
                for metric in ENDPOINTS:
                    marked = []
                    for step in TRACE_MILESTONES:
                        row = value["comparisons"][str(step)]
                        if abs(row[label]["score_delta_m"][stage][metric]) > abs(
                                row["native_repeat"]["score_delta_m"][stage][metric]):
                            marked.append(step)
                    first = marked[0] if marked else None
                    summary = value["observed_onset"][label][stage][metric]
                    require(summary["first_completed_updates"] == first
                            and summary["checkpoints_exceeding_repeat"] == marked
                            and summary["persistent_to_40"] == (first is not None and marked == [
                                step for step in TRACE_MILESTONES if step >= first]),
                            f"Courtyard divergence summary differs at seed {seed}")


def verify_authorities(plan, initial_records, initial_results, initial_sha):
    for name, digest in plan["authority_sha256_lf"].items():
        require(sha(PUBLIC / name, text=True) == digest, f"Stage evidence changed: {name}")
    diagnosis = read(PUBLIC / "diagnosis.json")
    require(diagnosis["initial_sha256"] == initial_sha and diagnosis["rental_compute_usd"] == 0
            and diagnosis["archived"] == initial_results["archived_rescore"]["archived"]
            and diagnosis["traces"] == initial_results["diagnose_courtyard"]["diagnosis"],
            "Diagnosis differs from phase-A receipts")
    for name in ("archived_rescore", "diagnose_courtyard"):
        require(diagnosis["receipt_sha256"][name] == initial_records[name]["attempts"][-1]["complete_sha256"],
                f"Wrong phase-A receipt hash: {name}")
    verify_diagnosis(diagnosis, initial_results)

    qualification = read(PUBLIC / "qualification.json")
    qualification_jobs = dict(fixture="camera_fixture", native_identity="identity_c8_off",
                              camera_identity="identity_camera", native_fixed="fixed_c8_off",
                              camera_fixed="fixed_camera", leak="leak_camera")
    require(qualification["qualified"] and qualification["authority_sha256"] == initial_sha
            and qualification["rental_compute_usd"] == 0
            and qualification["native_depth_batches"] == [[5]] and qualification["forward_hashes_identical"]
            and qualification["native_fixed_factors"]["tensors"] == 1536
            and qualification["native_fixed_factors"]["max_relative_l2"] <= 1e-4
            and qualification["native_fixed_predictions"]["max_relative_l2"] <= 1e-5
            and qualification["native_release_predictions"]["max_relative_l2"] <= 1e-5
            and qualification["leak_predictions"]["max_relative_l2"] == 0,
            "Implementation qualification failed")
    for field, name in qualification_jobs.items():
        require(name in initial_results and qualification["receipts"][field]
                == initial_records[name]["attempts"][-1]["complete_sha256"], f"Unbound qualification case: {name}")
    native, camera, leak = (initial_results[qualification_jobs[key]]
                            for key in ("native_fixed", "camera_fixed", "leak"))
    for key in ("taps", "render_orders", "pose_encoding_hashes", "adapter_initial", "update_predictions"):
        require(native["audit"][key] == camera["audit"][key] == leak["audit"][key],
                f"Paired qualification mismatch in {key}")
    require(camera["input_sha256"] == leak["input_sha256"] and camera["reference_sha256"] != leak["reference_sha256"]
            and qualification["leak_output_sha256"] == camera["output_sha256"] == leak["output_sha256"],
            "Evaluation reference altered the candidate output")

    screen = read(PUBLIC / "screen.json")
    require(screen["initial_sha256"] == initial_sha and screen["rental_compute_usd"] == 0, "Unbound capacity screen")
    for name, case in screen["cases"].items():
        evidence = initial_records[name]
        keys = ("attempt", "verdict", "log_sha256", "complete_sha256")
        require(case["verdict"] == evidence["verdict"] and len(case["attempts"]) == len(evidence["attempts"])
                and all(all(a[key] == b[key] for key in keys) for a, b in zip(case["attempts"], evidence["attempts"])),
                f"Unbound capacity case: {name}")
    verdict = screen_verdict(initial_records)
    require(screen["gate"] == verdict and screen["ten_view_promoted"] == (verdict == "promote")
            and plan["five_only"] == (verdict != "promote"), "Ten-view screening and lock disagree")
    return screen, diagnosis


def analyze_locked(plan, records, results, screen):
    seed, samples = plan["bootstrap_seed"], plan["bootstrap_samples"]
    primary = {}
    for dataset in ("7scenes", "eth3d"):
        jobs = [job for job in plan["jobs"] if job["cohort"] == "confirmation" and job["dataset"] == dataset]
        require(len(jobs) == 42 and all(job["scene"] != "courtyard" for job in jobs), "Wrong primary cohort")
        primary[dataset] = analyze_five_view(jobs, results, CAMERA, REFERENCE, seed, samples, inference=True)
    revisit = [job for job in plan["jobs"] if job["cohort"] == "revisit"]
    ten_jobs = [job for job in plan["jobs"] if job["cohort"] == "ten"]
    require(len(revisit) == 6 and all(job["scene"] == "courtyard" for job in revisit)
            and len(ten_jobs) == 18, "Incorrect diagnostic or ten-view coverage")
    ten = analyze_ten_view(ten_jobs, records, results, CAMERA, REFERENCE, (REFERENCE,))
    ten.update(screen=screen["gate"], locked_branch_ran=not plan["five_only"])
    return dict(primary=primary, courtyard=analyze_five_view(revisit, results, CAMERA, REFERENCE,
                                                             seed, samples, inference=False),
                ten_view=ten,
                five_view_supported=all(value["verdict"] == "within_margin" for value in primary.values()),
                completed=len(results), registered=len(plan["jobs"]), rental_compute_usd=0)


def replay(publish=False):
    initial, plan = read(INITIAL), read(LOCKED)
    initial_sha, lock_sha = sha(INITIAL, text=True), sha(LOCKED, text=True)
    require(plan["initial_sha256"] == initial_sha, "Detached initial registration")
    verify_roster(initial, initial_jobs())
    verify_roster(plan, locked_jobs(initial["planned_windows"]))
    initial_records, initial_results = verify_cases(initial, read(PUBLIC / "initial_evidence.json"), initial_sha)
    records, results = verify_cases(plan, read(PUBLIC / "evidence.json"), lock_sha)
    screen, diagnosis = verify_authorities(plan, initial_records, initial_results, initial_sha)
    decision = dict(study=STUDY, lock_sha256=lock_sha, diagnosis_sha256=payload_digest(diagnosis),
                    **analyze_locked(plan, records, results, screen))
    if publish:
        write_new(PUBLIC / "decision.json", decision)
    else:
        require(decision == read(PUBLIC / "decision.json"), "Independently replayed decision differs")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = replay(args.publish)
    print(json.dumps(dict(completed=result["completed"], registered=result["registered"],
                          ten_view=result["ten_view"]["screen"], five_view_supported=result["five_view_supported"])))


if __name__ == "__main__":
    main()
