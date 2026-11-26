"""Descriptive local or released 40-step resource measurements, including partial coverage."""

import argparse
from collections import Counter, defaultdict
import json
from statistics import median

from replay_tco_five_view_replay_cache import (
    PUBLIC, RESOURCES, ROOT, STUDY, authority, finite_positive, percentile, require, verified, window,
)
from run_profiled_d0 import read, sha, write_new


RAW = ROOT / "reports" / STUDY
OUTCOMES = ("completed", "timeout", "oom", "error", "nonfinite", "interrupted")
CONTRASTS = (
    ("camera", "native"), ("last", "native"), ("last", "camera"), ("retention", "native"),
    ("combined", "native"), ("combined", "last"), ("combined", "retention"), ("combined", "uniform"),
    ("uniform", "last"), ("camera_retention", "camera"), ("camera_retention", "uniform"),
    ("native_repeat", "native"),
)


def private_row(job):
    identifier = job["id"]
    attempts, result, waiting = [], None, None
    for folder in sorted((RAW / "launches" / identifier).glob("attempt*")):
        terminal = folder / "terminal.json"
        if not terminal.exists():
            require(waiting is None and result is None, f"Unexpected open attempt for {identifier}")
            waiting = folder.name
            continue
        attempt = read(terminal)
        require(waiting is None and result is None and attempt["job"] == identifier
                and attempt["attempt"] == folder.name and attempt["verdict"] in OUTCOMES
                and attempt["rental_compute_usd"] == 0, f"Invalid attempt history for {identifier}")
        attempts.append(attempt)
        if attempt["verdict"] == "completed":
            path = RAW / "cases" / identifier / folder.name / "case/complete.json"
            require(attempt["stopped"] and attempt["exit_code"] == 0 and sha(path) == attempt["complete_sha256"],
                    f"Unverified completed attempt {identifier}/{folder.name}")
            result = read(path)
    if waiting:
        state = "awaiting_terminal"
    else:
        state = attempts[-1]["verdict"] if attempts else "unattempted"
    return dict(id=identifier, attempts=attempts, verdict=state, result=result)


def counts(counter, states):
    return {key: counter.get(key, 0) for key in states}


def resources_for(cases):
    completed = [c for c in cases if c["result"] is not None]
    final_states = Counter(c["verdict"] for c in cases)
    attempts = Counter(a["verdict"] for c in cases for a in c["attempts"])
    group = dict(planned=len(cases), case_counts=counts(final_states, (*OUTCOMES, "unattempted", "awaiting_terminal")),
                 terminal_attempt_counts=counts(attempts, OUTCOMES),
                 completed_seeds=[c["seed"] for c in completed])
    if completed:
        times = [c["result"]["scene_seconds"] for c in completed]
        group["scene_seconds"] = dict(median=median(times), p90=percentile(times, 0.9), max=max(times))
        for name in ("peak_allocated", "peak_reserved"):
            values = [c["result"][name] / 2**30 for c in completed]
            group[name + "_gib"] = dict(median=median(values), max=max(values))
    return group


def contrast(candidate, baseline):
    pairs = []
    seeds = sorted(set(candidate) & set(baseline))
    for seed in seeds:
        lhs, rhs = candidate[seed]["result"], baseline[seed]["result"]
        if lhs is None or rhs is None:
            continue
        require(lhs["input_sha256"] == rhs["input_sha256"]
                and lhs["raster_shape"] == rhs["raster_shape"], "Matched seeds carry different inputs")
        pairs.append((seed, lhs, rhs))
    record = dict(planned_seeds=seeds, paired_seeds=[s for s, _, _ in pairs],
                  missing_seeds=[s for s in seeds if candidate[s]["result"] is None or baseline[s]["result"] is None])
    if pairs:
        for name in RESOURCES:
            unit = 1 if name == "scene_seconds" else 2**30
            key = name if unit == 1 else name + "_gib"
            lhs = [left[name] / unit for _, left, _ in pairs]
            rhs = [right[name] / unit for _, _, right in pairs]
            record[key] = dict(candidate_median=median(lhs), baseline_median=median(rhs),
                               ratio_of_matched_medians=median(lhs) / median(rhs),
                               median_paired_difference=median(a - b for a, b in zip(lhs, rhs)),
                               per_seed_ratio={str(s): a / b for (s, _, _), a, b in zip(pairs, lhs, rhs)})
    return record


def summarize(jobs, rows):
    windows = defaultdict(lambda: defaultdict(dict))
    require(len(jobs) == len(rows), "Result count differs from the development roster")
    for job, row in zip(jobs, rows):
        require(row["id"] == job["id"], "Unaligned development case")
        if job["cohort"] not in ("development", "repeatability"):
            continue
        require(job["steps"] == 40 and len(job["frames"]) == 5, "Unexpected full-length five-view workload")
        policy = "native_repeat" if job["cohort"] == "repeatability" else job["policy"]
        seed = job["seed"]
        cases = windows[window(job)][policy]
        require(seed not in cases, f"Duplicate case for {job['id']}")
        result = row["result"]
        if result is not None:
            require(all(result.get(key) == value for key, value in job.items()) and result["updates"] == 40
                    and result["lora_tensors"] == 384 and result["native_depth_batches"] == [[5]]
                    and result["runtime"]["gpu"] == "NVIDIA GeForce RTX 3080" and result["rental_compute_usd"] == 0
                    and all(finite_positive(result.get(k)) for k in RESOURCES), f"Invalid receipt for {job['id']}")
        cases[seed] = dict(row, seed=seed)
    output = []
    for (dataset, scene, frames, width), methods in sorted(windows.items()):
        groups = {name: resources_for([by_seed[s] for s in sorted(by_seed)])
                  for name, by_seed in sorted(methods.items())}
        pairs = {f"{a}_vs_{b}": contrast(methods[a], methods[b])
                 for a, b in CONTRASTS if a in methods and b in methods}
        output.append(dict(dataset=dataset, scene=scene, frames=list(frames), width=width,
                           policies=groups, contrasts=pairs))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("receipts", "evidence"), default="evidence",
                        help="Use receipts for live partial coverage or evidence for public CPU replay.")
    parser.add_argument("--output", help="Write a new JSON file. The default is stdout.")
    args = parser.parse_args()
    path = authority("development")
    if args.source == "evidence":
        plan, rows = verified("development")
        source = dict(development_evidence_sha256_lf=sha(PUBLIC / "development_evidence.json", text=True))
    else:
        require((RAW / "launches").is_dir(), "Cannot find private attempt receipts")
        plan = read(path)
        rows = [private_row(job) if job["cohort"] in ("development", "repeatability")
                else dict(id=job["id"], result=None, attempts=[], verdict="unattempted") for job in plan["jobs"]]
        source = {"completed_receipt_sha256": {row["id"]: row["attempts"][-1]["complete_sha256"]
                  for row in rows if row["result"] is not None}}
    windows = summarize(plan["jobs"], rows)
    groups = [group for scene in windows for group in scene["policies"].values()]
    totals = dict(planned_full_runs=sum(group["planned"] for group in groups))
    for key, states in (("case_counts", (*OUTCOMES, "unattempted", "awaiting_terminal")),
                        ("terminal_attempt_counts", OUTCOMES)):
        totals[key] = {state: sum(group[key][state] for group in groups) for state in states}
    payload = dict(study=STUDY, scope="Descriptive full-length development and native repeatability windows",
                   source=args.source, development_registration_sha256_lf=sha(path, text=True), **source,
                   timing="Scene time among completions. Timeouts carry no scene-time or peak estimate.",
                   pairing="Candidate divided by baseline on identical completed window/seed pairs",
                   totals=totals, windows=windows)
    if args.output:
        write_new(args.output, payload)
    else:
        print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
