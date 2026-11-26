"""Construct the five-view designs from preregistered diagnostics and resource rules."""

import argparse
from collections import defaultdict
import random
from statistics import median

from run_profiled_d0 import read, sha, write_new
from replay_tco_five_view_replay_cache import choose, resources_by_policy, verified
from tco_five_view_replay_cache_protocol import (
    DEV_SEEDS, ORDER_SEED, PUBLIC, REGISTRATIONS, STUDY, development_windows, job, window_fields,
)


ANCHORS = (0, 7, 15, 23)
SITES = tuple(f"{kind}:{stage}" for stage in range(24) for kind in ("frame", "global"))
WINDOWS = ("heads", "courtyard")
SET_SIZES = (1, 2, 4)


def stage(name):
    path = PUBLIC / f"{name}_evidence.json"
    plan, rows = verified(name)
    return plan, {row["id"]: row["result"] for row in rows if row["result"] is not None}, sha(path, text=True)


def authority(name, parent, evidence, extra_authorities=(), **payload):
    path = PUBLIC / f"{name}_plan.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    digests = {f"{key}_evidence.json": value for key, value in evidence.items()}
    digests.update({file: sha(PUBLIC / file, text=True) for file in extra_authorities})
    write_new(path, dict(study=STUDY, parent_registration_sha256=sha(REGISTRATIONS[parent], text=True),
                         authority_sha256_lf=digests, **payload))
    print(f"Wrote {path}: {len(payload.get('jobs', []))} jobs")


def dev_site(window):
    return window["id"].removeprefix("window_")


def design_job(identifier, cohort, policy, window, **kwargs):
    return job(identifier, cohort, policy, **window_fields(window), **kwargs)


def camera_arm(rows):
    qualification = read(PUBLIC / "qualification.json")
    if ("last" in qualification["eligible"] and
            all(f"baseline_{window}_last_{replica}" in rows for window in WINDOWS for replica in (1, 2))):
        return "last"
    return "camera"


def report_probe_labels(rows):
    observations = {}
    timing_by_op = defaultdict(float)
    for key in (f"profile_{window}_native" for window in WINDOWS):
        report = rows.get(key, {}).get("policy_execution", {})
        operations = report.get("ops", {})
        observed = set()
        for label, entry in operations.items():
            if "/recompute/" not in label or entry["calls"] <= 0:
                continue
            name = label.rsplit("/", 1)[-1]
            if "scaled_dot_product" in name and "backward" not in name:
                observed.add(name)
                timing_by_op[name] += report.get("cuda_event_ms", {}).get(label, 0)
        observations[key] = sorted(observed)
    shared = set.intersection(*(set(x) for x in observations.values()))
    operation = min(shared, key=lambda name: (-timing_by_op[name], name)) if shared else None
    return operation, observations


def screen():
    _, rows, digest = stage("initial")
    operation, observations = report_probe_labels(rows)
    camera = camera_arm(rows)
    eligible = [window for window in development_windows() if dev_site(window) in WINDOWS]
    jobs = []
    # Screen a fixed stratified sample of frame and global blocks at early, middle and late stages.
    sites = [f"{kind}:{stage}" for stage in ANCHORS for kind in ("frame", "global")]
    modes = ("op", "block") if operation else ("block",)
    for method in modes:
        for window in eligible:
            name = dev_site(window)
            for site in sites:
                label = site.replace(":", "_")
                jobs.append(design_job(f"single_{name}_{method}_{label}", "single_site", camera, window,
                                       steps=2, retention=method, sites=[site], operation=operation or "attention"))
            # Independent instrumented verification of the operation's actual replay/cache state.
            jobs.append(design_job(f"diagnostic_{name}_{method}", "screen_diagnostic", camera, window,
                                   steps=2, retention=method, sites=["frame:7", "global:7"],
                                   operation=operation or "attention", profile="timeline"))
    random.Random(ORDER_SEED).shuffle(jobs)
    authority("screen", "initial", {"initial": digest}, extra_authorities=("qualification.json",),
              jobs=jobs, camera=camera, operation=operation,
              site_sample=sites, observed_recompute_attention=observations)


def peak(rows):
    return median(row["peak_allocated"] for row in rows)


def rank(rows, anchors, settings):
    result = {}
    camera = settings["camera"]
    baselines = {window: [anchors.get(f"baseline_{window}_{camera}_{r}") for r in (1, 2)] for window in WINDOWS}
    if any(None in pair for pair in baselines.values()):
        return result
    for method in ("op", "block"):
        ranking = []
        if method == "op":
            hits = [sum(sum(row.get("hits", {}).values()) for row in
                        rows.get(f"diagnostic_{window}_op", {}).get("policy_execution", {}).get("selective_cache", []))
                    for window in WINDOWS]
            if min(hits) <= 0:
                continue
        for site in settings["site_sample"]:
            label = site.replace(":", "_")
            samples = {name: rows.get(f"single_{name}_{method}_{label}") for name in WINDOWS}
            if None in samples.values():
                continue
            details = []
            for name, row in samples.items():
                reference = baselines[name]
                delta = (row["peak_allocated"] - peak(reference)) / 2**20
                baseline_ms = median(item["mean_step_seconds"] for item in reference) * 1000
                saved_ms = baseline_ms - row["mean_step_seconds"] * 1000
                details.append(dict(window=name, peak_increment_mib=delta, step_ms_saved=saved_ms))
            delta = sum(row["peak_increment_mib"] for row in details) / len(details)
            saved = sum(row["step_ms_saved"] for row in details) / len(details)
            efficiency = saved / delta if delta > 0 else None
            ranking.append(dict(site=site, peak_increment_mib=delta, step_ms_saved=saved,
                                efficiency=efficiency, observations=details))
        ranking.sort(key=lambda row: (not (row["step_ms_saved"] > 0 and row["peak_increment_mib"] <= 0),
                                     -(row["efficiency"] if row["efficiency"] is not None else row["step_ms_saved"]),
                                     SITES.index(row["site"])))
        if ranking:
            result[method] = ranking
    return result


def uniformly_spaced(count):
    return [SITES[round((index + 0.5) * len(SITES) / count - 0.5)] for index in range(count)]


def combination():
    _, anchors, d0 = stage("initial")
    plan, rows, d1 = stage("screen")
    settings = read(PUBLIC / "screen_plan.json")
    rankings = rank(rows, anchors, settings)
    jobs = []
    configurations = {}
    windows = [window for window in development_windows() if dev_site(window) in WINDOWS]
    for method, ranking in rankings.items():
        configs = {}
        for count in SET_SIZES:
            if len(ranking) >= count:
                configs[f"rank{count}"] = [row["site"] for row in ranking[:count]]
            configs[f"uniform{count}"] = uniformly_spaced(count)
        for label, sites in configs.items():
            token = f"{method}_{label}"
            configurations[token] = dict(retention=method, sites=sites, operation=settings["operation"] or "attention")
            for repeat in (1, 2):
                for window in windows:
                    jobs.append(design_job(f"combo_{dev_site(window)}_{token}_{repeat}", "combination", settings["camera"],
                                           window, steps=2, **configurations[token]))
    random.Random(ORDER_SEED + 1).shuffle(jobs)
    authority("combination", "screen", {"initial": d0, "screen": d1}, jobs=jobs, rankings=rankings,
              camera=settings["camera"], configurations=configurations,
              parent_job_count=len(plan["jobs"]))


def combination_results(initial, rows, settings):
    summary = {}
    for token, config in settings["configurations"].items():
        observed = [[rows.get(f"combo_{name}_{token}_{repeat}") for repeat in (1, 2)] for name in WINDOWS]
        if any(None in pair for pair in observed):
            continue
        baselines = [[initial.get(f"baseline_{name}_native_{repeat}") for repeat in (1, 2)] for name in WINDOWS]
        if any(None in pair for pair in baselines):
            continue
        peaks = [peak(pair) for pair in observed]
        times = [median(row["scene_seconds"] for row in pair) /
                 median(row["scene_seconds"] for row in control) for pair, control in zip(observed, baselines)]
        normalized_peak = [used / peak(reference) for used, reference in zip(peaks, baselines)]
        summary[token] = dict(configuration=config, peaks=peaks, mean_time_ratio=sum(times) / len(times),
                              time_ratios=times, peak_ratios=normalized_peak, fits_native=max(normalized_peak) <= 1)
    return summary


def development():
    _, initial, d0 = stage("initial")
    _, rows, d1 = stage("combination")
    settings = read(PUBLIC / "combination_plan.json")
    options = combination_results(initial, rows, settings)
    ranked = [token for token in options if "_rank" in token]
    fitting = [token for token in ranked if options[token]["fits_native"]]
    key = lambda token: (options[token]["mean_time_ratio"], max(options[token]["peak_ratios"]),
                         len(options[token]["configuration"]["sites"]), token)
    chosen = (min(fitting, key=key) if fitting else
              min(ranked, key=lambda token: (max(options[token]["peak_ratios"]), *key(token))) if ranked else None)
    uniform = None
    differences = None
    if chosen:
        target = options[chosen]
        same = [token for token in options if "_uniform" in token
                and options[token]["configuration"]["retention"] == target["configuration"]["retention"]]
        affordable = [token for token in same if all(a <= b
                      for a, b in zip(options[token]["peaks"], target["peaks"]))]
        if affordable:
            uniform = min(affordable, key=lambda token: (
                max(abs(a - b) for a, b in zip(options[token]["peaks"], target["peaks"])),
                len(options[token]["configuration"]["sites"]), options[token]["configuration"]["sites"],
            ))
            differences = [(a - b) / 2**20 for a, b in zip(options[uniform]["peaks"], target["peaks"])]
    qualification = read(PUBLIC / "qualification.json")
    arms = {policy: {} for policy in ("native", "camera", "last") if policy in qualification["eligible"]}
    arms["native"] = {}
    if chosen:
        config = options[chosen]["configuration"]
        arms["retention"] = config
        combined = "combined" if settings["camera"] == "last" else "camera_retention"
        arms[combined] = config
        if uniform:
            arms["uniform"] = dict(options[uniform]["configuration"], camera=(
                "last" if combined == "combined" else "checkpoint"))
    jobs = []
    for policy, config in arms.items():
        if config:
            jobs.append(job(f"matched_{policy}", "qualification", policy, probe="real", width=294, steps=2,
                            gradient_mode="replay", backend="production", source_job="grad_production_capture",
                            **config))
    rng = random.Random(ORDER_SEED + 2)
    for window in development_windows():
        name = dev_site(window)
        for seed in DEV_SEEDS:
            group = [design_job(f"dev_{name}_{policy}_{seed}", "development", policy, window, seed=seed,
                                score=policy == "native" and name != "heads", **config)
                     for policy, config in arms.items()]
            if name != "heads":
                group.append(design_job(f"dev_{name}_repeat_{seed}", "repeatability", "native", window,
                                        seed=seed, score=True))
            rng.shuffle(group)
            jobs.extend(group)
    authority("development", "combination", {"initial": d0, "combination": d1},
              extra_authorities=("qualification.json",), jobs=jobs,
              selected_combination=chosen, chosen_uniform=uniform, uniform_difference_mib=differences,
              uniform_within_64_mib=differences is not None and max(map(abs, differences)) <= 64,
              resource_screen=options)


def lock():
    plan, runs, digest = stage("development")
    _, _, qualification_digest = stage("initial")
    rows = [dict(result=runs.get(item["id"])) for item in plan["jobs"]]
    resources = resources_by_policy(plan["jobs"], rows)
    qualification = read(PUBLIC / "qualification.json")
    selected_checks = read(PUBLIC / "retention_qualification.json")
    allowed = (set(qualification["eligible"]) | set(selected_checks["eligible"])) - {"native", "camera", "uniform"}
    candidate, reason = choose(resources, allowed)
    config = {}
    if candidate:
        first = next(job for job in plan["jobs"] if job["policy"] == candidate)
        config = {name: first[name] for name in ("camera", "retention", "sites", "operation")}
    authority("locked", "development", {"development": digest, "initial": qualification_digest},
              extra_authorities=("qualification.json", "retention_qualification.json"),
              candidate=candidate, reason=reason, configuration=config, qualified=sorted(allowed),
              resource_selection=resources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "combination", "development", "lock"))
    stage_name = parser.parse_args().stage
    {"screen": screen, "combination": combination, "development": development, "lock": lock}[stage_name]()
