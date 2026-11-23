"""Choose an integer full-TCO step budget using timings before its scores exist."""

import argparse
from statistics import mean

from replay_tco_one_percent import REFERENCE, analyze_development
from run_profiled_d0 import sha, write_new
from run_profiled_d1_local import host_lock
from run_tco_one_percent_local import launch
from tco_one_percent_protocol import MANIFEST, PUBLIC, SCENES, SEEDS, TIME_MANIFEST
from tco_one_percent_protocol import case_receipts, collect, job_for, read_committed, verify


def register():
    plan = verify()
    qualification = read_committed(PUBLIC / "qualification.json")
    jobs = [job for job in plan["jobs"] if job["arm"] != "cached14" or qualification["last14_screen"]["eligible"]]
    decision = analyze_development(plan, qualification, collect(jobs))
    if decision["verdict"] != "needs_equal_time":
        raise ValueError("Development has no complete candidate for a timing search")
    stats = decision["comparisons"]
    targets = {arm: stats[arm]["ratios"]["scene_seconds"] for arm, result in decision["candidates"].items()
               if result["qualifies_for_timing"]}
    measured = {arm: ratios["ratios"]["scene_seconds"] for arm, ratios in stats.items()
                if arm.startswith("four_") or arm == "zero_0"}
    pilots = [job_for(scene, "four", steps, SEEDS[0], "timing", probe="timing")
              for steps in range(1, 40) if steps not in (5, 10) for scene in SCENES]
    write_new(TIME_MANIFEST, dict(
        development_sha256=sha(MANIFEST, text=True), search_script_sha256=sha(__file__, text=True),
        target_ratios=targets, fixed_arm_time_ratios=measured,
        full40_scene_times={scene: vals["scene_seconds"]
                            for scene, vals in stats[REFERENCE]["medians_by_sequence"].items()},
        rule="Largest fitting tested step count using fixed controls and adjacent unscored bracketing pilots; "
             "choose interior probes by interpolating between the measured step/time endpoints",
        input_source_sha256=plan["input_source_sha256"], jobs=pilots))
    print("Registered optional single-seed timing pilots in the three development rooms")


def bracket(measured, target):
    # Zero is the lower bound when even the untuned anchor is above the target.
    low = max((step for step, duration in measured.items() if duration <= target), default=0)
    high = min(step for step, duration in measured.items() if step > low and duration > target)
    return low, high


def interpolate(measured, low, high, target):
    start = measured[low]
    finish = measured[high]
    if not start <= target < finish:
        return low + 1
    step = round(low + (high - low) * (target - start) / (finish - start))
    return min(high - 1, max(low + 1, step))


def search(plan, extension):
    if extension["development_sha256"] != sha(MANIFEST, text=True) or (
            extension["search_script_sha256"] != sha(__file__, text=True)):
        raise ValueError("Timing extension differs from the committed method or registration")
    if "locked_by_candidate" in extension:
        raise ValueError("Equal-time budgets are already locked")
    times = {int(arm.rsplit("_", 1)[-1]): duration for arm, duration in extension["fixed_arm_time_ratios"].items()}
    old_steps = set(times)
    with host_lock():
        while True:
            brackets = {arm: bracket(times, target) for arm, target in extension["target_ratios"].items()}
            unsettled = sorted((extension["target_ratios"][arm], low, high)
                               for arm, (low, high) in brackets.items() if high - low > 1)
            if not unsettled:
                locked = {arm: low for arm, (low, _) in brackets.items()}
                break
            target, low, high = unsettled[0]
            midpoint = interpolate(times, low, high, target)
            jobs = [job for job in extension["jobs"] if job["steps"] == midpoint]
            for job in jobs:
                rows, attempts = case_receipts(job)
                while not rows and len(attempts) < 2:
                    launch(job, "time", f"attempt0{len(attempts) + 1}", plan["launch_timeout_seconds"])
                    rows, attempts = case_receipts(job)
                if not rows:
                    raise ValueError(f"Timing search lacks a successful local case for {job['id']}")
            evidence = collect(jobs)
            times[midpoint] = mean(row["scene_seconds"] / extension["full40_scene_times"][row["scene"]]
                                   for row in evidence["rows"])
    observed = collect(extension["jobs"])
    write_new(PUBLIC / "timing_evidence.json", observed)
    selected_steps = set(locked.values())
    # Benchmark the adjacent smaller budget too when timing jitter could make the selected arm ineligible.
    smaller = {step - 1 for step in selected_steps if step > 1 and step not in old_steps}
    scored = [job_for(scene, "four", steps, seed, "equal_time")
              for steps in sorted((selected_steps | smaller) - old_steps) for seed in SEEDS for scene in SCENES]
    locked_extension = dict(**extension, locked_by_candidate=locked, pilot_time_ratios={
        str(steps): duration for steps, duration in sorted(times.items()) if steps not in old_steps})
    locked_extension["jobs"] = [*extension["jobs"], *scored]
    staged = TIME_MANIFEST.with_suffix(".locked.json")
    write_new(staged, locked_extension)
    print(f"Selected budgets {locked}. Commit {staged.name} at {TIME_MANIFEST.name} before scoring.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("register", "search"))
    args = cli.parse_args()
    if args.action == "register":
        register()
    else:
        search(verify(), read_committed(TIME_MANIFEST))
