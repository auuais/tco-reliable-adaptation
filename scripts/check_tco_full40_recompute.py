"""Check checkpoint parity and summarize the separate profiling receipts."""

import argparse
from collections import defaultdict
import json
import re
from statistics import median

import numpy as np

from run_profiled_d0 import read, sha, write_new
from tco_full40_recompute_protocol import INITIAL, POLICIES, PRIVATE, PUBLIC, REFERENCE, verify


def finished(case):
    paths = sorted((PRIVATE / "cases" / case).glob("*/case/complete.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected a single completed receipt for {case}, got {len(paths)}")
    return paths[0].parent, read(paths[0])


def compare_arrays(actual, expected, *, near_zero=1e-12):
    absolute, relative, count = 0.0, 0.0, 0
    with np.load(actual) as a, np.load(expected) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Tensor inventories differ across the parity fixture")
        for name in b.files:
            x, y = a[name].astype(np.float64), b[name].astype(np.float64)
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Invalid tensors for parity in {name}")
            delta = x - y
            absolute = max(absolute, float(np.abs(delta).max()))
            relative = max(relative, float(np.linalg.norm(delta.ravel())
                                           / max(near_zero, np.linalg.norm(y.ravel()))))
            count += 1
    return dict(tensors=count, max_abs=absolute, max_relative_l2=relative)


def qualification(candidate, same_chunk):
    fixture_path, fixture_row = finished("head_fixture")
    fixture = fixture_row["fixture"]
    if set(fixture["chunks"]) != {"8", "4", "2", "1"} or fixture["torch"].split("+")[0] != "2.7.1":
        raise ValueError("Missing pinned-container depth-head fixture")
    cases = {}
    for policy in dict.fromkeys((REFERENCE, candidate, same_chunk)):
        cases[policy] = {kind: finished(f"{kind}_{policy}") for kind in ("fixed", "identity")}
    leak_dir, leak = finished(f"leak_{candidate}")
    active_dir, active = cases[candidate]["fixed"]
    active_audit = active["audit"]
    if (leak["input_sha256"] != active["input_sha256"] or leak["reference_sha256"] == active["reference_sha256"]
            or leak["output_sha256"] != active["output_sha256"]
            or leak["audit"]["adapter_final"] != active_audit["adapter_final"]):
        raise ValueError("Changing evaluation-only geometry affected the result")
    if {key for key in active_audit["per_factor_norms"]} != {"0", "1"}:
        raise ValueError("Expected two full optimizer updates in the fixed-adjoint fixture")
    norms = active_audit["per_factor_norms"]
    for update in norms.values():
        if len(update) != 384 or not all(np.isfinite(list(update.values()))):
            raise ValueError("Invalid LoRA gradient inventory")
    if not all(any(value > 0 for name, value in norms["1"].items() if name.endswith(suffix))
               for suffix in (".lora_A", ".lora_B")):
        raise ValueError("Both LoRA factors must receive a gradient after the first update")
    matched_dir, matched = cases[same_chunk]["fixed"]
    reference_dir, reference = cases[REFERENCE]["fixed"]
    common = ("frozen_start_sha256", "frozen_end_sha256", "adapter_initial", "render_orders")
    for other in (matched, reference):
        for name in common:
            if other["audit"][name] != active_audit[name]:
                raise ValueError(f"Unexpected mismatch in {name}")
    exact_forward = all(matched["audit"][field] == active_audit[field]
                        for field in ("initial_prediction", "taps", "update_predictions"))
    factor = compare_arrays(active_dir / "lora_audit.npz", matched_dir / "lora_audit.npz")
    output = compare_arrays(active_dir / "prediction.npz", matched_dir / "prediction.npz")
    if not exact_forward or factor["max_relative_l2"] > 1e-4 or output["max_relative_l2"] > 1e-5:
        raise ValueError(f"Same-chunk end-to-end checkpoint parity failed: {exact_forward=}, {factor=}, {output=}")
    identity = cases[candidate]["identity"][0] / "prediction.npz"
    reference_identity = cases[REFERENCE]["identity"][0] / "prediction.npz"
    return dict(qualified=True, candidate=candidate, same_chunk=same_chunk, authority_sha256=sha(INITIAL, text=True),
                fixture=fixture, fixture_file_sha256=sha(fixture_path / "complete.json"),
                render_orders=active_audit["render_orders"], frozen_sha256=active_audit["frozen_end_sha256"],
                leak_output_sha256=leak["output_sha256"], leak_audit_file_sha256=sha(leak_dir / "complete.json"),
                same_chunk_forward_hashes_identical=exact_forward, same_chunk_factors=factor,
                same_chunk_predictions=output, raw_gradient_hashes_identical=matched["audit"]["gradients"] == active_audit["gradients"],
                native_fixed_factors=compare_arrays(active_dir / "lora_audit.npz", reference_dir / "lora_audit.npz"),
                native_fixed_predictions=compare_arrays(active_dir / "prediction.npz", reference_dir / "prediction.npz"),
                native_release_predictions=compare_arrays(identity, reference_identity),
                receipts={policy: {kind: sha(path / "complete.json") for kind, (path, _) in values.items()}
                          for policy, values in cases.items()}, rental_compute_usd=0)


def profile():
    summary = {}
    for case_id in ("profile_5", "profile_10"):
        for path in sorted((PRIVATE / "cases" / case_id).glob("*/case/profile.jsonl")):
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            phases = defaultdict(list)
            for row in rows:
                phases[row["phase"]].append(row)
            terminal_ooms = [row for row in rows if row["edge"] == "oom"]
            summary[f"{case_id}/{path.parent.parent.name}"] = dict(
                trace_sha256=sha(path), terminal_ooms=terminal_ooms,
                samples=len(rows), highest_global_allocated=max(row["peak_allocated"] for row in rows),
                sampled_updates=sorted({row["updates_done"] for row in rows}),
                phases={name: dict(samples=len(events), max_boundary_allocated=max(r["allocated"] for r in events),
                                   max_boundary_reserved=max(r["reserved"] for r in events),
                                   min_device_free=min(r["device_free"] for r in events))
                        for name, events in phases.items()})
    controls = {}
    for scene in ("heads", "chess"):
        try:
            samples = [finished(f"repeat_{scene}_r{repeat}")[1] for repeat in (1, 2)]
        except ValueError:
            continue
        controls[scene] = {metric: [row["metrics"][metric] * 1_000 for row in samples]
                           for metric in ("accuracy_mean_m", "completeness_mean_m")}
    return dict(timelines=summary, same_seed_errors_mm=controls, rental_compute_usd=0)


def screen():
    rows = {}
    failures = {}
    for views, prefix in ((5, "screen5"), (10, "screen")):
        rows[views] = {}
        for policy in POLICIES:
            try:
                _, result = finished(f"{prefix}_{policy}")
            except ValueError:
                for terminal in sorted((PRIVATE / "launches" / f"{prefix}_{policy}").glob("*/terminal.json")):
                    receipt = read(terminal)
                    if receipt["verdict"] == "completed":
                        raise
                    item = dict(verdict=receipt["verdict"], log_sha256=receipt["log_sha256"])
                    logfile = terminal.with_name("stdout.log")
                    if receipt["verdict"] == "oom" and sha(logfile) == receipt["log_sha256"]:
                        log = logfile.read_text(encoding="utf-8", errors="replace")
                        failure = re.search(r"^torch\.OutOfMemoryError: (.+)$", log, re.M)
                        if failure:
                            item["memory_error"] = failure.group(1)
                        last_update = re.findall(r"^\s*(\d+) \|", log, re.M)
                        item["last_loss_step"] = int(last_update[-1]) if last_update else None
                        item["phase"] = "backward" if "loss.backward()" in log else "forward"
                    failures[f"{prefix}_{policy}/{terminal.parent.name}"] = item
                continue
            rows[views][policy] = {key: result[key] for key in ("scene_seconds", "peak_allocated",
                                                              "peak_reserved", "mean_step_seconds")}
    ranks = {mode: sorted((policy for policy in rows[10] if policy.endswith(f"_{mode}")),
                          key=lambda policy: (rows[10][policy]["mean_step_seconds"],
                                              rows[10][policy]["peak_allocated"], -int(policy[1])))
             for mode in ("on", "off")}
    full = {}
    for policy in POLICIES:
        results = []
        for repeat in (1, 2, 3):
            try:
                results.append(finished(f"promote_{policy}_r{repeat}")[1])
            except ValueError:
                continue
        if results:
            full[policy] = dict(completed=len(results),
                                **{key: median(row[key] for row in results) for key in
                                   ("scene_seconds", "peak_allocated", "peak_reserved")})
    return dict(screen=rows, ranking=ranks, full_length=full, failures=failures, rental_compute_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("qualification", "profile", "screen"))
    parser.add_argument("--candidate")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    verify("initial")
    if args.action == "qualification":
        if args.candidate not in POLICIES or not args.candidate.endswith("_on"):
            parser.error("Specify a registered recomputation-on policy")
        result = qualification(args.candidate, args.candidate.replace("_on", "_off"))
    else:
        result = {"profile": profile, "screen": screen}[args.action]()
    if args.publish:
        write_new(PUBLIC / f"{args.action}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
