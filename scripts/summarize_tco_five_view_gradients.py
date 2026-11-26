"""Read private CPU tensors and produce compact registered real-gradient comparisons."""

import argparse
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from run_profiled_d0 import read, sha, write_new
from tco_five_view_gradients import array_stats, compare_cases, compare_q_cases, load, tensor_hash


ROOT = Path(__file__).resolve().parents[1]
STUDY = "tco_five_view_replay_cache_v1"
PRIVATE = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY


def registrations():
    initial = ROOT / "manifests" / f"{STUDY}.json"
    stages = [initial, *sorted(initial.parent.glob("tco_five_view_replay_cache_*_v1.json"))]
    jobs, sources = {}, {}
    for stage in stages:
        plan = read(stage)
        for item in plan["jobs"]:
            if item.get("gradient_mode"):
                if item["id"] in jobs:
                    raise ValueError(f"Repeated registered gradient case {item['id']}")
                jobs[item["id"]] = item
                sources[stage.name] = sha(stage, text=True)
    return jobs, sources


@lru_cache(maxsize=None)
def receipt(identifier):
    attempts = sorted((PRIVATE / "launches" / identifier).glob("attempt*/terminal.json"))
    entries = [read(path) for path in attempts]
    successes = [row for row in entries if row["verdict"] == "completed"]
    if not successes:
        return None, None, {"attempts": [row["verdict"] for row in entries]}
    successful, = successes
    folder = PRIVATE / "cases" / identifier / successful["attempt"] / "case"
    path = folder / "complete.json"
    if sha(path) != successful["complete_sha256"] or successful["job"] != identifier:
        raise ValueError(f"Unverified terminal receipt for {identifier}")
    record = read(path)
    if record["rental_compute_usd"] != 0 or record["id"] != identifier:
        raise ValueError(f"Invalid case receipt for {identifier}")
    for name, digest in record.get("artifact_sha256", {}).items():
        if Path(name).name != name or sha(folder/name) != digest:
            raise ValueError(f"Unverified {identifier}/{name}")
    summary_file = folder / "gradient_summary.json"
    summary = read(summary_file) if summary_file.is_file() else None
    return folder, summary, dict(attempts=[row["verdict"] for row in entries],
                                  complete_sha256=sha(path),
                                  gradient_summary_sha256=sha(summary_file) if summary else None)


def ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator/denominator


@lru_cache(maxsize=None)
def model_comparison(left, right, anchor):
    return compare_cases(receipt(left)[0], receipt(right)[0], receipt(anchor)[0])


def build(jobs, manifest_hashes):
    records = {name: receipt(name) for name in jobs}
    completed = {name for name, (path, _, _) in records.items() if path is not None}
    out = dict(study=STUDY, manifest_sha256_lf=manifest_hashes,
               runs={name: values[2] for name, values in records.items()}, comparisons={},
               detached={}, interpretation={}, rental_compute_usd=0)
    q_comparisons = {}

    def q_diff(left, right):
        key = left, right
        if key not in q_comparisons:
            q_comparisons[key] = compare_q_cases(records[left][0], records[right][0])
        return q_comparisons[key]

    def compare(left, right, anchor, label, matched=True):
        key = f"{left}__vs__{right}"
        if key in out["comparisons"]:
            return out["comparisons"][key]["updates"]
        if matched:
            for step in ("1", "2"):
                lhs, rhs = (records[name][1]["matched_updates"][step] for name in (left, right))
                if lhs["pre_state_sha256"] != rhs["pre_state_sha256"] or lhs["boundary"] != rhs["boundary"]:
                    raise ValueError(f"Incomparable {key} state, update {step}")
        values = {step: dict(families) for step, families in model_comparison(left, right, anchor).items()}
        for step in ("1", "2"):
            # Free-running controls generally have different incoming factors on step 2.
            if not matched and step == "2":
                values[step].pop("delta")
        out["comparisons"][key] = dict(purpose=label, anchor=anchor, matched_state=matched, updates=values)
        return values

    for name in sorted(completed):
        row = jobs[name]
        if row["gradient_mode"] == "regenerate":
            out["detached"][name] = records[name][1]["matched_updates"]
    controls = sorted(n for n in completed if jobs[n]["gradient_mode"] == "control")
    native = "grad_production_capture"
    for a, b in combinations([n for n in [native, *controls] if n in completed], 2):
        compare(b, a, native, "independent ordinary native trajectories", matched=False)

    groups = sorted({(j["backend"], j["source_job"]) for j in jobs.values()
                     if j["gradient_mode"] == "replay" and j["cohort"] == "gradient"})
    for backend, anchor in groups:
        if anchor not in completed:
            continue
        siblings = sorted(n for n in completed if jobs[n].get("source_job") == anchor
                          and jobs[n]["backend"] == backend and jobs[n]["cohort"] == "gradient")
        identical = [n for n in siblings if jobs[n]["gradient_mode"] == "replay"
                     and not jobs[n].get("q_source_job")]
        varied = [n for n in siblings if jobs[n]["gradient_mode"] == "replay" and jobs[n].get("q_source_job")]
        same_pairs = [compare(b, a, anchor, "identical signed derivatives") for a,b in combinations(identical, 2)]
        tag = f"{backend}__{anchor}"
        observations = out["interpretation"][tag] = {
            "identical_completed": len(identical), "independent_q_completed": len(varied),
            "fixed_q_comparisons": len(same_pairs), "steps": {},
        }
        if not identical:
            continue
        # Eager full loss backward vs standalone multi-output backward can change graph scheduling.
        compare(identical[0], anchor, anchor, "same q, standalone backward vs full native loss backward")
        for step in ("1", "2"):
            self_errors = [res[step]["preclip"] for res in same_pairs]
            observations["steps"][step] = dict(
                all_identical_q_pairs_pass=(all(x["passed"] for x in self_errors) if self_errors else None),
                largest_identical_q_diff_norm=max((x["diff_norm"] for x in self_errors), default=None),
                injected=[],
            )
        for left, right in [*[(v,identical[0]) for v in varied], *[(b,a) for a,b in combinations(varied,2)]]:
            lhs = jobs[left].get("q_source_job") or anchor
            rhs = jobs[right].get("q_source_job") or anchor
            qrows = q_diff(lhs, rhs)
            grows = compare(left, right, anchor, "two signed derivatives at the same incoming state")
            for step in ("1", "2"):
                q, g = qrows[step]["overall"], grows[step]["preclip"]
                observations["steps"][step]["injected"].append(dict(
                    actual=left, reference=right, q_overall=q,
                    g_diff_norm=g["diff_norm"], g_relative_l2=g["relative_l2"],
                    measured_absolute_ratio=ratio(g["diff_norm"], q["diff_norm"]),
                    measured_relative_ratio=ratio(g["relative_l2"], q["relative_l2"])))
        # A sign error should exceed native self variation. Report both criteria explicitly.
        for sign in (n for n in siblings if jobs[n]["gradient_mode"] == "negative"):
            values = compare(sign, identical[0], anchor, "flipped one active derivative")
            for step in ("1", "2"):
                bucket = observations["steps"][step]
                val = values[step]["preclip"]
                noise = bucket["largest_identical_q_diff_norm"]
                bucket.update(sign_negative_outside_tolerance=not val["passed"],
                              sign_negative_above_observed_native_spread=val["diff_norm"] > noise if noise is not None else None,
                              sign_negative_flipped=records[sign][1]["matched_updates"][step]["flipped"])
        for name in sorted(completed):
            j = jobs[name]
            if (j["cohort"] != "gradient" and j.get("source_job") == anchor and j["backend"] == backend
                    and j["gradient_mode"] == "replay"):
                compare(name, identical[0], anchor, "method replay against native at identical signed q")
    return out


def record(identifier):
    path, _, _ = receipt(identifier)
    return read(path / "complete.json") if path else None


def camera_vjp(left, right, step):
    filename = f"camera_token_{step}.pt"
    a, b = (load(receipt(case)[0] / filename) for case in (left,right))
    if a["update"] != step or b["update"] != step:
        raise ValueError("Camera-token archive update mismatch")
    return dict(same_tokens=tensor_hash(a["camera_tokens"]) == tensor_hash(b["camera_tokens"]),
                signed_gradient=array_stats(a["signed_gradient"], b["signed_gradient"]))


def envelope(candidate, samples, *, tensor=False):
    """Separate passing fixed tolerances from an empirical native-repeat envelope."""
    if candidate["passed"]:
        return "within_tolerance"
    if not samples:
        return "unresolved"
    field = "relative_l2"
    def discrepancy(row):
        if tensor:
            return row["diff_norm"], row[field]
        worst = row["worst_relative"]
        return row["diff_norm"], worst[field] if worst else None
    delta, relative = discrepancy(candidate)
    observed = [discrepancy(sample) for sample in samples]
    within = (delta <= max(x for x,_ in observed) and (relative is None
              or relative <= max((r for _,r in observed if r is not None), default=0.)))
    return "within_native_spread" if within else "unresolved"


def qualification(jobs, manifest_hashes, stage):
    policies = ("native", "camera", "last") if stage == "initial" else (
        "retention", "combined", "uniform", "camera_retention")
    anchor = "grad_production_capture"
    reference = "matched_native"
    repeats = [f"grad_production_fixed_{i}" for i in (1,2,3)]
    pairs = [(b,a) for a,b in combinations(repeats,2) if record(a) and record(b)]
    cpu, gpu = record("cpu_fixture"), record("gpu_fixture")
    fixtures = bool(cpu and gpu and all(row["fixture"].get("detached_last_token_detected")
                                        for row in (cpu,gpu)))
    retention_fixture = bool(cpu and cpu["fixture"].get("retention"))
    out = dict(study=STUDY, stage=stage, eligible=[], policy_checks={}, fixtures=dict(
        camera=fixtures, retention=retention_fixture), manifest_sha256_lf=manifest_hashes,
        rental_compute_usd=0)
    if not record(anchor) or not record(reference):
        out["unavailable"] = "Native capture or matched Native replay missing"
        return out
    sign = "grad_production_sign"
    negative = model_comparison(sign, repeats[0], anchor) if record(sign) and record(repeats[0]) else {}
    native_diffs = [model_comparison(a,b,anchor) for a,b in pairs]
    sign_checks = {}
    for step in ("1","2"):
        if step in negative and native_diffs:
            x = negative[step]["preclip"]
            sign_checks[step] = (not x["passed"] and
                                 x["diff_norm"] > max(d[step]["preclip"]["diff_norm"] for d in native_diffs))
    regen_id = "grad_production_regenerate_1"
    regen = receipt(regen_id)[1]
    alpha = bool(regen and all(row.get("detached_alpha_detected") for row in regen["matched_updates"].values()))
    out.update(native_fixed_repeat_pairs=pairs, sign_negative=sign_checks, detached_opacity_negative=alpha)
    cameras = {"native":"native", "camera":"camera", "last":"last", "retention":"native",
               "combined":"last", "uniform":"last", "camera_retention":"camera"}
    for policy in policies:
        name = "matched_" + policy
        if name not in jobs or not record(name):
            out["policy_checks"][policy] = {"status":"unavailable"}
            continue
        candidate = record(name)
        identity_name = "identity_" + policy
        if record(identity_name) is None:
            identity_name = "identity_" + cameras[policy]
        leak_name = "leak_" + policy
        if record(leak_name) is None:
            leak_name = "leak_" + cameras[policy]
        ident, native_ident, leak = (record(i) for i in (identity_name,"identity_native",leak_name))
        exact_zero = bool(ident and native_ident and leak and
                          ident["input_sha256"] == native_ident["input_sha256"] == leak["input_sha256"] and
                          ident["output_sha256"] == native_ident["output_sha256"] == leak["output_sha256"] and
                          ident["reference_sha256"] != leak["reference_sha256"] and
                          ident["updates"] == native_ident["updates"] == leak["updates"] == 0)
        controls = dict(camera_fixture=fixtures, zero_step_and_leak=exact_zero,
                        frozen_hash_unchanged=candidate["audit"]["frozen_start_sha256"] ==
                        candidate["audit"]["frozen_end_sha256"], signed_negative=len(sign_checks)==2
                        and all(sign_checks.values()), detached_opacity=alpha,
                        native_depth_and_factors=candidate["native_depth_batches"] == [[5]] and
                        candidate["lora_tensors"] == 384 and candidate["updates"] == 2)
        if stage == "development":
            controls["retention_fixture"] = retention_fixture
        details, statuses = {}, []
        comp = model_comparison(name,reference,anchor)
        for step in ("1","2"):
            x, y = (receipt(i)[1]["matched_updates"][step] for i in (name,reference))
            if x["pre_state_sha256"] != y["pre_state_sha256"] or x["boundary"] != y["boundary"]:
                controls["matched_state"] = False
            token = camera_vjp(name,reference,int(step))
            controls[f"camera_tokens_step_{step}"] = token["same_tokens"]
            token_samples = [camera_vjp(a,b,int(step))["signed_gradient"] for a,b in pairs]
            token_status = envelope(token["signed_gradient"], token_samples, tensor=True)
            families = {f: envelope(row, [d[step][f] for d in native_diffs]) for f,row in comp[step].items()}
            details[step] = dict(camera=token, camera_status=token_status, families=families, differences=comp[step])
            statuses.extend([token_status, *families.values()])
        if not all(controls.values()):
            status = "failed_or_unverified_structural_controls"
        elif all(s == "within_tolerance" for s in statuses):
            status = "passed_numerical_tolerance"
        elif all(s != "unresolved" for s in statuses):
            status = "repeatability_limited"
        else:
            status = "unresolved_outside_observed_native_spread"
        if status in ("passed_numerical_tolerance","repeatability_limited"):
            out["eligible"].append(policy)
        out["policy_checks"][policy] = dict(status=status, controls=controls, updates=details,
                identity_job=identity_name, leak_job=leak_name, matched_job=name,
                identity_scope="same policy" if identity_name == "identity_"+policy else "same camera at zero updates")
    ids = set(["cpu_fixture", "gpu_fixture", reference, anchor, sign, regen_id, *repeats])
    for info in out["policy_checks"].values():
        ids.update(info[key] for key in ("identity_job","leak_job","matched_job") if key in info)
    out["source_receipts"] = {name: receipt(name)[2] for name in sorted(ids)}
    return out


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--qualification", choices=("initial","development"))
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    jobs, manifests = registrations()
    if args.qualification:
        result = qualification(jobs, manifests, args.qualification)
        if "unavailable" in result and not args.output:
            raise ValueError(result["unavailable"])
        target = args.output or PUBLIC / ("qualification.json" if args.qualification == "initial"
                                          else "retention_qualification.json")
    else:
        result = build(jobs, manifests)
        if not any(record.get("complete_sha256") for record in result["runs"].values()):
            raise ValueError("No verified gradient evidence is available")
        target = args.output or PUBLIC / "gradient_attribution.json"
    if target.exists():
        if read(target) != result:
            raise ValueError("Changed inputs or existing analysis; write to a fresh output path")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_new(target, result)
    print(f"Verified gradient analysis written to {target}")


if __name__ == "__main__":
    main()
