"""Apply the registered decoder multipliers and render their native endpoints."""
import argparse
import math
import os
import time

import numpy as np
import torch

from mast3r_network_priors import digest
from native_lora_decoder_groups import (ARCHIVED, BRIDGE, MIN_PARTITION, POLICIES, RAW, STATES,
                                        archive_case, check_endpoint, choose, group, load_depth,
                                        registered)
from native_lora_step_inputs import EXPECTED_RUNTIME, verify_registration
from profiled_runtime import configure_runtime, verify_runtime
from run_native_lora_direction import WEIGHTS, assign, model_and_factors, native_endpoint
from run_native_lora_step import load_permitted
from run_profiled_d0 import read, sha, write_new
from run_profiled_lora import scientific_gpu_lock
from score_native_lora_direction import evaluate, means


def grouped_factors(factors):
    groups = tuple([(name, p) for name, p in factors if group(name) == index] for index in (0, 1))
    for index, items in enumerate(groups):
        blocks = {(name.split(".")[0], int(name.split(".")[1])) for name, _ in items}
        expected = {(decoder, block) for decoder in ("dec_blocks", "dec_blocks2")
                    for block in range(index*6, index*6+6)}
        if blocks != expected or len(items) != 144:
            raise ValueError("Unexpected decoder projection inventory")
    return groups


def effective_norms(groups, before, after):
    norms = []
    for items in groups:
        values = []
        for name, _ in items:
            if name.endswith(".lora_B"):
                a_key = name[:-1] + "A"
                if bool(before[name].any()) or not torch.equal(before[a_key], after[a_key]):
                    raise ValueError("The first Adam step requires pristine B and unchanged A")
                a, b = after[a_key].double(), after[name].double()
                values.append(float(((a @ a.T) * (b.T @ b)).sum()) / 16)
        norm = math.fsum(values)
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid group effective norm")
        norms.append(norm)
    return norms


def assign_groups(groups, before, after, multipliers):
    for items, multiplier in zip(groups, multipliers):
        assign(items, before, after, multiplier)


def verify_assignments(factors, groups, before, after):
    for multiplier in sorted({pair[0] for pair in STATES.values() if pair[0] == pair[1]}):
        assign(factors, before, after, multiplier)
        expected = digest(factors)
        assign_groups(groups, before, after, (1.5, .5))
        assign_groups(groups, before, after, (multiplier, multiplier))
        if digest(factors) != expected:
            raise ValueError("Diagonal assignment differs from legacy allocation")
        if multiplier in (0, 1):
            state = before if multiplier == 0 else after
            if expected != digest((name, state[name]) for name, _ in factors):
                raise ValueError("Zero or full assignment did not restore the exact snapshot")


def render(key, pair, destination, model, fitting, bundle, permitted, factor_sha):
    destination.mkdir()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    gauge, tree, graph = native_endpoint(model, fitting, bundle, destination / "cache", destination)
    selector_masks = [p["fold"] == 2 for p in permitted]
    depth, _ = load_depth(destination, {"g0": gauge})
    rows = evaluate(depth, permitted, selector_masks, gauge, MIN_PARTITION)
    if any(row["mae_mm"] is None for row in rows):
        raise ValueError("Invalid fixed selector sample")
    record = dict(candidate=key, multipliers=list(pair), source="fresh", adapter_sha256=factor_sha,
                  g0=gauge, native_sha256=sha(destination / "native.npz"),
                  trace_sha256=sha(destination / "trace.json"), tree=tree, match_edges=graph,
                  selector_rows=rows, selector_mae_mm=means(rows), wall_seconds=time.monotonic()-start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    write_new(destination / "complete.json", record)
    return record


def compare_bridge(now_path, prior_path, now, prior):
    for field in ("adapter_sha256", "g0", "selector_rows", "tree", "match_edges"):
        if now[field] != prior[field]:
            raise ValueError(f"Native bridge disagrees on {field} at {now_path.name}")
    with np.load(now_path / "native.npz") as observed, np.load(prior_path / "native.npz") as replayed:
        for name in ("depth", "c2w", "intrinsics", "g0"):
            np.testing.assert_array_equal(observed[name], replayed[name])


def mechanism(output, records, permitted):
    states = ("s050", "s100", "s150", "e050_l150", "e150_l050")
    matched = {}
    for key in states:
        depth, scale = load_depth(output(key), records[key])
        pixels = [p["xy"][p["fold"] == 2] for p in permitted]
        matched[key] = [z[xy[:, 1], xy[:, 0]].astype(np.float64)*scale*1000 for z, xy in zip(depth, pixels)]
    reference = [a-b for a, b in zip(matched["s150"], matched["s050"])]
    results = {}
    for key in ("s050", "s150", "e050_l150", "e150_l050"):
        rows = []
        for view, (a, b, r) in enumerate(zip(matched[key], matched["s100"], reference)):
            delta = a - b
            product = float(np.linalg.norm(delta) * np.linalg.norm(r))
            rows.append(dict(view=view, count=len(delta), mean_signed_mm=float(delta.mean()),
                             rms_mm=float(np.sqrt(np.mean(delta**2))),
                             cosine_to_scalar=float(np.dot(delta, r)/product) if product else None,
                             signed_delta_mm=delta.tolist()))
        results[key] = rows
    return results


def run(job, attempt):
    contract = registered()
    case = next(case for case in contract["development_cases"] if case["id"] == job)
    if (os.environ.get("PROFILED_D0_IMAGE_ID") != contract["image"] or
            sha(WEIGHTS / "model.safetensors") != contract["checkpoint_sha256"] or
            sha(WEIGHTS / "config.json") != contract["config_sha256"]):
        raise ValueError("Pinned image, checkpoint or config differs")
    configure_runtime()
    ref = read(EXPECTED_RUNTIME)
    if (torch.__version__, torch.version.cuda, torch.cuda.get_device_name()) != (ref["torch"], ref["cuda"], ref["gpu"]):
        raise ValueError("Unexpected Torch, CUDA or GPU")
    fitting, bundle, permitted = load_permitted(verify_registration(), case["env"], case["side"])
    output = RAW / "cases" / job / attempt
    output.mkdir(parents=True, exist_ok=False)
    prior_path, prior = archive_case(job)
    start = time.monotonic()
    model, factors, original, original_sha = model_and_factors()
    groups = grouped_factors(factors)
    snapshots = []
    for label in ("before", "after"):
        with np.load(prior_path / f"{label}.npz", allow_pickle=False) as stored:
            if set(stored.files) != {name for name, _ in factors}:
                raise ValueError("Stored factor identity differs")
            snapshots.append({name: torch.from_numpy(stored[name]) for name, _ in factors})
    before, after = snapshots
    if digest(factors) != digest((name, before[name]) for name, _ in factors):
        raise ValueError("Loaded model does not reproduce the historical pristine factors")
    norms = effective_norms(groups, before, after)
    verify_assignments(factors, groups, before, after)
    records = {}

    def compute(key, pair):
        assign_groups(groups, before, after, pair)
        factor_sha = digest(factors)
        row = render(key, pair, output / key, model, fitting, bundle, permitted, factor_sha)
        if digest(factors) != factor_sha or digest(original) != original_sha:
            raise ValueError("Adapter or frozen backbone changed during native alignment")
        print(job, key, "native alignment complete", flush=True)
        return row

    for key in ARCHIVED:
        reference = prior["states"][ARCHIVED[key]]
        prior_endpoint = prior_path / ARCHIVED[key]
        check_endpoint(prior_endpoint, reference)
        if job == BRIDGE:
            records[key] = compute(key, STATES[key])
            compare_bridge(output / key, prior_endpoint, records[key], reference)
        else:
            assign_groups(groups, before, after, STATES[key])
            if digest(factors) != reference["adapter_sha256"]:
                raise ValueError("Historical diagonal factors differ from the registered update")
            records[key] = {k: v for k, v in reference.items() if k not in ("state", "multiplier")}
            records[key].update(candidate=key, multipliers=list(STATES[key]), source="archive")
    bridge = None
    if job == BRIDGE:
        repeat = compute("repeat_full", (1, 1))
        a, _ = load_depth(output / "s100", records["s100"])
        b, _ = load_depth(output / "repeat_full", repeat)
        magnitude = float(np.max(np.abs(a.astype(np.float64)*records["s100"]["g0"] -
                                        b.astype(np.float64)*repeat["g0"]))*1000)
        bridge = dict(historical_diagonals_verified=list(ARCHIVED), repeat=repeat,
                      max_dense_metric_difference_mm=magnitude)
    for key in (key for key in STATES if key not in ARCHIVED):
        records[key] = compute(key, STATES[key])
    errors = {key: row["selector_mae_mm"] for key, row in records.items()}
    selected = {name: choose(errors, norms, policy) for name, policy in POLICIES.items()}

    def target(key):
        return output / key if records[key]["source"] == "fresh" else prior_path / ARCHIVED[key]

    diagnostics = mechanism(target, records, permitted)
    verify_runtime()
    write_new(output / "choice.json", dict(job=job, attempt=attempt, selected=selected, states=records,
              group_norms_sq=norms, original_sha256=original_sha, prior_choice_sha256=prior["choice_sha256"],
              factor_files=prior["direction"]["factor_files"], bridge=bridge, mechanism=diagnostics,
              wall_seconds=time.monotonic()-start, rental_compute_usd=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("attempt")
    args = parser.parse_args()
    with scientific_gpu_lock(dict(study="native-lora-decoder-groups", case=args.job)):
        run(args.job, args.attempt)
