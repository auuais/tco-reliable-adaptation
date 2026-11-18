"""Register and score the matched MASt3R decoder group development trial."""
import argparse
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from native_lora_step_inputs import (BASE_GAIN, EPSILON_MM, MIN_HELD, MIN_PARTITION, ROOMS,
                                     RAW as PREVIOUS_RAW, PUBLIC as PREVIOUS_PUBLIC, array_digest,
                                     cases, verify_registration)
from run_profiled_d0 import IMAGE, read, sha, write_new
from score_native_lora_direction import data_packet, evaluate, held_mask, means

ROOT = Path(__file__).resolve().parents[1]
STUDY = "native_lora_decoder_groups_v1"
RAW = ROOT / "reports" / STUDY
PUBLIC = ROOT / "results" / STUDY
MANIFEST = ROOT / "manifests" / f"{STUDY}.json"
SEAL = PUBLIC / "selection_seal.json"
SCORES = PUBLIC / "held_scores.json"
PREVIOUS_SEAL = PREVIOUS_PUBLIC / "selection_seal.json"
PREVIOUS_SCORES = PREVIOUS_PUBLIC / "held_scores.json"
BRIDGE = "apt1_earlier"
ARCHIVED = dict(s000="zero", s025="quarter", s050="half", s100="full")
COMMON = dict(s000=(0, 0), s025=(.25, .25), s050=(.5, .5), s100=(1, 1), s150=(1.5, 1.5))
POLICIES = dict(scalar={**COMMON, "s075": (.75, .75), "s125": (1.25, 1.25)},
                split={**COMMON, "e050_l150": (.5, 1.5), "e150_l050": (1.5, .5)})
STATES = POLICIES["scalar"] | POLICIES["split"]
CONTROLS = ("s000", "s100", "legacy_selected")
EXPECTED_SOURCES = (
    "scripts/native_lora_decoder_groups.py", "scripts/run_native_lora_decoder_groups.py",
    "scripts/run_native_lora_decoder_groups_local.py", "scripts/replay_native_lora_decoder_groups.py",
    "scripts/test_native_lora_decoder_groups.py", "docs/MAST3R_DECODER_GROUP_HANDOFF.md",
)


def group(name):
    match = re.fullmatch(r"dec_blocks2?\.(\d+)\..+\.lora_[AB]", name)
    if not match or not 0 <= int(match[1]) < 12:
        raise ValueError(f"Unexpected LoRA factor: {name}")
    return int(match[1]) // 6


def choose(errors, norms, candidates):
    if any(not math.isfinite(errors[key]) for key in candidates):
        raise ValueError("Missing or nonfinite selector error")
    near = min(errors[key] for key in candidates) + EPSILON_MM
    return min((key for key in candidates if errors[key] <= near),
               key=lambda key: (math.fsum(w*w*n for w, n in zip(candidates[key], norms)),
                                candidates[key]))


def execution_contract():
    previous = verify_registration()
    return dict(
        schema=STUDY, prior_revision="eb7fde7ae653eae8d8c7409476f871e78fe0c1e7",
        image=IMAGE, checkpoint_sha256=previous["checkpoint_sha256"],
        config_sha256=previous["config_sha256"],
        source_sha256_lf={name: sha(ROOT / name, text=True) for name in EXPECTED_SOURCES},
        protocol_sha256_lf=previous["source_sha256_lf"]["docs/NATIVE_LORA_STEP_SELECTION.md"],
        development_inputs_sha256=previous["input_sha256"],
        previous_seal_sha256_lf=sha(PREVIOUS_SEAL, text=True),
        previous_scores_sha256_lf=sha(PREVIOUS_SCORES, text=True),
        development_cases=[dict(id=f"{env}_{side}", env=env, side=side, room=ROOMS[env],
                                packet=previous["inputs"]["environments"][env]["packets"][side])
                           for env, side in cases()],
        policies={name: {key: list(value) for key, value in policy.items()}
                  for name, policy in POLICIES.items()},
        bridge=BRIDGE, archived=ARCHIVED, candidates_per_policy=7,
        development_endpoints=54, development_bridge_repeat_endpoints=1,
        fresh_endpoints_if_four_rooms=72, selection_tolerance_mm=EPSILON_MM,
        min_partition=MIN_PARTITION, min_held=MIN_HELD, hindsight_relative_gain=BASE_GAIN,
        fresh=dict(environment="office1", lock_until_development_pass=True, sparse_relative_gain=BASE_GAIN,
                   dense_shape=[384, 512], dense_min_grid_fraction=.1, dense_source_depth_m=[.2, 10.],
                   dense_mask="exclude both sparse packets, test one-pixel dilation as sensitivity"),
        rental_compute_usd=0)


def registered():
    item = read(MANIFEST)
    if item != execution_contract():
        raise ValueError("Execution contract differs from the registered study")
    committed = subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(ROOT),
                                        "show", f"HEAD:manifests/{STUDY}.json"])
    if committed.replace(b"\r\n", b"\n") != MANIFEST.read_bytes().replace(b"\r\n", b"\n"):
        raise ValueError("Commit the registered study before execution")
    return item


def archive_case(job):
    prior = read(PREVIOUS_SEAL)["cases"][job]
    base = PREVIOUS_RAW / "cases" / job / prior["attempt"]
    if sha(base / "choice.json") != prior["choice_sha256"]:
        raise ValueError(f"Prior case receipt changed for {job}")
    for name, digest in prior["direction"]["factor_files"].items():
        if sha(base / f"{name}.npz") != digest:
            raise ValueError(f"Prior {name} snapshot changed for {job}")
    return base, prior


def check_endpoint(base, row):
    if (not math.isfinite(row["g0"]) or row["g0"] <= 0 or
            row["selector_mae_mm"] != means(row["selector_rows"])):
        raise ValueError("Invalid metric gauge or selector score")
    for name, field in (("native.npz", "native_sha256"), ("trace.json", "trace_sha256")):
        if sha(base / name) != row[field]:
            raise ValueError(f"Native evidence changed: {base / name}")


def load_depth(base, row):
    with np.load(base / "native.npz", allow_pickle=False) as output:
        depth, gauge = output["depth"], float(output["g0"])
    if depth.shape != (5, 384, 512) or gauge != row["g0"]:
        raise ValueError("Raster shape or scale differs from the sealed reconstruction")
    return depth, gauge


def endpoint_path(path, prior_path, row):
    if row["source"] == "archive":
        return prior_path / ARCHIVED[row["candidate"]]
    if row["source"] == "fresh":
        return path / row["candidate"]
    raise ValueError("Unknown endpoint source")


def build_seal(contract):
    results = {}
    for case in contract["development_cases"]:
        job = case["id"]
        receipts = sorted((RAW / "launches" / job).glob("*/terminal.json"))
        attempts = [read(p) for p in receipts]
        successes = [row for row in attempts
                     if row["stopped"] and row["exit_code"] == 0 and not row["error"]]
        if len(successes) != 1:
            raise ValueError(f"Expected one successful stopped run for {job}")
        launch = successes[0]
        if launch["job"] != job:
            raise ValueError("Launch belongs to another case")
        path = RAW / "cases" / job / launch["attempt"]
        prior_path, prior = archive_case(job)
        receipt = read(path / "choice.json")
        if (receipt["job"] != job or receipt["attempt"] != launch["attempt"] or
                receipt["prior_choice_sha256"] != prior["choice_sha256"] or
                receipt["factor_files"] != prior["direction"]["factor_files"] or
                set(receipt["states"]) != STATES.keys() or receipt["original_sha256"] != prior["original_sha256"]):
            raise ValueError("Run receipt has mismatched case, factors, or candidate inventory")
        if (job == BRIDGE) != (receipt["bridge"] is not None):
            raise ValueError("Missing or unexpected bridge")
        errors = {}
        for key, row in receipt["states"].items():
            expected_source = "archive" if job != BRIDGE and key in ARCHIVED else "fresh"
            if (row["candidate"] != key or row["multipliers"] != list(STATES[key]) or
                    row["source"] != expected_source):
                raise ValueError("Run receipt has mismatched endpoint identity")
            path_to_endpoint = endpoint_path(path, prior_path, row)
            check_endpoint(path_to_endpoint, row)
            if expected_source == "archive":
                ref = prior["states"][ARCHIVED[key]]
                if any(row[field] != ref[field] for field in ref if field not in ("state", "multiplier")):
                    raise ValueError("Archived endpoint changed since the prior study")
            elif row != read(path_to_endpoint / "complete.json"):
                raise ValueError("Fresh endpoint differs from its completion receipt")
            errors[key] = row["selector_mae_mm"]
        if job == BRIDGE:
            repeat = receipt["bridge"]["repeat"]
            if (repeat["candidate"] != "repeat_full" or repeat["source"] != "fresh" or
                    repeat["multipliers"] != [1, 1] or
                    receipt["states"]["s100"]["adapter_sha256"] != repeat["adapter_sha256"] or
                    read(path / "repeat_full/complete.json") != repeat):
                raise ValueError("Full-step repeat identity differs")
            check_endpoint(path / "repeat_full", repeat)
        expected = {name: choose(errors, receipt["group_norms_sq"], policy)
                    for name, policy in POLICIES.items()}
        if receipt["selected"] != expected:
            raise ValueError("Sealed choices must depend only on adaptation scores")
        retired = [dict(attempt=a["attempt"], exit_code=a["exit_code"], error=a["error"],
                        stopped=a["stopped"], wall_seconds=a["wall_seconds"])
                   for a in attempts if a != launch]
        results[job] = dict(**receipt, receipt_sha256=sha(path / "choice.json"),
                            launch_wall_seconds=launch["wall_seconds"], retired_attempts=retired)
    return dict(manifest_sha256_lf=sha(MANIFEST, text=True), cases=results, rental_compute_usd=0)


def summarize(scored):
    directional = {}
    for side in ("earlier", "later"):
        subset = [entry["summary_mm"] for entry in scored.values() if entry["side"] == side]
        names = ("scalar_best", "split_best", "scalar_selected", "split_selected", *CONTROLS)
        averages = {name: math.fsum(c[name] for c in subset)/len(subset)
                    if all(c[name] is not None for c in subset) else None for name in names}

        def gain(a, b):
            return ((averages[a] - averages[b])/averages[a]
                    if averages[a] and averages[b] is not None else None)

        directional[side] = dict(mean_mm=averages,
                                 hindsight_gain=gain("scalar_best", "split_best"),
                                 selected_gain=gain("scalar_selected", "split_selected"))
    valid = sum(all(v is not None for v in result["errors_mm"].values()) for result in scored.values())
    passes = valid == len(scored) and all(d["hindsight_gain"] >= BASE_GAIN and d["selected_gain"] > 0
                                          for d in directional.values())
    return dict(directional=directional, valid_cases=valid, open_office1=passes,
                verdict="advance" if passes else "stop_at_development" if valid == len(scored) else "inconclusive")


def score(contract, seal):
    previous = read(PREVIOUS_SCORES)["cases"]
    scored = {}
    for case in contract["development_cases"]:
        env, side, job = case["env"], case["side"], case["id"]
        other = "earlier" if side == "later" else "later"
        prior_manifest = read(ROOT / "manifests/native_lora_step_v1.json")
        inputs = prior_manifest["inputs"]["environments"][env]["packets"]
        base = PREVIOUS_RAW / "intake" / env
        permitted, held = [data_packet(base / f"{s}.npz", inputs[s]["sha256"]) for s in (side, other)]
        masks = [held_mask(a, b) for a, b in zip(permitted, held)]
        coverage = [dict(view=i, count=int(mask.sum()), excluded=int((~mask).sum()),
                         xy_sha256=array_digest(p["xy"][mask])) for i, (p, mask) in enumerate(zip(held, masks))]
        if coverage != case["packet"]["held_coverage"] or min(r["count"] for r in coverage) < MIN_HELD:
            raise ValueError(f"Registered held mask changed for {job}")
        receipt = seal["cases"][job]
        path = RAW / "cases" / job / receipt["attempt"]
        prior_path, _ = archive_case(job)
        rows = {}
        for key, row in receipt["states"].items():
            raster, gauge = load_depth(endpoint_path(path, prior_path, row), row)
            measurement = evaluate(raster, held, masks, gauge, MIN_HELD)
            rows[key] = dict(views=measurement, applied_scale=gauge, native_sha256=row["native_sha256"])
            if key in ARCHIVED and measurement != previous[job]["view_rows"][ARCHIVED[key]]:
                raise ValueError(f"Historical held depth did not reproduce: {job}/{key}")
        errors = {key: means(row["views"]) for key, row in rows.items()}
        best = {name: min(policy, key=lambda key: (errors[key], policy[key]))
                if all(errors[key] is not None for key in policy) else None
                for name, policy in POLICIES.items()}
        selected = receipt["selected"]
        legacy = min((key for key in ARCHIVED
                      if receipt["states"][key]["selector_mae_mm"] <=
                      min(receipt["states"][k]["selector_mae_mm"] for k in ARCHIVED) + EPSILON_MM),
                     key=lambda key: STATES[key])
        summary = {f"{name}_best": errors.get(best[name]) for name in POLICIES}
        summary.update({f"{name}_selected": errors[selected[name]] for name in POLICIES})
        summary.update(s000=errors["s000"], s100=errors["s100"], legacy_selected=errors[legacy])
        repeat_rows = None
        if job == BRIDGE:
            raster, gauge = load_depth(path / "repeat_full", receipt["bridge"]["repeat"])
            repeat_rows = evaluate(raster, held, masks, gauge, MIN_HELD)
        scored[job] = dict(room=case["room"], side=side, coverage=coverage,
                           states=rows, errors_mm=errors, hindsight_best=best,
                           selected=selected, legacy_selected=legacy, summary_mm=summary,
                           full_repeat_views=repeat_rows)
    return dict(seal_sha256_lf=sha(SEAL, text=True), cases=scored, rental_compute_usd=0, **summarize(scored))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "seal", "score"))
    mode = parser.parse_args().mode
    if mode == "register":
        write_new(MANIFEST, execution_contract())
        return
    contract = registered()
    seal = build_seal(contract)
    if mode == "seal":
        write_new(SEAL, seal)
    else:
        if read(SEAL) != seal:
            raise ValueError("Selection seal changed before held scoring")
        write_new(SCORES, score(contract, seal))


if __name__ == "__main__":
    main()
