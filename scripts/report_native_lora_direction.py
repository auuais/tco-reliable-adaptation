"""Write the directional-study result table and summarize technical checks."""
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "results/native_lora_direction_v1"
READOUT = "held_source_scores.json"


def read(path):
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def fmt(value, digits=3):
    return "—" if value is None else f"{value:+.{digits}f}"


def report():
    scored = read(READOUT)
    seal = read("prediction_seal.json")
    audit = read("independent_replay.json")
    details, repeats = [], []
    cases = scored["cases"]
    counts = Counter()
    endpoints = []
    hours = 0.
    match_changes = 0
    minimum_held = 1000
    invalid_views = []
    for key, case in cases.items():
        sealed = seal["cases"][key]
        hours += sum(row["seconds"] for row in sealed["launches"]) / 3600
        ref = case["classification"]
        if "mae_mm" not in case or ref["status"] == "invalid":
            details.append(f"| {case.get('room', key)} | {case.get('side', '')} | — | — | — | — | — | Invalid |")
            invalid_views.extend((case["room"], case["side"], row["view"], row["held_count"])
                                 for row in case.get("sampled", []) if row["held_count"] < 8)
            continue
        e, query = case["mae_mm"], case["query_mae_mm"]
        j = case["direction"]["states"]
        counts["surrogate"] += int(ref["surrogate_descent"])
        endpoints.extend(sealed["endpoints"].values())
        match_changes += int(sealed["endpoints"]["zero"]["match_edges"] !=
                             sealed["endpoints"]["plus_full"]["match_edges"])
        minimum_held = min(minimum_held, *(v["held_count"] for v in case["sampled"]))
        repeats.extend([abs(e["repeat_full"] - e["plus_full"]), abs(e["repeat_zero"] - e["zero"])])
        query_change = (query["plus_full"] - query["zero"]
                        if query["plus_full"] is not None and query["zero"] is not None else None)
        details.append(
            f"| {case['room']} | {case['side']} | {fmt(j['plus_full']['J'] - j['zero']['J'], 6)} | "
            f"{fmt(ref['slope_half_mm'])} | {fmt(ref['slope_full_mm'])} | {fmt(ref['actual_change_mm'])} | "
            f"{fmt(query_change)} | {ref['status'].replace('_', ' ')} |")
    statuses = {
        "aligned": "The first step pointed downhill and reduced native held depth error with the registered room agreement.",
        "opposed": "The training loss fell while the native held depth error rose consistently under the registered room rule.",
        "mixed": "The first LoRA step did not show a reproducible directional trend under the registered room rule.",
        "indeterminate": "One or more cases lacked a valid result. The registered screen is indeterminate.",
    }
    actual = scored["actual_steps"]
    coverage = [
        f"- {room}, {side} adaptation, view index {view} retained {count} held pixels after overlap exclusion."
        for room, side, view, count in invalid_views
    ]
    valid_cases = len(cases) - actual["invalid"]
    surrogate_all = sum(c["direction"]["dot"] < 0 and
                        c["direction"]["states"]["plus_full"]["J"] < c["direction"]["states"]["zero"]["J"]
                        for c in cases.values() if "direction" in c)
    lines = [
        "# MASt3R one-step LoRA directional test",
        "",
        statuses[scored["verdict"]],
        f"The full step helped in {actual['helped']} cases, worsened error in {actual['worsened']}, "
        f"fell within the repeat deadband in {actual['within_deadband']}, and was invalid in {actual['invalid']}.",
        "",
        "Each sign is relative to the same initial LoRA factors. The primary metric is held-source axial "
        "depth MAE after fresh native MASt3R alignment and one metric scale fitted to the permitted adaptation samples. "
        "Negative depth changes favor the proposed step. The adaptation query contributed to training and is a diagnostic.",
        "",
        "| Room | Adaptation source | Surrogate loss change | Half-step slope (mm) | Full-step slope (mm) | "
        "Held full-step change (mm) | Query full-step change (mm) | Directional class |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        *details,
        "",
        "The slope is the signed central difference at the stated scale, in mm per unit first Adam step. "
        "The realized change is the full step minus the zero state. The six cases share three rooms.",
        "",
        *coverage,
        "",
        "![Signed step and held depth error](signed_native_depth.svg)",
        "",
        "## Mean held depth error by adaptation direction",
        "",
        "| Adaptation source | Initial MAE (mm) | After full step (mm) | Change (mm) |",
        "|---|---:|---:|---:|",
    ]
    for side, means in scored["equal_rooms_mae_mm"].items():
        zero, full = means["zero"], means["plus_full"]
        if zero is not None and full is not None:
            lines.append(f"| {side} | {zero:.3f} | {full:.3f} | {fmt(full - zero)} |")
        else:
            lines.append(f"| {side} | — | — | — |")
    lines += [
        "",
        "## Verification and scope",
        "",
        f"- Surrogate loss and its gradient both favored the actual step in {surrogate_all} of six cases, "
        f"including {counts['surrogate']} of {valid_cases} cases with sufficient held coverage. "
        "The endpoint uses fresh FP32-tensor inference and a new match graph. "
        f"The correspondence fingerprint changed between the initial and full-step states in {match_changes} cases.",
        f"- The independent replay verified {audit['rows']} published held-view rows and "
        f"{audit['verified_local_view_scores']} held and adaptation-query view errors from local arrays. "
        "It also recomputed the metric scale and checked the held coordinate hashes.",
        f"- Scientific launches used {hours:.2f} summed local hours on the RTX 3080 and USD 0 rental compute.",
    ]
    if endpoints:
        lines.append(f"- Within valid cases, the smallest held view had {minimum_held} samples and the largest "
                     f"absolute repeat difference was {max(repeats):.6f} mm. Valid cases contributed "
                     f"{len(endpoints)} of the {sum(len(v.get('endpoints', {})) for v in seal['cases'].values())} "
                     "completed native endpoint captures.")
    lines += [
        "- The three rooms contribute sparse, viewpoint-shared depth checks at one initialized LoRA step. "
        "This experiment leaves later steps, dense surfaces and independent datasets open.",
        "",
        "[Prospective protocol](../../docs/NATIVE_LORA_DIRECTIONAL_TEST_HANDOFF.md) · "
        "[Execution manifest](../../manifests/native_lora_direction_execution_v1.json) · "
        "[Prediction seal](prediction_seal.json) · [Scores](held_source_scores.json) · "
        "[Independent replay](independent_replay.json)",
        "",
        "## Replay",
        "",
        "The aggregate command needs only the public JSON files and Python. The private flag also evaluates "
        "the sealed local native predictions and depth packets.",
        "",
        "```powershell",
        "python scripts/replay_native_lora_direction.py",
        "python scripts/replay_native_lora_direction.py --private",
        "```",
        "",
    ]
    (HERE / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    report()
