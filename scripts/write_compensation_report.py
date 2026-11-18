"""Format the scoped research report from verified score and training evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from replay_compensation_published_metrics import ARMS, PUBLIC, ROOT, SCENES, SIDES, replay, require


def read(path):
    return json.loads(path.read_bytes())


def pct(gain):
    return "unavailable" if gain is None else f"{100*gain:+.3f}%"


def mm(mae):
    return "unavailable" if mae is None else f"{1000*mae:.3f}"


def first_screen(initial):
    text = ["| Original candidate | Earlier vs P1 | Later vs P1 | Common improving rooms |",
            "|---|---:|---:|---|"]
    for arm in ("P3", "P4"):
        row = initial["candidates"][arm]
        text.append(f"| {arm} | {pct(row['directional_gain']['earlier'])} | "
                    f"{pct(row['directional_gain']['later'])} | {', '.join(row['same_scenes_improving_both']) or 'none'} |")
    return "\n".join(text)


def direction_table(verdict):
    text = ["| Arm | Earlier calibrated | Later calibrated | Earlier native | Later native |",
            "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        means = verdict["directional_mae_m"]
        values = [mm(means[endpoint][side][arm]) for endpoint in ("final", "native") for side in SIDES]
        text.append("| " + " | ".join([arm]+values) + " |")
    return "\n".join(text)


def contrast_table(verdict):
    text = ["| R1 comparator | Earlier adaptation | Later adaptation |",
            "|---|---:|---:|"]
    for arm in ("P1", "P3", "R0", "R2"):
        row = verdict["r1_directional_relative_gain"][arm]
        text.append(f"| {arm} | {pct(row['earlier'])} | {pct(row['later'])} |")
    return "\n".join(text)


def paired_table(verdict):
    text = ["| Room | Adaptation source | P1 MAE (mm) | R1 MAE (mm) | R1 reduction |",
            "|---|---|---:|---:|---:|"]
    values, valid = verdict["case_mae_m"]["final"], verdict["validity"]["final"]
    for scene in SCENES:
        for side in SIDES:
            c, a = (f"{scene}_{side}_{arm}" for arm in ("P1", "R1"))
            baseline, candidate = values[c], values[a]
            gain = (1-candidate/baseline if valid[a] and valid[c] and baseline is not None
                    and baseline > 0 and candidate is not None else None)
            text.append(f"| {scene.title()} | {side} | {mm(baseline)} | {mm(candidate)} | {pct(gain)} |")
    return "\n".join(text)


def resource_table(audit):
    text = ["| Arm | Verified cases | Median local case time (s) | Peak CUDA allocation (MiB) |",
            "|---|---:|---:|---:|"]
    for arm in ARMS:
        row = audit["by_arm"][arm]
        text.append(f"| {arm} | {row['cases']} | {row['median_local_wall_seconds']:.1f} | {row['peak_allocated_mib']:.1f} |")
    return "\n".join(text)


def compensation_table(audit):
    text = ["| Arm | Fold 0 median compensation norm | Fold 1 median compensation norm |",
            "|---|---:|---:|"]
    for arm in ("P3", "R0", "R1", "R2"):
        cases = [row for row in audit["cases"] if row["arm"] == arm]
        fold_values = [median([row["last_observed_before_update_by_fold"][str(f)]["compensation_l2"]
                              for row in cases]) for f in (0, 1)]
        text.append(f"| {arm} | {fold_values[0]:.6f} | {fold_values[1]:.6f} |")
    return "\n".join(text)


def document(score, initial, audit):
    verdict = score["verdict"]
    nominee = verdict["nomination"]
    if nominee is None:
        recommendation = ("The bounded study ends with a negative development verdict. "
                          "An accuracy improvement or a mechanism that survives dense confirmation remains unestablished. "
                          "The evidence supports a reproducible account of the tested calibration mechanism and its limits. "
                          "An accuracy improvement paper would need a new independently validated result.")
        decision = ("R1 missed the prospective development gate. "
                    "Dense confirmation, conditional mechanism ablations, independent-scene testing and VGGT transfer remain unrun. "
                    "The single local redesign has been used.")
        next_study = ("A separate study could test whether the five-scale surrogate and native alignment "
                      "favor different adapter updates. Measure directional agreement using finite differences "
                      "of native-aligned depth error on fresh development scenes. If they disagree consistently, "
                      "compare a differentiated version of the same native alignment with its forward-identical "
                      "detached control at matched inputs and compute. Confirm any gain on independent surfaces.")
        reasons = ", ".join(verdict["rejection_reasons"])
    else:
        recommendation = ("R1 qualified for dense confirmation on the observed development scenes. "
                          "Publication readiness still requires evaluation of previously unscored scenes and repeatability.")
        decision = ("R1 passed the held-source gate. The next conditional stage is matched dense surface evaluation "
                    "and the three fresh-process repeats. The rental threshold remains separate.")
        next_study = ("Use the registered confirmation to test whether gains carry through native alignment to dense "
                      "surfaces and exceed repeat variation with the detached control. Any generalization claim "
                      "requires a frozen method and previously unscored scenes.")
        reasons = "none"
    common = ", ".join(s.title() for s in verdict["same_scenes_improving_over_p1"]) or "none"
    hours = audit["total_local_wall_seconds"]/3600
    reproduction = audit["historical_control_reproduction"]
    return f"""# Calibration-aware predictive LoRA in MASt3R

{recommendation}

## What the experiments tested

All comparisons freeze the original MASt3R weights. The first screen compares differentiating a fitted calibration solve with a forward-identical detached solver. The redesign compares a query-space calibration compensation penalty with trace-matched ordinary damping and a detached version of the same forward objective. The derivative, ridge convention and compensation penalty are defined in the method specification. [Original design](../../docs/FROZEN_LORA_ACCURACY_PLAN.md) · [Redesign](../../docs/PROFILED_COMPENSATION_REDESIGN.md)

The original 36-state experiment nominated neither P3 nor P4. The no-update diagnostic then measured attenuation of roughly 29–31% on the training path in the two folds of Heads/earlier, while the fitted-query improvement survived the FP32 path. That one adaptation case motivated the redesign. [Original screen](../lora_profiled_v1/held_source_scores.json) · [Calibration diagnosis](../lora_profiled_d1_v1/README.md)

{first_screen(initial)}

## Held-source development results

The redesign uses both adaptation directions for Heads, Chess and Fire. The support and query sets both influence adaptation. The opposite temporal packet is evaluated after the common prediction seal, with a fixed mask excluding adaptation pixel overlap. Each direction aggregates observations within views, averages the five views, then weights the three rooms equally. Both directions use the same three rooms. The primary MAE is measured after native alignment and the common final calibration. The secondary native endpoint is reported in the same metric gauge.

{direction_table(verdict)}

All table entries above are MAE in millimeters. The relative gains below use calibrated MAE with a positive value denoting an improvement over the designated control.

{contrast_table(verdict)}

{paired_table(verdict)}

Common rooms improving against P1 in both directions were {common}. The registered rejection checks report {reasons}. The score report records {score['numerical_failures']} numerical failures.

{decision}

The held-source measurement covers sparse depth on three development rooms. Both directions reuse those rooms. The mask and balanced averaging limit overlap and frame-count bias. Surface accuracy, completeness and cross-scene transfer require separate measurements.

![Directional and paired-room MAE reductions](held_source_gains.svg)

## Mechanism, repeatability and cost

The shared first full-gradient step matched exactly across P3, R0 and R1 in all six conditions. R1 and R2 shared their first forward computation. The output-adjoint replay and parameter finite differences passed the real-model engineering check. Across {reproduction['checked']} fresh P0, P1 and P3 controls, {reproduction['exact']} matched both historical reconstruction endpoints, the frozen backbone fingerprint and each recorded adapter update exactly. The audit records every comparison against the original prediction seal. [Training audit](training_audit.json) · [Engineering bridge](engineering/bridge01/report.json)

The compensation norm below is the training residual change associated with re-fitting scale. The fold 0 value is measured before macro-step index 6 and the fold 1 value before macro-step index 7. Both use the last recorded pre-update loss for that fold. These are adaptation diagnostics and use the allowed packet.

{compensation_table(audit)}

The full versus detached held-source contrast measures performance on this development design. Evidence beyond repeat variation requires fresh-process dense repeatability confirmation. The comparison across official TCO, CAPA, Test3R and LoRA3D is methodological here since they use different objectives or input information. [Primary-source comparison](../../docs/PROFILED_COMPENSATION_RELATED_WORK.md)

{resource_table(audit)}

The 36 frozen or trained redesign states used {hours:.2f} summed local container hours and {audit['total_optimizer_updates']} optimizer updates. Peak CUDA allocation was {audit['peak_allocated_mib']:.1f} MiB and peak reservation was {audit['peak_reserved_mib']:.1f} MiB. Per-case wall times include model setup and launcher overhead. Rental compute cost attributable to the study is USD {audit['rental_compute_usd']:.2f}. [Budget ledger](../../docs/BUDGET.md)

## Next research question

{next_study}

## Reproduction

The execution manifest and evaluator were committed before scientific optimization. The prediction seal binds the endpoints before held-source score access. Local source-backed replay and an independent sample replay match the original decision. The public aggregate replay checks the entire 180-view table and gate arithmetic without needing sampled measurements, full predictions or weights. [Execution](../../manifests/lora_profiled_compensation_execution_v1.json) · [Evaluation](../../manifests/lora_profiled_compensation_evaluation_v1.json) · [Global seal](prediction_seal.json) · [Scores](held_source_scores.json) · [Source replay](source_replay.json) · [Independent replay](independent_replay.json) · [C/E backup](verdict_backup01.json)

```powershell
python scripts/replay_compensation_published_metrics.py
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PUBLIC/"research_verdict.md")
    args = parser.parse_args()
    require(args.output.resolve().parent == PUBLIC.resolve(), "Place the report beside release evidence")
    raw = (PUBLIC/"held_source_scores.json").read_bytes()
    score = json.loads(raw)
    proof = read(PUBLIC/"independent_replay.json")
    replayed = replay(score)
    source = read(PUBLIC/"source_replay.json")
    audit = read(PUBLIC/"training_audit.json")
    seal = read(PUBLIC/"prediction_seal.json")
    require(proof["pass_all"] and source["pass_all"] and proof["nomination"] == replayed["nomination"]
            and proof["report_sha256"] == source["report"]["sha256"] == hashlib.sha256(raw).hexdigest()
            and proof["samples_sha256"] == source["sample_replay"]["sha256"] == score["sample_replay"]["sha256"],
            "Independent or source replay does not match the current verdict")
    require(audit["pass_all"] and audit["complete_count"] == seal["completed_states"] == 36
            and audit["total_optimizer_updates"] == 240 and seal["pass_all"],
            "Expected all 36 audited redesign states and 240 updates")
    history = audit["historical_control_reproduction"]
    require(history["checked"] == history["expected"] == 18, "Expected all 18 historical control comparisons")
    require({row["run_id"]: row["report"]["sha256"] for row in audit["cases"]}
            == {row["run_id"]: row["selected"]["report_sha256"] for row in seal["cases"]},
            "Training audit and sealed prediction reports differ")
    conditions = audit["conditions"]
    require(set(conditions) == {f"{scene}_{side}" for scene in SCENES for side in SIDES}
            and all(row["p3_r0_r1_first_full_step_exact"] and row["r1_r2_first_forward_exact"]
                    for row in conditions.values()), "Training comparison checks differ")
    initial = read(ROOT/"results/lora_profiled_v1/held_source_scores.json")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(document(score, initial, audit))


if __name__ == "__main__":
    main()
