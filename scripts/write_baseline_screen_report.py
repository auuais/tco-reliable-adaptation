"""Generate descriptive tables from the complete validated screen only."""
import argparse
import csv
import json
import pathlib
import statistics

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/baselines_v4"))
args = parser.parse_args()
summary = json.loads((args.results/"summary.json").read_text())
assert summary["status"] == "complete" and summary["jobs"] == 39
rows = list(csv.DictReader((args.results/"per_run.csv").open()))
conditions = ["released", "lr3e4", "lr1e4", "visual"]
names = {"released": "Released LR 0.001", "lr3e4": "LR 0.0003", "lr1e4": "LR 0.0001", "visual": "Visual-only LR 0.0003"}
lines = ["# Learning-rate and visual-only development screen", "",
    "All 39 planned jobs completed: three fresh zero-step controls and 36 adaptations, totaling 1,440 optimizer updates. Each condition uses three optimizer seeds on Heads 01, Chess 03 and Fire 03. These are previously inspected development sequences.", "",
    f"Execution commit: `{summary['code_commit']}`. Frozen manifest SHA-256: `{summary['manifest_sha256']}`.", "",
    "## Aggregate descriptive results", "",
    "Relative error averages accuracy/base-accuracy and completeness/base-completeness, with equal scene weight. A value below 1 improves on the fresh unadapted control. Harm counts any run with accuracy OR completeness >1% worse than that control; this is a descriptive threshold, not a failure probability.", "",
    "| Condition | Relative error | Accuracy cm | Completeness cm | Normal consistency | Adaptation and inference s | Harmful / 9 |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for condition in conditions:
    group = [r for r in rows if r["condition"] == condition]
    assert len(group) == 9
    avg = lambda key: statistics.mean(float(r[key]) for r in group)
    harm = sum(r["harmful_vs_zero"] == "True" for r in group)
    lines.append(f"| {names[condition]} | {avg('relative_error_score'):.4f} | {100*avg('accuracy_mean_m'):.3f} | {100*avg('completeness_mean_m'):.3f} | {avg('normal_consistency_mean'):.4f} | {avg('seconds'):.2f} | {harm}/9 |")
lines.extend(["", "## Per-scene results", "", "Observed seed ranges appear in the figure and machine-readable summary. Individual outcomes remain in per_run.csv.", "",
    "| Scene | Condition | Accuracy cm | Completeness cm | Normal consistency | Relative error | Harmful / 3 |",
    "|---|---|---:|---:|---:|---:|---:|"])
for group in summary["groups"]:
    m = group["metrics"]
    lines.append(f"| {group['scene']} | {names[group['condition']]} | {100*m['accuracy_mean_m']['mean']:.3f} | {100*m['completeness_mean_m']['mean']:.3f} | {m['normal_consistency_mean']['mean']:.4f} | {m['relative_error_score']['mean']:.4f} | {group['harmful_executions']}/3 |")
lines.extend(["", "![Seed-level outcomes](baseline_screen.png)", "", "## Limits", "",
    "This screen does not establish a new method, robustness to estimated/noisy priors, or held-out generalization. Three optimizer seeds coexist with unresolved execution variability. Scores use Sim(3)+ICP alignment and therefore do not establish absolute metric scale. Timings exclude one-time model load and offline reference evaluation; end-to-end deployment comparisons must state that boundary and include any extra matching/probes.", "",
    "All runs have finite predictions/losses/gradients, unchanged tracked base-parameter version counters, matching declared inputs/source/configuration, and a clean precommitted research checkout. Missing or inconsistent evidence causes aggregation to fail rather than produce a partial mean.", ""])
(args.results/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
