"""Static scientific figure: individual seed outcomes and observed ranges."""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results", type=pathlib.Path, default=pathlib.Path("results/baselines_v4"))
args = parser.parse_args()
summary = json.loads((args.results/"summary.json").read_text())
assert summary["status"] == "complete" and summary["jobs"] == 39
rows = list(csv.DictReader((args.results/"per_run.csv").open()))
conditions = ["zero", "released", "lr3e4", "lr1e4", "visual"]
labels = ["No update", "LR 0.001", "LR 0.0003", "LR 0.0001", "Visual only"]
colors = ["#a8b2c1", "#ff8a80", "#5ed6c0", "#84b7ff", "#f5d27b"]
metrics = [("accuracy_mean_m", 100, "Accuracy error (cm)"),
           ("completeness_mean_m", 100, "Completeness error (cm)"),
           ("normal_consistency_mean", 1, "Normal consistency")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#edf3fa",
    "axes.labelcolor": "#edf3fa", "xtick.color": "#bac8d8", "ytick.color": "#bac8d8",
    "axes.facecolor": "#172335", "figure.facecolor": "#101827", "axes.edgecolor": "#526178",
    "savefig.facecolor": "#101827"})
fig, axes = plt.subplots(3, 3, figsize=(14, 9.5))
for row_index, scene in enumerate(["heads/seq-01", "chess/seq-03", "fire/seq-03"]):
    for col, (metric, scale, title) in enumerate(metrics):
        ax = axes[row_index, col]
        for x, condition in enumerate(conditions):
            values = [float(r[metric])*scale for r in rows if r["scene"] == scene and r["condition"] == condition]
            mean = np.mean(values)
            ax.errorbar(x, mean, yerr=[[mean-min(values)], [max(values)-mean]], fmt="o", color=colors[x], capsize=5, markersize=7, zorder=4)
            if len(values) > 1:
                ax.scatter(x+np.linspace(-.1, .1, len(values)), values, color=colors[x], alpha=.8, marker="x", s=30, zorder=5)
        ax.axhline(summary["baselines"][scene][metric]*scale, color=colors[0], linewidth=1, linestyle="--", alpha=.65)
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(scene + " | " + title, loc="left", fontsize=11, pad=12)
        ax.grid(axis="y", alpha=.15)
        ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
fig.suptitle("TCO baseline screen: learning rate and visual-only adaptation", fontsize=19, x=.055, ha="left", y=.98)
fig.text(.055, .939, "Three development sequences | 40 adaptation steps | three optimizer seeds per condition", color="#b4c4d6", fontsize=11)
fig.text(.055, .018, "Dots: means; crosses: seeds; bars: observed ranges (not confidence intervals). Dashed line: fresh zero-step control.\nReference camera priors; Sim(3) + ICP aligned shape. Lower error and higher normal consistency are better.", color="#b4c4d6", fontsize=10)
fig.subplots_adjust(left=.065, right=.98, top=.89, bottom=.105, hspace=.56, wspace=.26)
fig.savefig(args.results/"baseline_screen.png", dpi=170)
plt.close(fig)
