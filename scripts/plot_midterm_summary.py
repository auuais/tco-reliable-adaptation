"""Plot the complete validated five-sequence comparison without hiding regressions."""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=pathlib.Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.comparison / "summary.json").read_text())
    assert summary["status"] == "complete"
    assert summary["planned_pairs"] == summary["completed_pairs"] == 5
    with (args.comparison / "per_sequence.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    sequences = list(dict.fromkeys(row["sequence"] for row in rows))
    assert len(rows) == 10 and len(sequences) == 5
    values = {(row["sequence"], row["method"]): row for row in rows}
    assert len(values) == 10
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    colors = {"VGGT": "#7b8799", "TCO": "#087f8c"}
    labels = {"VGGT": "VGGT: no adaptation", "TCO": "TCO: 40 steps"}
    x = np.arange(len(sequences))
    metrics = [("accuracy_mean_m", "Accuracy error (cm) ↓", 100),
               ("completeness_mean_m", "Completeness error (cm) ↓", 100),
               ("normal_consistency_mean", "Normal consistency ↑", 1)]
    for ax, (key, title, scale) in zip(axes, metrics):
        for method, offset in [("VGGT", -0.18), ("TCO", 0.18)]:
            y = [float(values[(seq, method)][key]) * scale for seq in sequences]
            ax.bar(x + offset, y, width=0.35, color=colors[method], label=labels[method])
        ax.set_title(title, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels([seq.replace("/seq-", " ") for seq in sequences], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        if scale == 1:
            ax.set_ylim(0, 1)
    fig.suptitle("TCO reproduction: all five fixed public test sequences", fontsize=16, y=0.98)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.text(0.5, 0.02, "5 views · 392×518 · seed 42 · reference camera priors · Sim(3) + ICP evaluation\n"
             "Single measurements on a fixed subset; not the published full-benchmark result.",
             ha="center", va="bottom", fontsize=10, color="#4b5563")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.77, bottom=0.25, wspace=0.27)
    output = args.comparison / "comparison.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
