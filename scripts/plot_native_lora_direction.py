"""Plot held depth error versus signed optimizer step on each development room."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/native_lora_direction_v1"


def plot(path):
    data = json.loads((DATA / "held_source_scores.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, room in zip(axes, ("apt1", "apt2", "office2")):
        earlier, later = (data["cases"][f"{room}_{side}"] for side in ("earlier", "later"))
        ax.set_title(earlier.get("room", later.get("room", room)))
        plotted = False
        for case, color, line, label in ((earlier, "#2166ac", "-", "Earlier adaptation"),
                                         (later, "#b35806", "--", "Later adaptation")):
            errors = case.get("mae_mm", {})
            labels = ("minus_full", "minus_half", "zero", "plus_half", "plus_full")
            if any(errors.get(key) is None for key in labels):
                continue
            plotted = True
            baseline = errors["zero"]
            ax.plot([-1, -.5, 0, .5, 1], [errors[k] - baseline for k in labels],
                    color=color, linestyle=line, marker="o", markersize=4, label=label)
            ax.scatter([0, 1], [errors[k] - baseline for k in ("repeat_zero", "repeat_full")],
                       s=38, facecolor="none", edgecolor=color)
        if not plotted:
            ax.text(.5, .6, "Insufficient held coverage", transform=ax.transAxes,
                    ha="center", color="#4f5c68", fontsize=9)
        ax.axhline(0, color="#7e8997", lw=.8)
        ax.grid(axis="y", color="#e1e6eb", lw=.6)
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].set_ylabel("Held depth MAE change (mm)")
    handles = {}
    for ax in axes:
        h, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, h))
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=2, frameon=False)
    fig.text(.52, .055, "Signed first Adam step (open circles show repeats)", ha="center")
    fig.subplots_adjust(top=.78, bottom=.24, left=.105, right=.98, wspace=.18)
    fig.savefig(path, metadata={"Date": None} if path.suffix == ".svg" else None)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "signed_native_depth.svg")
    plot(parser.parse_args().output)
