"""Plot paired held-source development gains from a verified compensation report."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replay_compensation_published_metrics import PUBLIC, SCENES, SIDES, replay, require

COLORS = {"earlier": "#176879", "later": "#934523"}


def make_figure(report, destination):
    verdict = report["verdict"]
    gains = verdict["r1_directional_relative_gain"]
    baseline = verdict["case_mae_m"]["final"]
    rooms = {}
    for scene in SCENES:
        rooms[scene.title()] = {}
        for side in SIDES:
            control = baseline[f"{scene}_{side}_P1"]
            candidate = baseline[f"{scene}_{side}_R1"]
            valid = all(verdict["validity"]["final"][f"{scene}_{side}_{arm}"]
                        for arm in ("P1", "R1"))
            rooms[scene.title()][side] = (1-candidate/control
                if valid and control is not None and control > 0 and candidate is not None else None)

    with plt.rc_context({"figure.dpi": 160, "font.family": "DejaVu Sans", "svg.hashsalt": "tco-compensation"}):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), constrained_layout=True)
        for ax, table, title in zip(axes, (gains, rooms),
                                   ("Directional mean against each control", "Within each room against P1")):
            ax.axvline(0, color="#49535d", lw=1)
            ax.set_title(title, fontsize=11, loc="left")
            for name in ("top", "right", "left"):
                ax.spines[name].set_visible(False)
            ax.grid(axis="x", alpha=.22)
            ax.set_axisbelow(True)
            labels = list(table)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            ax.set_ylim(len(labels)-.45, -.45)
            for side, offset in (("earlier", -.11), ("later", .11)):
                coords = [(100*table[label][side], idx+offset) for idx, label in enumerate(labels)
                          if table[label][side] is not None]
                if coords:
                    ax.scatter(*zip(*coords), label=side.title()+" adaptation", color=COLORS[side], zorder=3)
                missing = [idx for idx, label in enumerate(labels) if table[label][side] is None]
                for idx in missing:
                    ax.text(.98, idx+offset, side+" unavailable", transform=ax.get_yaxis_transform(),
                            ha="right", va="center", fontsize=7, color=COLORS[side])
            ax.set_xlabel("R1 relative MAE reduction (%)")
            ax.tick_params(axis="y", length=0)
        p1 = list(gains).index("P1")
        lo, hi = axes[0].get_xlim()
        axes[0].set_xlim(min(lo, -.1), max(hi, 1.12))
        axes[0].plot([1, 1], [p1-.22, p1+.22], linestyle="--", lw=1.1,
                     color="#49535d", label="P1 gate (+1%)")
        axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(.5, -.34), ncol=2,
                       frameon=False)
        fig.suptitle("MASt3R calibration compensation on held-source depth", fontsize=13, x=.04, ha="left")
        with destination.open("xb") as output:
            fig.savefig(output, format="svg", metadata={"Date":None})
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=PUBLIC/"held_source_scores.json")
    parser.add_argument("--proof", type=Path, default=PUBLIC/"independent_replay.json")
    args = parser.parse_args()
    data = args.report.read_bytes()
    proof = json.loads(args.proof.read_bytes())
    require(proof["pass_all"] and proof["report_sha256"] == hashlib.sha256(data).hexdigest(),
            "Plot needs a successful independent replay of the same report")
    report = json.loads(data)
    result = replay(report)
    require(result["nomination"] == proof["nomination"], "Verified nomination differs")
    make_figure(report, args.output)


if __name__ == "__main__":
    main()
