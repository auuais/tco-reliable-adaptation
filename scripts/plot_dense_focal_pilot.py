"""Plot scene-level reconstruction scores and the matched-target execution differences."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from score_dense_focal_pilot import PUBLIC
from score_saved_geometry import read


def plot():
    data = read(PUBLIC / "summary.json")
    audit = read(PUBLIC / "same_target_pairs.json")
    reference = "registered_rgb"
    scores = data["results"]["references"][reference]
    pairs = {p["adaptive_tag"]: p for p in audit["pairs"]}
    scenes = sorted(scores["by_scene"])
    arms = ("frozen", "retain", "consensus", "adaptive")
    names = ("Frozen", "Keep supplied", "Always replace", "Adaptive")
    scene_colors = ("#306e86", "#b65330", "#4f7347")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True,
                                      gridspec_kw={"width_ratios": (1.13, 1)})
    for scene, color in zip(scenes, scene_colors):
        left.plot(range(4), [scores["by_scene"][scene][a]["score"] for a in arms], color=color,
                  marker="o", markersize=4, linewidth=1.35, label=scene.split("/")[0].title())
    left.plot(range(4), [scores["overall"][a]["score"] for a in arms], color="#202a32", marker="D",
              linewidth=2.4, markersize=5.5, label="Mean")
    left.set_xticks(range(4))
    left.set_xticklabels(names)
    left.set_ylabel("Mean normalized 3D error (lower is better)")
    left.set_title("Reconstruction on registered RGB depth", loc="left", fontsize=12, fontweight="bold", pad=14)
    left.legend(loc="best", ncol=2, frameon=True, fontsize=8, framealpha=.88)

    # A deterministic offset makes both optimizer seeds and both zoom directions visible.
    right.axhline(0, color="#38424a", linewidth=.8)
    for case, name, color, marker, offset in (("metadata_fault", "Focal fault", "#306e86", "o", -.13),
                                               ("valid_zoom", "Genuine zoom", "#b65330", "^", .13)):
        points = [r for r in data["rows"] if r["job"]["arm"] == "adaptive" and r["job"]["case"] == case]
        x, y = [], []
        for row in points:
            j = row["job"]
            jitter = (.055 if j["seed"] == 123 else -.055) + j["direction"] * .022
            x.append(scenes.index(j["scene"]) + offset + jitter)
            y.append(pairs[row["tag"]]["adaptive_minus_matched"][reference]["normalized_score"])
        right.scatter(x, y, color=color, marker=marker, label=name, s=33, alpha=.9)
    right.set_xticks(np.arange(len(scenes)))
    right.set_xticklabels([s.split("/")[0].title() for s in scenes])
    right.set_xlim(-.5, len(scenes)-.5)
    right.set_ylabel("Adaptive error minus same-target control error")
    right.set_title("Repeatability with the same focal target", loc="left", fontsize=12, fontweight="bold", pad=14)
    right.legend(loc="best", frameon=True, fontsize=8, framealpha=.88)
    for ax in (left, right):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", color="#d7dfe3", linewidth=.65)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
    fig.savefig(PUBLIC / "dense_focal_comparisons.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot()
