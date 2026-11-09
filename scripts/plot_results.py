"""Make figures only from completed, input-matched measured runs."""
import argparse
import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=pathlib.Path, required=True)
parser.add_argument("--adapted", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records = [json.loads((path / "run.json").read_text()) for path in (args.baseline, args.adapted)]
assert all(record["status"] == "complete" for record in records)
assert records[0]["arguments"]["steps"] == 0
for name in ("scene", "frames", "size", "seed", "checkpoint"):
    assert records[0]["arguments"][name] == records[1]["arguments"][name], name
assert records[0]["input_sha256"] == records[1]["input_sha256"]
names = ["VGGT (0 steps)", f"TCO ({records[1]['arguments']['steps']} steps)"]
metrics = [record["metrics"] for record in records]
with (args.output / "results.csv").open("w", newline="") as stream:
    fields = ["method", "run_id", "scope", "accuracy_mean_m", "completeness_mean_m",
              "normal_consistency_mean", "inference_and_adaptation_seconds", "peak_allocated_GiB"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, record in zip(names, records):
        writer.writerow({"method": name, "run_id": record["run_id"], "scope": record["scope"],
                         **{key: record["metrics"][key] for key in fields[3:6]},
                         "inference_and_adaptation_seconds": record["inference_and_adaptation_seconds"],
                         "peak_allocated_GiB": record["peak_allocated_bytes"] / 2**30})

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
for ax, key, title, multiplier in zip(axes,
        ["accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean"],
        ["Accuracy error (cm) — lower is better", "Completeness error (cm) — lower is better", "Normal consistency — higher is better"],
        [100, 100, 1]):
    values = [metric[key] * multiplier for metric in metrics]
    bars = ax.bar(names, values, color=["#6d7d91", "#2469aa"], width=.55)
    ax.bar_label(bars, fmt="%.3f", padding=4)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(values) * 1.18)
label = "DEBUG: reduced resolution/view count" if records[0]["scope"] == "debug" else "Single-sequence subset reproduction"
fig.suptitle(f"{records[0]['arguments']['scene']} | {label}", fontsize=14)
fig.savefig(args.output / "metrics.png", dpi=180)
plt.close(fig)

data = np.load(args.baseline / "evaluation_inputs.npz")
images = data["images"].transpose(0, 2, 3, 1)
gt, valid, extr = data["reference_points"], data["valid_mask"], data["extrinsics"]
aligned = []
for path, metric in zip((args.baseline, args.adapted), metrics):
    pred = np.load(path / "predictions.npy")
    pred = metric["similarity_scale"] * np.einsum("nhwj,ij->nhwi", pred, np.array(metric["similarity_rotation"]))
    pred += np.array(metric["similarity_translation"]).T
    transform = np.array(metric["icp_transformation"])
    pred = np.einsum("nhwj,ij->nhwi", pred, transform[:3, :3]) + transform[:3, 3]
    aligned.append(pred)

frames = [0, len(images) - 1] if len(images) > 1 else [0]
fig, axes = plt.subplots(len(frames), 5, figsize=(15, 3.5 * len(frames)), squeeze=False, constrained_layout=True)
for row, index in enumerate(frames):
    depth_gt = (gt[index] @ extr[index, :3, :3].T + extr[index, :3, 3])[..., 2]
    depths = [(points[index] @ extr[index, :3, :3].T + extr[index, :3, 3])[..., 2] for points in aligned]
    low, high = np.quantile(depth_gt[valid[index]], [.01, .99])
    axes[row, 0].imshow(images[index])
    axes[row, 0].set_title(f"Input frame {records[0]['arguments']['frames'][index]}")
    for col, depth, title in zip([1, 2, 3], [depth_gt] + depths, ["Reference depth", names[0], names[1]]):
        shown = np.where(valid[index], depth, np.nan)
        axes[row, col].imshow(shown, cmap="viridis", vmin=low, vmax=high)
        axes[row, col].set_title(title)
    error_base = np.linalg.norm(aligned[0][index] - gt[index], axis=-1)
    error_tco = np.linalg.norm(aligned[1][index] - gt[index], axis=-1)
    delta = np.where(valid[index], (error_base - error_tco) * 100, np.nan)
    bound = max(float(np.nanquantile(np.abs(delta), .95)), 0.1)
    im = axes[row, 4].imshow(delta, cmap="RdBu", vmin=-bound, vmax=bound)
    axes[row, 4].set_title("Pointwise error reduction\n(cm; blue = improvement)", fontsize=11)
    fig.colorbar(im, ax=axes[row, 4], shrink=.75)
    for ax in axes[row]:
        ax.axis("off")
fig.suptitle(label + " | Depths use evaluation alignment; blue = lower pointwise error after adaptation", fontsize=12)
fig.savefig(args.output / "qualitative.png", dpi=180)
plt.close(fig)
print(f"Saved measured comparison to {args.output}")
