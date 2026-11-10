"""Display every prespecified calibration cell and identical-seed execution."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args=parser.parse_args()
    summary=json.loads((args.results/"summary.json").read_text())
    assert summary["all_evidence_validated"] and summary["completed_jobs"]==40
    rows=summary["rows"]
    variants=["released", "axis_nominal", "axis_registration"]
    labels=["Released K", "Axis resize\nnominal focal", "Axis resize\n525 focal"]
    colors=["#2166ac", "#b35806", "#6a51a3"]
    fig,axes=plt.subplots(1,3,figsize=(12,4.7),sharey=True)
    for ax,scene in zip(axes,["heads/seq-01","chess/seq-03","fire/seq-03"]):
        for index,variant in enumerate(variants):
            selected=[r for r in rows if r["scene"]==scene and r["condition"]=="both"
                      and r["calibration"]==variant and r["execution_repeat"]==0]
            assert len(selected)==3
            values=[]
            for j,seed in enumerate([42,123,2026]):
                row=next(r for r in selected if r["seed"]==seed)
                values.append(row["score"])
                ax.scatter(index+(j-1)*.1,row["score"],color=colors[j],s=40,zorder=3,
                           label=f"Seed {seed}" if index==0 else None)
            ax.plot([index,index],[min(values),max(values)],color="#777777",linewidth=1)
            ax.plot([index-.18,index+.18],[np.mean(values)]*2,color="black",linewidth=2)
        ax.axhline(1,color="#888888",linestyle="--",linewidth=1)
        ax.set_xticks(range(3),labels)
        ax.set_title(scene)
        ax.grid(axis="y",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Normalized distance error (lower is better)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,bbox_to_anchor=(.5,.92),frameon=False)
    fig.suptitle("Calibration controls: all 27 prespecified cells",y=.99)
    fig.text(.5,.015,"Same images, pose priors and evaluation reference. 40 updates at LR 0.0003.\n"
             "Black bar: three-seed mean. Spans: observed range, not uncertainty intervals. Calibration uses repeat 0 only.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.88])
    fig.savefig(args.results/"calibration_controls.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,scene in zip(axes,["heads/seq-01","fire/seq-03"]):
        for index,condition in enumerate(["both","visual"]):
            group=next(g for g in summary["repeats"] if g["scene"]==scene and g["condition"]==condition)
            for j,score in enumerate(group["scores"]):
                ax.scatter(index+(j-1)*.1,score,color=colors[j],s=45,label=f"Execution {j}" if index==0 else None,zorder=3)
            ax.plot([index,index],[min(group["scores"]),max(group["scores"])],color="#777777",linewidth=1)
        ax.axhline(1,color="#888888",linestyle="--",linewidth=1)
        ax.set_xticks([0,1],["Both priors","Visual only"])
        ax.set_yscale("log")
        ax.set_title(scene)
        ax.grid(axis="y",which="both",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Normalized distance error (log scale)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=3,bbox_to_anchor=(.5,.92),frameon=False)
    fig.suptitle("Same seed, GPU, inputs and settings: three executions",y=.99)
    fig.text(.5,.015,"Seed 42; released K; 40 updates. Observed variation is not a confidence interval.\n"
             "Each panel has its own vertical scale. Dashed line: fresh zero-step control.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.1,1,.88])
    fig.savefig(args.results/"identical_seed_repeats.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":main()
