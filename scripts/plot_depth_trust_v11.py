"""Visualize paired development trajectories without treating execution repeats as seeds."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    out=Path(__file__).resolve().parents[1]/"results/depth_trust_v11"
    d=json.loads((out/"summary.json").read_text());rows=d["rows"]
    order=["tco","half_shape","freeze_shape","bounded_5pct"]
    figure,axes=plt.subplots(1,3,figsize=(10.4,4.5),sharey=True)
    colors=["#0072B2","#D55E00","#009E73"]
    for ax,scene,title in zip(axes,["heads/seq-01","chess/seq-03","fire/seq-03"],["Heads","Chess","Fire"]):
        ax.axhline(1,color="#444444",lw=1.1,ls="--")
        for seed,c in zip([42,123,2026],colors):
            group={r["mode"]:r["score"] for r in rows if r["scene"]==scene and r["seed"]==seed and r["repeat"]==0}
            ax.plot(range(4),[group[m] for m in order],marker="o",lw=1.5,ms=4.2,color=c)
        ax.set_xticks(range(4),["TCO","Half","Frozen","Bounded"],rotation=24,ha="right")
        ax.set_title(title,fontsize=12,weight="bold")
        ax.grid(axis="y",alpha=.16);ax.spines[["top","right"]].set_visible(False);ax.margins(x=.1)
    axes[0].set_ylabel("Distance error / FP32 zero (lower is better)")
    figure.suptitle("Depth-change controls on the same adapted cameras",fontsize=16,weight="bold",x=.07,ha="left")
    handles=[Line2D([0],[0],color=c,marker="o",label="Seed "+str(s)) for s,c in zip([42,123,2026],colors)]
    handles.append(Line2D([0],[0],color="#444444",ls="--",label="Zero-step reference"))
    figure.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.008),ncol=4,frameon=False)
    figure.subplots_adjust(left=.09,right=.985,top=.78,bottom=.25,wspace=.16)
    figure.savefig(out/"paired_depth_controls.png",dpi=170,bbox_inches="tight",pad_inches=.18)
    plt.close(figure)
if __name__=="__main__":main()
