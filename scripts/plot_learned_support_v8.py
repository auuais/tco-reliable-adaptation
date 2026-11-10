"""Descriptive development associations; no uncertainty or causal claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT=Path(__file__).resolve().parents[1]
result=ROOT/"results/learned_support_v8"
data=json.loads((result/"fixed_match_proxy_validity.json").read_text())
zero={r["scene"]:r["proxy"]["score"] for r in data["rows"] if r["condition"]=="zero"}
fig,axes=plt.subplots(1,3,figsize=(13,4.8))
colors=dict(released="#2266aa",axis_nominal="#bf6800",axis_registration="#7452a4",visual="#b52a40")
labels=dict(released="Both / released K",axis_nominal="Both / axis nominal",axis_registration="Both / axis 525",visual="Visual only")
for ax,summary in zip(axes,data["scenes"]):
    scene=summary["scene"]
    for key,color in colors.items():
        rows=[r for r in data["rows"] if r["scene"]==scene and r["condition"]!="zero" and ("visual" if r["condition"]=="visual" else r["calibration"])==key]
        ax.scatter([r["proxy"]["score"]/zero[scene] for r in rows],[r["reference_score"] for r in rows],c=color,s=38,label=labels[key])
    accepted=[r for r in data["rows"] if r["scene"]==scene and r.get("would_accept")]
    ax.scatter([r["proxy"]["score"]/zero[scene] for r in accepted],[r["reference_score"] for r in accepted],facecolors="none",edgecolors="black",s=130,linewidths=1.5)
    ax.axvline(1,color="gray",ls="--",lw=1);ax.axhline(1,color="gray",ls="--",lw=1)
    ax.set_xscale("log");ax.set_yscale("log");ax.grid(alpha=.15)
    xticks={"heads/seq-01":[1,2,3],"chess/seq-03":[.9,1,1.2,1.4,1.6],"fire/seq-03":[1,2,5,10]}[scene]
    yticks={"heads/seq-01":[1,2,5,10],"chess/seq-03":[.7,.8,.9,1],"fire/seq-03":[1,1.5,2]}[scene]
    ax.set_xticks(xticks,[f"{x:g}" for x in xticks]);ax.set_yticks(yticks,[f"{x:g}" for x in yticks])
    ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
    ax.set_title(f"{scene}\nRank correlation {summary['rank_correlation_with_reference_error']:.2f}")
    ax.set_xlabel("Match inconsistency / fresh control")
axes[0].set_ylabel("Normalized reconstruction error")
handles,legend=axes[0].get_legend_handles_labels();fig.legend(handles,legend,ncol=4,loc="upper center",bbox_to_anchor=(.5,.99),frameon=False)
fig.suptitle("Learned matching: coverage improves, acceptance remains sparse",y=1.06,fontsize=15)
fig.text(.5,.01,"All 37 adapted development runs. Black outline: accepted by the unchanged two-fold rule (1/37).\nLower is better on both axes. Panels use separate log scales; observations are dependent, not held-out evidence.",ha="center",fontsize=10)
fig.subplots_adjust(top=.76,bottom=.23,wspace=.3)
fig.savefig(result/"proxy_association.png",dpi=170,bbox_inches="tight")
