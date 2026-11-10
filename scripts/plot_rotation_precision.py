"""Plot analytic controls, explicitly separate from reconstruction accuracy."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/"results/rotation_precision_v6/summary.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for mode,label,color in [("float32","Float32 pose arithmetic","#2166ac"),("cuda_autocast_bfloat16","Released autocast path","#b35806")]:
    rows=[r for r in data["rows"] if r["mode"]==mode and r["second_view_angle_degrees"]<=10]
    angles=[r["second_view_angle_degrees"] for r in rows]
    axes[0].plot(angles,np.rad2deg([r["rotation_loss"] for r in rows]),"o-",markersize=4,color=color,label=label)
    axes[1].plot(angles,[r["theta_gradient"] for r in rows],"o-",markersize=4,color=color,label=label)
axes[0].plot([0,10],[0,5],"--",color="#777777",linewidth=1,label="Unclamped analytic mean")
axes[1].axhline(.5,linestyle="--",color="#777777",linewidth=1)
axes[0].set_ylabel("Returned mean rotation loss (degrees)")
axes[1].set_ylabel("Derivative w.r.t. angle in radians")
for ax in axes:
    ax.set_xlabel("Known rotation of second view (degrees)")
    ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
axes[0].legend(frameon=False,fontsize=9)
axes[0].set_title("Same exact input rotations")
axes[1].set_title("Zero autocast gradients at tested 0.1-3 degrees")
fig.suptitle("Controlled audit of the upstream pose loss",y=.99)
fig.text(.5,.015,"Two views; first rotation is identity. Markers show tested angles; lines only connect observations.\n"
         "The upstream clamp creates a small floor at zero. This is a loss audit, not reconstruction accuracy.",ha="center",fontsize=9)
fig.tight_layout(rect=[0,.1,1,.94])
fig.savefig(ROOT/"results/rotation_precision_v6/rotation_precision.png",dpi=180)
