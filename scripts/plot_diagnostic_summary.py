"""Static figures from the complete diagnostic matrix; no partial-condition averaging."""
import argparse
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--report',type=pathlib.Path,required=True)
args=parser.parse_args()
summary=json.loads((args.report/'summary.json').read_text())
assert summary['status']=='complete' and summary['completed_new_runs']==61
conditions={x['condition']:x for x in summary['conditions']}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
metrics=[('accuracy_mean_m','Accuracy error (cm) ↓',100),
         ('completeness_mean_m','Completeness error (cm) ↓',100),
         ('normal_consistency_mean','Normal consistency ↑',1)]
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
steps=[0,5,10,20,40]
for ax,(metric,label,scale) in zip(axes,metrics):
    xs=[conditions[f'clean_s{s}']['mean_seconds'] for s in steps]
    ys=[conditions[f'clean_s{s}']['mean_metrics'][metric]*scale for s in steps]
    ax.plot(xs,ys,'o-',color='#087f8c')
    for s,x,y in zip(steps,xs,ys):ax.annotate(f'{s} steps',(x,y),xytext=(4,6),textcoords='offset points',fontsize=9)
    ax.set_ylabel(label);ax.set_xlabel('Mean inference + adaptation time (seconds)')
    ax.grid(alpha=.18)
    ax.margins(x=.15,y=.2)
fig.suptitle('Clean-prior quality versus compute: same-session GPU controls',fontsize=15)
fig.text(.5,.02,'Five previously inspected sequences · independent fixed-budget runs · one seed · loading and evaluation excluded',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(args.report/'budget_tradeoff.png',dpi=180);plt.close(fig)

names=['clean_s40','rotation5_s40','translation10_s40','focal_plus10_s40','mixed_moderate_s40','mixed_severe_s40','mixed_severe_weak_s40','visual_only_s40']
labels=['Clean','Rotation 5°','Translation 10%','Focal +10%','Mixed moderate','Mixed severe','Severe, 0.1× priors','Visual only']
x=np.arange(len(names));fig,axes=plt.subplots(2,1,figsize=(13,7),gridspec_kw={'height_ratios':[2.4,1]})
for key,offset,label,color in [('accuracy_improvement_percent',-.18,'Accuracy error reduction','#087f8c'),('completeness_improvement_percent',.18,'Completeness error reduction','#e19733')]:
    axes[0].bar(x+offset,[conditions[n][key] for n in names],width=.35,label=label,color=color)
axes[0].axhline(0,color='#475569',linewidth=1)
axes[0].set_ylabel('Reduction relative to zero-step baseline (%)\nPositive = better; negative = worse')
axes[0].legend(frameon=False,ncol=2);axes[0].grid(axis='y',alpha=.18);axes[0].set_axisbelow(True)
axes[0].set_xticks(x);axes[0].set_xticklabels([])
bars=axes[1].bar(x,[conditions[n]['harmful_scene_count'] for n in names],color='#a33f4e',width=.6)
axes[1].bar_label(bars,fmt='%d / 5',padding=3)
axes[1].set_ylim(0,6);axes[1].set_yticks(range(0,6));axes[1].set_ylabel('Harmful scenes')
axes[1].set_xticks(x);axes[1].set_xticklabels(labels,rotation=15,ha='right')
fig.suptitle('Synthetic prior errors and simple controls: all five pilot sequences',fontsize=15)
fig.text(.5,.015,'Harm = >1% worse accuracy OR completeness than zero-step. One corruption seed; synthetic errors, not measured sensor uncertainty.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.055,1,.96));fig.savefig(args.report/'prior_robustness.png',dpi=180);plt.close(fig)
print(args.report/'budget_tradeoff.png');print(args.report/'prior_robustness.png')
