"""Standalone scientific figure of all runs; bars are observed ranges, not CIs."""
import csv
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/diagnostics_v3_repeats'
summary=json.loads((OUT/'summary.json').read_text())
assert summary['status']=='complete' and summary['completed_jobs']==24
rows=list(csv.DictReader((OUT/'per_run.csv').open()))
conditions=['clean10','clean40','severe40','severe_weak40']
labels=['Clean\n10 steps','Clean\n40 steps','Severe priors\n40 steps','Severe priors\n0.1 weight']
colors=['#68d6ea','#9baaff','#ff927d','#ffd471']
plt.style.use('dark_background')
plt.rcParams.update({'font.size':12,'axes.facecolor':'#162131','figure.facecolor':'#101722','savefig.facecolor':'#101722'})
fig,axs=plt.subplots(2,2,figsize=(13.5,8.5))
for row,scene in enumerate(['heads/seq-01','chess/seq-03']):
    for col,metric in enumerate(['accuracy_mean_m','completeness_mean_m']):
        ax=axs[row,col]
        baseline=summary['baselines'][scene][metric]*100
        ax.axhline(baseline,color='#cad4df',linestyle='--',linewidth=1.5,label='Unadapted VGGT')
        for x,condition in enumerate(conditions):
            vals=np.array([float(r[metric])*100 for r in rows if r['scene']==scene and r['condition']==condition])
            assert len(vals)==3
            ax.plot([x,x],[min(vals),max(vals)],color=colors[x],lw=2)
            ax.scatter(x+np.array([-.06,0,.06]),vals,s=45,color=colors[x],edgecolor='#101722',zorder=4)
            ax.plot([x-.18,x+.18],[vals.mean(),vals.mean()],color=colors[x],lw=3)
        ax.set_xticks(range(4));ax.set_xticklabels(labels)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.15)
        ax.set_ylabel(('Accuracy' if col==0 else 'Completeness')+' error (cm)')
        ax.set_title(scene.replace('/',' / '),loc='left',fontweight='bold')
        if row==0 and col==0:ax.legend(loc='upper left',fontsize=10)
fig.suptitle('TCO under a compute budget and imperfect camera priors',fontsize=20,x=.06,ha='left',y=.98)
fig.text(.06,.923,'Each dot is one fresh-process execution. Same seed and inputs; three repeats per condition. Lower errors are better.',color='#c7d0dc',fontsize=12)
fig.text(.06,.025,'Two exploratory scenes • 5 views at 392 × 518 • Sim(3) + ICP evaluation\nSevere: 10° rotation, 20% trajectory-diameter translation, +20% focal length.\nHorizontal bars: means. Vertical bars: observed ranges, not confidence intervals.',color='#c7d0dc',fontsize=11)
fig.subplots_adjust(left=.085,right=.98,top=.87,bottom=.18,hspace=.5,wspace=.25)
fig.savefig(OUT/'repeated_screen.png',dpi=170)
fig.savefig(OUT/'repeated_screen.pdf')
plt.close(fig)
