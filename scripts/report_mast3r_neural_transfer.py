"""Export complete distance tables and a fixed development-comparison figure."""
import csv
from statistics import mean
import numpy as np
from score_mast3r_neural_transfer import PUBLIC,REFS,read


def report():
    seal=read(PUBLIC/'predictions.json');metrics=read(PUBLIC/'metrics.json');scores=read(PUBLIC/'scores.json')
    baselines={r['input_key']:r['prediction'] for r in seal['rows'] if r['arm']=='zero' and r['post']=='ordinary'}
    rows=[]
    for r in seal['rows']:
        for reference in REFS:
            actual=metrics[r['input_key']][r['prediction']][reference]
            base=metrics[r['input_key']][baselines[r['input_key']]][reference]
            accuracy=actual['accuracy_mean_m']/base['accuracy_mean_m']
            completeness=actual['completeness_mean_m']/base['completeness_mean_m']
            rows.append(dict(scene=r['scene'],grid=r['grid'],neural_arm=r['arm'],alignment=r['post'],
                reference=reference,aligned_accuracy_cm=100*actual['accuracy_mean_m'],
                aligned_completeness_cm=100*actual['completeness_mean_m'],
                relative_accuracy=accuracy,relative_completeness=completeness,
                relative_surface_score=mean([accuracy,completeness]),prediction_sha256=r['prediction']))
    assert len(rows)==210
    with (PUBLIC/'per_reconstruction.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),sharey=True)
    all_values=[]
    for row,grid in enumerate(('centered','shifted')):
        for col,condition in enumerate(('clean','noisy')):
            ax=axes[row,col];cell=scores['grids'][grid][condition]
            ref=cell['references'][REFS[0]]
            methods=[a+'_KDP_'+condition for a in ('zero','D','K')]+[cell['candidate']]
            if cell['best_control'] not in methods:methods.insert(0,cell['best_control'])
            x=np.arange(4);width=.13
            colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']
            for index,method in enumerate(methods):
                values=[ref['scenes'][s+'_base'][method]['score'] for s in ('heads','chess','fire')]+[ref['overall'][method]['score']]
                offset=(index-(len(methods)-1)/2)*width
                label=method.replace('historical_','earlier ').replace('_KDP_'+condition,' + KDP').replace('zero','no adaptation')
                ax.scatter(x+offset,values,s=40,label=label,color=colors[index],zorder=3)
                all_values.extend(values)
            ax.axhline(1,color='.5',linestyle='--',linewidth=1,label='ordinary MASt3R')
            ax.axvline(2.5,color='.8',linewidth=.8)
            ax.set_xticks(x,['Heads','Chess','Fire','Mean'])
            ax.set_title(f'{grid.capitalize()} grid / {condition} pose channel\nCandidate gain over strongest control: {100*cell["local_gain"]:+.2f}%')
            ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7,loc='best')
            if col==0:ax.set_ylabel('Normalized aligned surface error (lower is better)')
    low=min([1.]+all_values);high=max([1.]+all_values);pad=max(.025,.08*(high-low))
    for ax in axes.flat:ax.set_ylim(max(0,low-pad),high+pad)
    fig.suptitle('MASt3R neural adaptation: development comparisons',fontsize=15)
    fig.text(.5,.015,'Registered RGB reference; equal weighting of three development scenes. One execution per case.\nAll 105 reconstructions and both references are retained in per_reconstruction.csv. No generalization or significance claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95))
    fig.savefig(PUBLIC/'comparison.png',dpi=180)
    fig.savefig(PUBLIC/'comparison.svg')
    svg=PUBLIC/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)
    print('Exported all 210 distance rows and the four fixed comparison panels.')


if __name__=='__main__':report()
