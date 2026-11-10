"""Plot every v5 execution; ranges are observed values, not confidence intervals."""
import argparse
import csv
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=pathlib.Path, required=True)
args = parser.parse_args()
with (args.results/'per_run.csv').open(newline='') as source:
    rows = list(csv.DictReader(source))
conditions = ['both', 'pose_only', 'intrinsics_only', 'visual']
labels = ['Both', 'Pose only', 'Intrinsics only', 'Visual only']
colors = ['#2166ac', '#b35806', '#6a51a3']
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
for ax, scene in zip(axes, ['heads/seq-01', 'chess/seq-03', 'fire/seq-03']):
    all_values = []
    for i, condition in enumerate(conditions):
        selected = [r for r in rows if r['scene'] == scene and r['condition'] == condition]
        assert len(selected) == 3
        values = []
        for j, seed in enumerate([42, 123, 2026]):
            value = float(next(r for r in selected if int(r['seed']) == seed)['relative_error_score'])
            assert value > 0
            values.append(value)
            ax.scatter(i+(j-1)*.11, value, color=colors[j], s=38, zorder=3,
                       label=f'Seed {seed}' if i == 0 else None)
        ax.plot([i, i], [min(values), max(values)], color='#555555', linewidth=1.2, zorder=1)
        ax.plot([i-.18, i+.18], [np.mean(values)]*2, color='#111111', linewidth=2.2, zorder=2)
        all_values += values
    ax.axhline(1, color='#777777', linestyle='--', linewidth=1)
    ax.set_yscale('log')
    ax.set_xticks(range(4), labels, rotation=22, ha='right')
    ax.set_title(scene, fontsize=12)
    ax.grid(axis='y', which='both', color='#eeeeee')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Relative distance error (log scale; lower is better)')
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', ncol=3, bbox_to_anchor=(.5,.94), frameon=False)
fig.suptitle('Pose and intrinsics constraints: all 36 adaptation runs', fontsize=14, y=1.0)
fig.text(.5, .01, 'Dashed line: fresh zero-step control. Black bar: mean. Vertical span: observed range.\n'
         'Three development sequences; 40 steps at LR 0.0003; released calibration retained.', ha='center', fontsize=9)
fig.tight_layout(rect=[0,.1,1,.89])
fig.savefig(args.results/'component_ablation.png', dpi=180, bbox_inches='tight')
fig.savefig(args.results/'component_ablation.svg', bbox_inches='tight')
svg = args.results/'component_ablation.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n', encoding='utf-8')
