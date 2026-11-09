"""Build a human-readable report only from the complete validated screen."""
import csv
import json
import pathlib
import statistics

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/diagnostics_v3_repeats'
s=json.loads((OUT/'summary.json').read_text())
assert s['status']=='complete' and s['completed_jobs']==24
rows=list(csv.DictReader((OUT/'per_run.csv').open()))
names={'clean10':'Reference priors, 10 steps','clean40':'Reference priors, 40 steps',
       'severe40':'Severe priors, 40 steps','severe_weak40':'Severe priors, 0.1 weight, 40 steps'}
def fmt(v,scale=1):
    return f"{v['mean']*scale:.3f} [{v['minimum']*scale:.3f}, {v['maximum']*scale:.3f}]"
time10=statistics.mean(float(r['seconds']) for r in rows if r['condition']=='clean10')
time40=statistics.mean(float(r['seconds']) for r in rows if r['condition']=='clean40')
severe=[r for r in rows if r['condition']=='severe40']
weak=[r for r in rows if r['condition']=='severe_weak40']
def harmed(rr):return sum(r['harmful_vs_zero']=='True' for r in rr)
text=f'''# TCO repeated diagnostic results and next decision

9 November 2026. All 24 planned executions completed and passed source, input,
configuration, supplied-prior hash, finite-gradient and frozen-backbone checks.
The finding is an opportunity for reliable adaptation under uncertain priors,
subject to beating tuned simple baselines and validating on fresh public scenes.

The original 61-job screen was interrupted at its control gate after same-seed
variation appeared. This replacement is a separately frozen exploratory study,
not a continuation with a relaxed tolerance. The [numerical investigation](../NUMERICAL_STABILITY_STATUS.md)
preserves the original-runner repeats and initialization/gradient probes.

## Main observations

- Severe mixed priors caused a descriptive regression in {harmed(severe)} of 6 executions.
  The 0.1-weight variant regressed in {harmed(weak)} of 6 executions. Regression means
  accuracy OR completeness error exceeded the same-scene zero-step value by >1%.
- Ten clean steps averaged {time10:.2f} seconds versus {time40:.2f} seconds for 40,
  a {100*(1-time10/time40):.1f}% reduction in measured inference/adaptation time.
  Quality is reported separately below; speed alone is not a quality improvement.
- These are three same-seed executions per condition on two previously inspected
  scenes. The ranges are execution variability, not confidence intervals or an
  estimate of real sensor failure frequency.

## Exact comparison

Each cell shows mean [minimum, maximum] over three fresh-process executions.
Accuracy and completeness are in centimetres and lower is better. Normal
consistency is higher is better. Harm counts use the convention above.
'''
for scene in ['heads/seq-01','chess/seq-03']:
    b=s['baselines'][scene]
    text+=f"\n### {scene}\n\nUnadapted VGGT: accuracy {b['accuracy_mean_m']*100:.3f} cm, completeness {b['completeness_mean_m']*100:.3f} cm, normal consistency {b['normal_consistency_mean']:.4f}.\n\n"
    text+='| Condition | Accuracy cm | Completeness cm | Normal consistency | Seconds | Harm count |\n|---|---:|---:|---:|---:|---:|\n'
    for condition,name in names.items():
        g=next(x for x in s['groups'] if x['scene']==scene and x['condition']==condition)
        text+=f"| {name} | {fmt(g['metrics']['accuracy_mean_m'],100)} | {fmt(g['metrics']['completeness_mean_m'],100)} | {fmt(g['metrics']['normal_consistency_mean'])} | {fmt(g['seconds'])} | {g['harmful_executions']}/3 |\n"
text+='''
![All 24 measured executions](repeated_screen.png)

## Recommended next work

First test fixed learning rates 1e-3, 3e-4 and 1e-4 and shorter budgets on fresh
development windows. Include visual-only adaptation, each prior group alone,
global weakening and residual-based rejection. Numerical stability and a simpler
step-size choice may account for an apparent benefit of a more complex policy.

Then ask whether independent image correspondences predict which prior-driven
updates improve reconstruction. Compare short trials with and without rotation,
translation or intrinsics constraints from matched adapter/optimizer/RNG states.
Score held-out tracks or pairs with fixed support and a coverage penalty. Ground
truth can label usefulness offline but must never enter the acceptance rule.

Only if that signal beats training loss, residuals and native confidence should
we build a bounded acceptance/rollback policy. Restore Adam moments and counters
as well as adapter weights. Charge all matching, trials and rejected updates to
total latency. Adding IMU or changing the backbone comes after this mechanism
survives stronger baselines and a scene-disjoint public evaluation.

## Protocol and practical limits

The fixed inputs are frames 0/200/400/600/800, five images at 392 × 518, optimizer
seed 42 and independent corruption seed 1729. Severe priors use 10-degree random-axis
rotation perturbations, camera-center offsets equal to 20% of the maximum clean
pairwise camera-center distance, and a shared +20% focal bias. The first camera
pose is unchanged. This is our controlled stress test, not a replication of the
paper supplement's exact noise distribution or calibrated sensor uncertainty.

Reference pose and nominal intrinsics are the clean-prior condition; they are not
perfect independent metrology. Registered reference depth is used only for
evaluation. Upstream Sim(3) then rigid ICP evaluates aligned shape, not absolute
scale or SLAM accuracy. Every run starts a fresh model. Runtime excludes loading,
data preparation and evaluation; future deployment comparisons must also report
cold-start and any new validation overhead.

No inferential statistical or high-end publication claim follows from two scenes.
The result identifies a tractable question and the baselines that a new method
must beat. The original five-sequence reproduction remains separate evidence.

- [Every individual result](per_run.csv)
- [Validated machine-readable summary](summary.json)
- [Full research plan](../NEXT_RESEARCH_PLAN.md)
- [Implementation handoff](../NEXT_IMPLEMENTATION_HANDOFF.md)
'''
(OUT/'RESULTS_AND_NEXT_STEPS.md').write_text(text,encoding='utf-8')
print(OUT/'RESULTS_AND_NEXT_STEPS.md')
