"""CI cross-check of the scientific tables; no weights, raw scenes or cloud required."""
import csv, hashlib, json, math
from pathlib import Path
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
def read(s):return json.loads((ROOT/s).read_text())
def sha(s):return hashlib.sha256((ROOT/s).read_bytes()).hexdigest()
def near(a,b):return math.isclose(a,b,abs_tol=1e-11,rel_tol=1e-11)
def main():
    v6=read('results/repeat_calibration_v6/summary.json')
    assert v6['manifest_sha256']==sha('manifests/repeat_calibration_v6.json')
    assert len(set(r['run_id'] for r in v6['rows']))==v6['completed_jobs']==40 and sum(r['steps'] for r in v6['rows'])==1480
    zero={r['scene']:r for r in v6['rows'] if r['steps']==0};keys=['accuracy_mean_m','completeness_mean_m']
    for k,c in v6['conditions'].items():
        rows=[r for r in v6['rows'] if r['condition']=='both' and r['calibration']==k and r['execution_repeat']==0]
        ratios=[[r[x]/zero[r['scene']][x] for x in keys] for r in rows]
        assert len(rows)==9 and near(mean(map(mean,ratios)),c['score'])
        assert sum(max(a)>1.01 for a in ratios)==c['harmful_cells']
    v9=read('results/pointmap_precision_v9/summary.json')
    assert v9['manifest_sha256']==sha('manifests/pointmap_precision_v9.json') and len(v9['rows'])==12
    zero={r['scene']:r['metrics'] for r in v9['rows'] if r['mode']=='upstream_cpu_float32'}
    stopped=read('results/pose_precision_v7b_local/interrupted.json')
    assert len(stopped['rows'])==29 and stopped['counts']=={k:sum(r['status']==k for r in stopped['rows']) for k in ('complete','failed','not_started')}
    assert not stopped['original_gate_evaluable']
    p='results/depth_trust_v11/';s=read(p+'summary.json');sealed=read(p+'sealed_predictions.json')
    assert s['manifest_sha256']==sealed['manifest_sha256']==sha('manifests/depth_trust_v11.json')
    assert s['sealed_predictions_sha256']==sha(p+'sealed_predictions.json')
    ids={r['job_id']:r['output_sha256'] for r in sealed['records']}
    assert {r['job_id']:r['output_sha256'] for r in s['rows']}==ids and len(s['rows'])==len(ids)==52
    modes={}
    for r in s['rows']:
        ratios=[r['metrics'][x]/zero[r['scene']][x] for x in keys]
        assert near(mean(ratios),r['score']) and (max(ratios)>1.01)==r['harmful']
    for m in ['tco','freeze_shape','half_shape','bounded_5pct']:
        allrows=[r for r in s['rows'] if r['mode']==m];base=[r for r in allrows if r['repeat']==0]
        assert len(base)==9 and len(set((r['scene'],r['seed']) for r in base))==9
        byscene={x:mean(r['score'] for r in base if r['scene']==x) for x in zero}
        spans=[]
        for scene in ('heads/seq-01','fire/seq-03'):
            tr=[r['score'] for r in allrows if r['scene']==scene and r['seed']==42]
            assert len(tr)==3;spans.append(max(tr)-min(tr))
        modes[m]=(mean(r['score'] for r in base),sum(r['harmful'] for r in base),byscene,max(spans))
        assert near(s['modes'][m]['score'],modes[m][0]) and s['modes'][m]['harmful']==modes[m][1]
    modes['zero']=(1.,0,{x:1. for x in zero},0.)
    best=min(['tco','freeze_shape','half_shape','zero'],key=lambda x:modes[x][0])
    assert best==s['best_simple_baseline']
    b=modes[best];c=modes['bounded_5pct'];gain=b[0]-c[0]
    checks=dict(gain_at_least_5pct=gain/b[0]>=.05,harm_not_increased=c[1]<=b[1],
                no_scene_regression=all(c[2][x]<=1.01*b[2][x] for x in zero),
                gain_exceeds_repeat_range=gain>max(c[3],b[3]))
    assert checks==s['gate'] and all(checks.values())==s['passes'] and near(gain/b[0],s['relative_gain'])
    print("Verified public calibration, precision and bounded-depth study tables and development decision.")
if __name__=='__main__':main()
