"""Validate the frozen diagnostic matrix, then report every condition and regression."""
import argparse
import csv
import json
import math
import pathlib
import statistics

ROOT=pathlib.Path(__file__).resolve().parents[1]
METRICS=['accuracy_mean_m','completeness_mean_m','normal_consistency_mean']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runs',type=pathlib.Path,default=ROOT/'runs')
    parser.add_argument('--baseline-runs',type=pathlib.Path,default=ROOT/'runs')
    parser.add_argument('--output',type=pathlib.Path,default=ROOT/'reports/diagnostics_v2')
    args=parser.parse_args()
    plan=json.loads((ROOT/'manifests/diagnostics_v2.json').read_text())
    rows=[]; missing=[]; baselines={}; adaptations={}; generators=set(); runtime_sources=set()
    for scene in dict.fromkeys(j['scene'] for j in plan['jobs']):
        for steps,lookup in [(0,baselines),(40,adaptations)]:
            r=json.loads((args.runs/f"diag2_{scene.replace('/','_')}_bridge_clean_s{steps}/run.json").read_text())
            assert r['status']=='complete' and not r['base_parameters_modified']
            lookup[scene]=r
            rows.append(dict(scene=scene,condition=f'clean_s{steps}',run_id=r['run_id'],steps=steps,
                             profile='clean',weight=1,seconds=r['inference_and_adaptation_seconds'],
                             allocated_gib=r['peak_allocated_bytes']/2**30,**{k:r['metrics'][k] for k in METRICS}))
    for job in plan['jobs']:
        path=args.runs/job['run_id']/'run.json'
        if not path.exists():missing.append(job['run_id']);continue
        r=json.loads(path.read_text())
        if r['status']!='complete':missing.append(job['run_id']);continue
        b=baselines[job['scene']]
        expected={'scene':job['scene'],'frames':job['frames'],'steps':job['steps'],'size':job['size'],
                  'seed':job['seed'],'prior_seed':job['prior_seed'],'prior_profile':job['profile'],'prior_weight':job['weight']}
        assert all(r['arguments'][k]==v for k,v in expected.items()),f"Arguments mismatch: {job['run_id']}"
        assert r['study_id']==plan['study_id'] and not r['base_parameters_modified']
        for key in ['input_sha256','reference_sha256','upstream_method_sha256','versions','gpu','gsplat_backend','checkpoint_download','image_shape']:
            assert r[key]==b[key],f"Baseline mismatch: {job['run_id']} {key}"
        cfg=dict(b['configuration']);cfg['tco_steps']=job['steps'];cfg['lambda_pose']*=job['weight'];cfg['lambda_intrinsics']*=job['weight']
        assert r['configuration']==cfg,'Unexpected scientific change'
        assert r['prior_metadata']['settings']==plan['profiles'][job['profile']]
        assert all(math.isfinite(r['metrics'][k]) for k in METRICS)
        generators.add(r['prior_generator_sha256']);runtime_sources.add(r['runner_sha256'])
        if job['steps']:
            gradients=json.loads((path.parent/'gradients.json').read_text())
            losses=json.loads((path.parent/'losses.json').read_text())
            assert len(gradients)==len(losses)==job['steps']
            assert all(g['finite'] and g['gradient_norm_after_upstream_clipping']>0 for g in gradients)
        if 'bridge_' in job['run_id']: continue
        condition=job['run_id'].split(job['scene'].replace('/','_')+'_',1)[1]
        rows.append(dict(scene=job['scene'],condition=condition,run_id=r['run_id'],steps=job['steps'],
                         profile=job['profile'],weight=job['weight'],seconds=r['inference_and_adaptation_seconds'],
                         allocated_gib=r['peak_allocated_bytes']/2**30,**{k:r['metrics'][k] for k in METRICS}))
    args.output.mkdir(parents=True,exist_ok=True)
    if missing:
        (args.output/'summary.json').write_text(json.dumps({'status':'incomplete','missing':missing},indent=2))
        raise RuntimeError(f'{len(missing)} planned jobs missing or failed; no partial means')
    assert len(generators)==len(runtime_sources)==1
    grouped={}
    for row in rows:grouped.setdefault(row['condition'],[]).append(row)
    results=[]
    base_means={k:statistics.mean(b['metrics'][k] for b in baselines.values()) for k in METRICS}
    for condition,items in grouped.items():
        assert len(items)==5 and len({x['scene'] for x in items})==5
        means={k:statistics.mean(x[k] for x in items) for k in METRICS}
        harm=[x['scene'] for x in items if any(x[k]>baselines[x['scene']]['metrics'][k]*1.01 for k in METRICS[:2])]
        result={'condition':condition,'mean_metrics':means,'mean_seconds':statistics.mean(x['seconds'] for x in items),
                'max_allocated_gib':max(x['allocated_gib'] for x in items),'harmful_scene_count':len(harm),'harmful_scenes':harm,
                'accuracy_improvement_percent':100*(1-means[METRICS[0]]/base_means[METRICS[0]]),
                'completeness_improvement_percent':100*(1-means[METRICS[1]]/base_means[METRICS[1]])}
        results.append(result)
    summary={'status':'complete','study_id':plan['study_id'],'planned_new_runs':len(plan['jobs']),
             'completed_new_runs':len(plan['jobs']),'conditions':results,
             'aggregation':'equal weight per scene; all five scenes in every condition',
             'harm_definition':'accuracy OR completeness error exceeds same-scene zero-step error by more than 1%',
             'limitations':['previously inspected exploratory subset','one optimization and corruption seed',
                            'synthetic perturbations, not measured sensor errors','reference depth is evaluation-only',
                            'Sim(3) plus ICP removes global gauge errors; no metric-scale claim']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'per_run.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
