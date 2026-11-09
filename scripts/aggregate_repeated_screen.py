"""Validate all 24 execution repeats, then summarize without inferential claims."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics

import numpy as np
from prior_corruptions import tensor_hash

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ['accuracy_mean_m', 'completeness_mean_m', 'normal_consistency_mean']
CONDITIONS = ['clean10', 'clean40', 'severe40', 'severe_weak40']


def summarize(values):
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', type=pathlib.Path, default=ROOT/'runs')
    p.add_argument('--controls', type=pathlib.Path, default=ROOT/'runs')
    p.add_argument('--output', type=pathlib.Path, default=ROOT/'reports/diagnostics_v3_repeats')
    args = p.parse_args()
    manifest = ROOT/'manifests/diagnostics_v3_repeats.json'
    plan = json.loads(manifest.read_text())
    rows, baselines = [], {}
    for scene in ['heads/seq-01', 'chess/seq-03']:
        b = json.loads((args.controls/f"diag2_{scene.replace('/', '_')}_bridge_clean_s0/run.json").read_text())
        assert b['status'] == 'complete'
        baselines[scene] = b
    for j in plan['jobs']:
        folder = args.runs/j['run_id']
        r = json.loads((folder/'run.json').read_text())
        assert r['status'] == 'complete', j['run_id']
        assert r['study_id'] == plan['study_id']
        b = baselines[j['scene']]
        for key in ['input_sha256', 'reference_sha256', 'upstream_method_sha256', 'versions',
                    'gpu', 'gsplat_backend', 'checkpoint_download', 'image_shape', 'evaluation_tensor_sha256']:
            assert r[key] == b[key], (j['run_id'], key)
        expected = dict(scene=j['scene'], steps=j['steps'], frames=j['frames'], size=j['size'],
            seed=j['seed'], prior_seed=j['prior_seed'], prior_profile=j['profile'], prior_weight=j['weight'])
        assert all(r['arguments'][k] == v for k, v in expected.items())
        cfg = dict(b['configuration'])
        cfg.update(tco_steps=j['steps'], lambda_pose=cfg['lambda_pose']*j['weight'],
                   lambda_intrinsics=cfg['lambda_intrinsics']*j['weight'])
        assert r['configuration'] == cfg
        assert r['runner_sha256'] == plan['source_sha256']['scripts/run_diagnostic.py']
        assert r['prior_generator_sha256'] == plan['source_sha256']['scripts/prior_corruptions.py']
        prior = np.load(folder/'supplied_priors.npz')
        for name, field in [('extrinsics', 'extrinsics_sha256'), ('intrinsics', 'intrinsics_sha256')]:
            assert tensor_hash(prior[name]) == r['prior_metadata'][field]
        assert not r['base_parameters_modified']
        gradients = json.loads((folder/'gradients.json').read_text())
        losses = json.loads((folder/'losses.json').read_text())
        assert len(gradients) == len(losses) == j['steps']
        assert all(x['finite'] and x['gradient_norm_after_upstream_clipping'] > 0 for x in gradients)
        assert all(math.isfinite(v) for x in losses for v in x.values())
        prediction = np.load(folder/'predictions.npy', mmap_mode='r')
        assert np.isfinite(prediction).all() and prediction.shape == (5, 392, 518, 3)
        row = dict(run_id=j['run_id'], scene=j['scene'], condition=j['condition'], repeat=j['repeat'],
            steps=j['steps'], seconds=r['inference_and_adaptation_seconds'],
            allocated_gib=r['peak_allocated_bytes']/2**30,
            final_loss=losses[-1]['total'], **{k:r['metrics'][k] for k in METRICS})
        row['harmful_vs_zero'] = any(row[k] > b['metrics'][k]*1.01 for k in METRICS[:2])
        rows.append(row)
    assert len(rows) == 24
    grouped = []
    for scene, b in baselines.items():
        for condition in CONDITIONS:
            rr = [x for x in rows if x['scene']==scene and x['condition']==condition]
            assert len(rr)==3 and {x['repeat'] for x in rr}=={1,2,3}
            grouped.append(dict(scene=scene, condition=condition, count=3,
                metrics={k:summarize([x[k] for x in rr]) for k in METRICS},
                seconds=summarize([x['seconds'] for x in rr]),
                maximum_allocated_gib=max(x['allocated_gib'] for x in rr),
                harmful_executions=sum(x['harmful_vs_zero'] for x in rr)))
    out = dict(status='complete', study_id=plan['study_id'], completed_jobs=24,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        baselines={k:v['metrics'] for k,v in baselines.items()}, groups=grouped,
        harm_definition='accuracy OR completeness >1% above same-scene zero-step error; descriptive threshold',
        limitations=['two inspected scenes', 'three same-seed execution repeats, not independent optimization seeds',
                     'one synthetic corruption seed and severe profile', 'no confidence intervals or generalization claims',
                     'Sim(3)+ICP aligned shape, not metric scale'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
    with (args.output/'per_run.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(out, indent=2))


if __name__=='__main__': main()
