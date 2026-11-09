"""Offline comparison of captured initialization, RNG and clipped gradients."""
import hashlib
import json
import pathlib
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'cloud/downloads/20261109-diagnostics/numerical_evidence'
OUT=ROOT/'reports/numerical_stability'
OUT.mkdir(exist_ok=True)
comparisons=[]
for prefix in ['numprobe_heads','numprobe_disable_mv','numprobe_strict_cublas']:
    paths=sorted((EVIDENCE/'runs').glob(prefix+'*/numerical_probe.json'))
    assert len(paths)==2
    a,b=[json.loads(p.read_text()) for p in paths]
    assert len(a['steps'])==len(b['steps'])==2
    aa,bb=[np.load(p.parent/'first_gradients.npz') for p in paths]
    diff2=norma2=normb2=dot=0.;maximum=0.
    for key in aa.files:
        x,y=aa[key].astype(np.float64),bb[key].astype(np.float64)
        d=x-y
        diff2+=np.sum(d*d);norma2+=np.sum(x*x);normb2+=np.sum(y*y);dot+=np.sum(x*y)
        maximum=max(maximum,float(np.max(np.abs(d))))
    loss_a,loss_b=[json.loads((p.parent/'losses.json').read_text()) for p in paths]
    pred_a,pred_b=[np.load(p.parent/'predictions.npy') for p in paths]
    comparisons.append(dict(probe=prefix, run_ids=[p.parent.name for p in paths],
        same_initial_parameters=a['parameters']==b['parameters'],
        same_setup_rng=a['rng_before_setup']==b['rng_before_setup'] and a['rng_after_setup']==b['rng_after_setup'],
        same_step_rng=[x['rng']==y['rng'] for x,y in zip(a['steps'],b['steps'])],
        same_step_gradient_hashes=[x['gradient_sha256']==y['gradient_sha256'] for x,y in zip(a['steps'],b['steps'])],
        same_first_losses=loss_a[0]==loss_b[0],
        first_gradient_relative_l2=float(np.sqrt(diff2/norma2)),
        first_gradient_cosine=float(dot/np.sqrt(norma2*normb2)),
        first_gradient_maximum_absolute_difference=maximum,
        identical_final_pointmaps=bool(np.array_equal(pred_a,pred_b)),
        maximum_pointmap_difference=float(np.max(np.abs(pred_a-pred_b))),
        probe_source_sha256=a['probe_source_sha256']))
repeats=[]
for p in sorted((EVIDENCE/'runs').glob('reprocheck*/run.json')):
    r=json.loads(p.read_text());assert r['status']=='complete'
    repeats.append({k:r[k] for k in ['run_id','metrics','inference_and_adaptation_seconds','runner_sha256']})
out=dict(status='complete', comparisons=comparisons, original_runner_repeats=repeats,
    conclusion='Same RNG/initialization and first losses do not produce identical gradients or two-step predictions in the tested modes.',
    unresolved='Exact responsible operator(s) and numerical amplification path are not localized; disabling rendering alone does not remove variation.',
    evidence_archive_sha256='c4ce07f2f3b2fd289818f07505d2b28dfbc8cba1c75de13163fe994f63a06fa4')
(OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
