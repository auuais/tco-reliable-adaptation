"""Small real-GPU integration checks; never substitutes for full-resolution results."""
import json
import pathlib
import subprocess
import sys
import numpy as np
from prior_ablation_config import CONDITIONS, validate_loss_components

ROOT=pathlib.Path(__file__).resolve().parents[1]
cases=[('zero_both',0,1.,1.),('zero_visual',0,0.,0.)]+[(c,1,*v) for c,v in CONDITIONS.items()]
records=[]
for c,steps,ps,ks in cases:
    name='prior5_local_smoke_'+c
    subprocess.run([sys.executable,'-u','scripts/run_prior_ablation.py','--run-id',name,
        '--study-id','prior_ablation_v5_local_smoke','--scene','heads/seq-01','--size','224',
        '--frames','0','200','--steps',str(steps),'--learning-rate','.0003',
        '--pose-weight-scale',str(ps),'--intrinsics-weight-scale',str(ks),
        '--dataset-cache',str(ROOT/'environment/validation_dataset_cache.npy')],cwd=ROOT,check=True)
    folder=ROOT/'runs'/name
    r=json.loads((folder/'run.json').read_text())
    assert r['status']=='complete' and not r['base_parameters_modified']
    if records:
        for key in ['input_sha256','reference_sha256','evaluation_tensor_sha256','prior_metadata']:
            assert r[key]==records[0][key]
    if steps:
        losses=json.loads((folder/'losses.json').read_text())
        assert len(losses)==1;validate_loss_components(r['configuration'],losses[0])
    records.append(r)
np.testing.assert_array_equal(np.load(ROOT/'runs/prior5_local_smoke_zero_both/predictions.npy'),
                              np.load(ROOT/'runs/prior5_local_smoke_zero_visual/predictions.npy'))
out=ROOT/'results/prior_ablation_v5_preflight/smoke.json'
assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(status='passed',runs=6,adaptation_updates=4,
    scope='Two 168x224 views on local RTX 3080; integration checks only',
    zero_step_predictions_bitwise_invariant=True,reference_targets_and_supplied_priors_identical=True,
    all_four_component_configurations_backward_passed=True,
    run_ids=[r['run_id'] for r in records],gpu=records[0]['gpu']),indent=2)+'\n',encoding='utf-8')
print(out.read_text())
