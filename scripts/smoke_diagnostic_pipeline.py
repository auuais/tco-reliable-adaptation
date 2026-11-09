"""Run small real-GPU invariance/gradient checks before renting full-resolution compute."""
import json
import pathlib
import subprocess
import sys
import numpy as np

root=pathlib.Path(__file__).resolve().parents[1]
cases=[('diag_smoke_corrupt224_s0','mixed_severe',0,1),
       ('diag_smoke_corrupt224_s1','mixed_severe',1,1),
       ('diag_smoke_visual224_s1','clean',1,0)]
for name,profile,steps,weight in cases:
    subprocess.run([sys.executable,'scripts/run_diagnostic.py','--run-id',name,'--prior-profile',profile,
                    '--size','224','--frames','0','200','--steps',str(steps),'--prior-weight',str(weight)],cwd=root,check=True)
clean=root/'runs/diag_smoke_clean224_s0'
noisy=root/'runs/diag_smoke_corrupt224_s0'
assert np.array_equal(np.load(clean/'predictions.npy'),np.load(noisy/'predictions.npy'))
r0=json.loads((clean/'run.json').read_text())
for name,_,steps,_ in cases:
    r=json.loads((root/'runs'/name/'run.json').read_text())
    assert r['evaluation_tensor_sha256']==r0['evaluation_tensor_sha256']
    assert r['reference_sha256']==r0['reference_sha256']
    assert not r['base_parameters_modified'] and r['status']=='complete'
    if steps:
        g=json.loads((root/'runs'/name/'gradients.json').read_text())
        assert len(g)==steps and all(x['finite'] and x['gradient_norm_after_upstream_clipping']>0 for x in g)
report={'status':'passed','zero_step_predictions_bitwise_invariant_to_corrupted_priors':True,
        'reference_targets_unchanged':True,'noisy_and_visual_only_backward_passed':True,
        'scope':'two views at 168x224; not full-resolution research results','run_ids':[clean.name]+[x[0] for x in cases]}
(root/'environment/diagnostic_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
