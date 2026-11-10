"""Publish an interrupted batch without reporting a gate on a selected subset."""
import json
from pathlib import Path
import numpy as np
from evaluation_stability import dump, sha
from repeatability_capture import array_digest

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'pose_precision_v7b_local'
read = lambda p: json.loads(p.read_text())

def main():
    planpath = ROOT / 'manifests' / (STUDY+'.json')
    plan = read(planpath)
    status = read(ROOT/'reports'/STUDY/'batch_status.json')
    assert [x['run_id'] for x in status] == [x['run_id'] for x in plan['jobs'][:len(status)]]
    assert status[-1]['status']=='failed' and 'TimeoutExpired' in status[-1]['error']
    assert all(sha(ROOT/name)==v for name,v in plan['sources'].items())
    rows=[]
    for i,job in enumerate(plan['jobs']):
        folder=ROOT/'runs'/job['run_id']
        if i>=len(status):
            assert not folder.exists()
            rows.append(dict(**job,status='not_started'));continue
        r=read(folder/'run.json')
        assert not r['research_dirty'] and r['manifest_sha256']==sha(planpath)
        assert r['research_commit']=='428988c8bb7eacfa2d5f7cd55eb855a2403655b9'
        assert r['evaluation_tensor_sha256']==array_digest(dict(np.load(ROOT/r['evaluation_inputs_relative'],allow_pickle=False)))
        cap=read(folder/'capture.json')['records']
        for name,record in cap.items():
            path=folder/record['file'];assert sha(path)==record.get('file_sha256',record.get('sha256')),name
            if 'tensor_sha256' in record:assert array_digest(dict(np.load(path,allow_pickle=False)))==record['tensor_sha256']
        losses=read(folder/'losses.json') if (folder/'losses.json').is_file() else []
        grads=read(folder/'gradients.json') if (folder/'gradients.json').is_file() else []
        assert all(g['finite'] for g in grads)
        row=dict(**job,status=status[i]['status'],recorded_losses=len(losses),recorded_gradients=len(grads),evaluation_inputs_sha256=r['evaluation_tensor_sha256'])
        if job['steps']:
            grad=read(folder/'first_clipped_gradients.json')
            assert sha(folder/'first_clipped_gradients.npz')==grad['file_sha256']
            row['first_clipped_gradient_sha256']=grad['tensor_sha256']
        if row['status']=='complete':
            assert len(losses)==len(grads)==job['steps'] and r['status']=='complete' and not r['base_parameters_modified']
            assert read(folder/'metrics.json')==r['metrics']
            p=np.load(folder/'predictions.npy',allow_pickle=False);assert p.shape==(5,392,518,3) and np.isfinite(p).all()
            row.update(prediction_sha256=sha(folder/'predictions.npy'),metrics={k:r['metrics'][k] for k in ['accuracy_mean_m','completeness_mean_m','normal_consistency_mean']},inference_and_adaptation_seconds=r['inference_and_adaptation_seconds'])
        else:
            assert not (folder/'metrics.json').exists() and not (folder/'predictions.npy').exists()
            row['failure']='900-second per-process timeout; no final prediction or outcome'
        rows.append(row)
    public=ROOT/'results'/STUDY;public.mkdir(exist_ok=True)
    dump(public/'interrupted.json',dict(manifest_sha256=sha(planpath),execution_commit=r['research_commit'],rows=rows,
         counts={s:sum(x['status']==s for x in rows) for s in ['complete','failed','not_started']},
         original_gate_evaluable=False,reason='Frozen stop-on-failure policy. No imputation, retry, partial gate or change to frozen scripts.',
         later_allocations_released=False))
if __name__=='__main__':main()
