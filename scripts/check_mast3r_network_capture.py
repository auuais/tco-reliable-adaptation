"""Check zero-adapter end-to-end float32 fidelity before neural-transfer experiments."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R
from capture_mast3r_transfer import WEIGHTS
from mast3r_network_capture import capture,views_for,training_features,training_pair
from mast3r_network_priors import install,digest
from prepare_dense_focal_pilot import ROOT
from evaluation_stability import dump,sha


def run(weights):
    public=ROOT/'results/mast3r_neural_transfer'
    raw=ROOT/'reports/mast3r_neural_transfer'
    assert sha(weights/'model.safetensors')==WEIGHTS
    model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda().requires_grad_(False)
    base=list(model.state_dict().items());initial=digest(base)
    _,_,views=views_for('heads_base')
    features=training_features(model,views)
    with torch.no_grad(): before=training_pair(model,features,0,1)
    torch.manual_seed(42)
    modules=install(model,ROOT/'third_party/TCO/peft/lora.py')
    with torch.no_grad(): after=training_pair(model,features,0,1)
    fidelity={f'{i}_{name}':bool(torch.equal(value,after[i][name])) for i,b in enumerate(before) for name,value in b.items()}
    del features,before,after
    meta=capture(model,'heads_base',raw/'bridge_cache',raw/'bridge',public/'bridge','heads_zero')
    difference={}
    with np.load(raw/'bridge/heads_zero.npz') as new,np.load(ROOT/'results/mast3r_transfer/native_outputs/heads_base.npz') as old:
        for name in new:
            delta=np.abs(new[name].astype(float)-old[name].astype(float))
            difference[name]=dict(exact=bool(np.array_equal(new[name],old[name])),max_difference=float(delta.max()))
    final=digest(base)
    passed=all(fidelity.values()) and all(v['exact'] for v in difference.values()) and initial==final
    dump(public/'zero_adapter_bridge.json',dict(checkpoint_sha256=WEIGHTS,training_pair_exact=fidelity,
        final_inference_difference=difference,original_state_hash=initial,final_state_hash=final,
        adapter_module_count=len(modules),capture=meta,pass_all=passed))
    print('Zero-adapter fresh-matching bridge:',passed,flush=True)
    if not passed: raise ValueError('Zero-adapter bridge failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True)
    run(p.parse_args().weights)
