"""Profile a single, label-free MASt3R decoder-adapter update at native resolution."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mast3r.model import AsymmetricMASt3R
from capture_mast3r_transfer import WEIGHTS
from prepare_dense_focal_pilot import ROOT,INPUT,write_arrays
from run_depth_scale_study import PLAN
from collect_relative_focal_windows import FRAMES
from evaluation_stability import dump,sha
from score_saved_geometry import read

PUBLIC=ROOT/'results/mast3r_adapter_probe'
RAW=ROOT/'reports/mast3r_adapter_probe'
ATTEMPT='decoder_checkpoint01'
LORA=ROOT/'third_party/TCO/peft/lora.py'


def text_sha(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()


def prepare():
    meta=read(PLAN)['inputs']['heads_base']
    value=dict(attempt=ATTEMPT,checkpoint_sha256=WEIGHTS,input_sha256=sha(INPUT/meta['file']),
        measurements_sha256=sha(ROOT/'results/neighbor_sparse_depth/measurements.npz'),
        source_sha256={n:text_sha(ROOT/n) for n in ('scripts/probe_mast3r_adapter.py','docs/MAST3R_ADAPTER_PROBE.md','third_party/TCO/peft/lora.py')},
        frames=list(FRAMES[:2]),size=[384,512],rank=4,alpha=1,learning_rate=.0001,updates=1,
        precision='bfloat16_autocast_float32_heads',decoder_checkpointing='non_reentrant')
    path=PUBLIC/(ATTEMPT+'_plan.json')
    if path.exists():
        assert read(path)==value
    dump(path,value)
    return value


def state_digest(state):
    h=hashlib.sha256()
    for name,tensor in state:
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def run(weights):
    manifest=prepare()
    result_path=PUBLIC/(ATTEMPT+'.json')
    if result_path.exists() or (PUBLIC/(ATTEMPT+'_failure.json')).exists():
        raise ValueError('Attempt is already recorded; never overwrite or automatically retry it')
    assert sha(weights/'model.safetensors')==WEIGHTS
    try:
        torch.manual_seed(42);np.random.seed(42)
        model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda()
        model.requires_grad_(False)
        original_state=list(model.state_dict().items())
        initial_hash=state_digest(original_state)
        meta=read(PLAN)['inputs']['heads_base']
        with np.load(INPUT/meta['file']) as z:
            rgb=torch.from_numpy(z['images'][:2]).cuda()
        rgb=2*F.interpolate(rgb,size=(384,512),mode='bilinear',align_corners=False,antialias=True)-1
        views=[dict(img=rgb[i:i+1],true_shape=np.array([[384,512]],dtype=np.int32),instance=[str(i)]) for i in range(2)]
        def forward():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                return model(*views)
        with torch.no_grad():
            baseline=[{k:v.detach().cpu() for k,v in branch.items()} for branch in forward()]
        spec=importlib.util.spec_from_file_location('tco_released_lora',LORA)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        selected=[]
        for name,layer in list(model.named_modules()):
            parts=name.split('.')
            if parts[0] not in ('dec_blocks','dec_blocks2') or not isinstance(layer,torch.nn.Linear):
                continue
            if parts[-2] not in ('attn','cross_attn'):
                continue
            parent=model.get_submodule('.'.join(parts[:-1]))
            wrapped=module.LoRALayer(layer,rank=4,alpha=1,dropout=0.)
            setattr(parent,parts[-1],wrapped)
            selected.append(name)
        assert selected
        adapters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        assert adapters and all('lora_' in n for n,p in adapters)
        for block in list(model.dec_blocks)+list(model.dec_blocks2):
            original=block.forward
            def checked(*args,_original=original,**kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(_original,*args,use_reentrant=False,**kwargs)
                return _original(*args,**kwargs)
            block.forward=checked
        with torch.no_grad():
            unchanged=forward()
            exact={f'{i}_{k}':bool(torch.equal(v.detach().cpu(),baseline[i][k])) for i,b in enumerate(unchanged) for k,v in b.items()}
        if not all(exact.values()):
            raise ValueError('Zero-update decoder adapters changed pairwise outputs')
        del unchanged
        with np.load(ROOT/'results/neighbor_sparse_depth/measurements.npz') as z:
            samples=z[f'heads_{FRAMES[0]}']
        xy=np.rint((samples[:,:2]+.5)*[512/518,384/392]-.5).astype(int)
        inside=((xy>=0)&(xy<[512,384])).all(axis=1)
        nearest={}
        for i in np.argsort(samples[:,2]):
            if inside[i]: nearest.setdefault(tuple(xy[i]),i)
        ids=np.array(list(nearest.values()))
        x,y=torch.as_tensor(xy[ids].T,dtype=torch.long,device='cuda')
        measured=torch.as_tensor(samples[ids,2],dtype=torch.float32,device='cuda')
        initial=baseline[0]['pts3d'][0,y.cpu(),x.cpu(),2].cuda()
        good=torch.isfinite(initial)&(initial>0)
        x,y,measured,initial=x[good],y[good],measured[good],initial[good]
        assert len(measured)>=8
        scale=(measured/initial).median().detach();unit=measured.median().detach()
        optimizer=torch.optim.Adam([p for n,p in adapters],lr=.0001)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        start=time.perf_counter()
        out=forward()
        predicted=out[0]['pts3d'][0,y,x,2]
        residual=(scale*predicted-measured)/(.02*unit)
        loss=F.huber_loss(residual,torch.zeros_like(residual),delta=1.)
        assert torch.isfinite(loss)
        loss.backward()
        gradients={n:dict(present=p.grad is not None,finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else True,
                          norm=float(p.grad.norm()) if p.grad is not None else 0.) for n,p in adapters}
        assert all(v['finite'] for v in gradients.values()) and any(v['norm']>0 for v in gradients.values())
        loss_before=float(loss.detach());del out,predicted,residual,loss
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            after=forward()[0]['pts3d'].detach().cpu()
        assert torch.isfinite(after).all()
        change=(after-baseline[0]['pts3d']).abs()
        assert change.max()>0
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated()/2**20
        final_hash=state_digest(original_state)
        assert final_hash==initial_hash
        RAW.mkdir(parents=True,exist_ok=True)
        path=RAW/(ATTEMPT+'.npz')
        write_arrays(path,dict(before=baseline[0]['pts3d'].numpy(),after=after.numpy()))
        result=dict(plan_sha256=sha(PUBLIC/(ATTEMPT+'_plan.json')),
            execution_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            original_state_sha256_before=initial_hash,original_state_sha256_after=final_hash,
            adapter_modules=selected,trainable_parameters=sum(p.numel() for n,p in adapters),zero_update_exact=exact,
            gradients=gradients,sparse_count=len(measured),metric_scale=float(scale),unit_m=float(unit),
            initial_sparse_loss=loss_before,max_point_change=float(change.max()),mean_point_change=float(change.mean()),
            peak_allocated_mib=peak,update_and_postforward_seconds=seconds,device=torch.cuda.get_device_name(),
            point_archive_sha256=sha(path),feasibility_pass=True)
        dump(result_path,result)
        print('Adapter probe passed:',{k:result[k] for k in ('trainable_parameters','peak_allocated_mib','update_and_postforward_seconds','max_point_change')},flush=True)
    except Exception as error:
        dump(PUBLIC/(ATTEMPT+'_failure.json'),dict(error_type=type(error).__name__,message=str(error),plan_sha256=sha(PUBLIC/(ATTEMPT+'_plan.json'))))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--weights',type=Path)
    args=p.parse_args()
    if args.prepare: prepare();print('Registered one decoder-only adapter update.')
    elif args.weights is None: p.error('--weights required')
    else: run(args.weights)
