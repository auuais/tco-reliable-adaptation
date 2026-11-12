"""Repeat one frozen development case in a fresh process, preserving all comparison flags."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import torch
from mast3r.model import AsymmetricMASt3R
import mast3r.cloud_opt.sparse_ga as alignment
from run_mast3r_neural_transfer import (PUBLIC as PRIMARY,ROOT,WEIGHTS,PACKETS,FRAMES,ARMS,
    intrinsic,train,read,dump,sha,write_arrays,views_for,training_features,install,digest,
    capture,CalibratedPriors,optimizer_hook)

PUBLIC=ROOT/'results/mast3r_neural_repeat'
RAW=ROOT/'reports/mast3r_neural_repeat'
CASE='heads_centered_J'


def plan():
    source=ROOT/'scripts/repeat_mast3r_neural_case.py'
    value=dict(original_case=CASE,primary_plan_sha256=sha(PRIMARY/'plan.json'),
        original_completion_sha256=sha(PRIMARY/'cases'/CASE/'complete.json'),
        runner_sha256=hashlib.sha256(source.read_text(encoding='utf-8').encode()).hexdigest(),
        seed=42,updates=40,native_rtol=1e-5,native_atol=1e-6,
        interpretation='Numerical repeat diagnostic. Preserve exact and tolerance flags. Nonexact outputs require metric comparison; this is not a population reliability estimate.')
    path=PUBLIC/'plan.json'
    if path.exists():assert read(path)==value
    dump(path,value)
    return value


def run(weights):
    registered=plan()
    if (PUBLIC/'result.json').exists() or (PUBLIC/'failure.json').exists():raise ValueError('Repeat attempt already recorded')
    assert sha(weights/'model.safetensors')==WEIGHTS
    try:
        model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda().requires_grad_(False)
        original=list(model.state_dict().items());base=digest(original)
        torch.manual_seed(42);np.random.seed(42)
        install(model,ROOT/'third_party/TCO/peft/lora.py')
        _,_,views=views_for('heads_base');features=training_features(model,views)
        with np.load(PACKETS['centered']) as z:samples=[z[f'heads_{frame}'] for frame in FRAMES]
        with np.load(PRIMARY/'pose_packets.npz') as z:poses=z['heads_clean']
        trace,support=train(model,features,samples,poses,'J',40)
        assert digest(original)==base
        RAW.mkdir(parents=True,exist_ok=True)
        write_arrays(RAW/'adapters.npz',{n:p.detach().cpu().numpy() for n,p in model.named_parameters() if p.requires_grad})
        write_arrays(RAW/'gauge_support.npz',support)
        dump(PUBLIC/'training.json',trace)
        comparisons={}
        for post in ('ordinary','KD','KDP_clean'):
            if post=='ordinary':capture(model,'heads_base',RAW/'cache',RAW,PUBLIC,post)
            else:
                weight=0. if post=='KD' else .01
                def factory(arguments):return CalibratedPriors(alignment,arguments,samples,poses,intrinsic(),weight)
                with optimizer_hook(alignment,factory) as callbacks:capture(model,'heads_base',RAW/'cache',RAW,PUBLIC,post)
                assert len(callbacks)==1 and len(callbacks[0].trace)==300
                dump(PUBLIC/(post+'_trace.json'),callbacks[0].trace)
            fields={}
            with np.load(RAW/(post+'.npz')) as new,np.load(ROOT/'reports/mast3r_neural_transfer/cases'/CASE/(post+'.npz')) as old:
                for name in new:
                    delta=np.abs(new[name].astype(float)-old[name].astype(float))
                    fields[name]=dict(exact=bool(np.array_equal(new[name],old[name])),max_difference=float(delta.max()),
                        mean_difference=float(delta.mean()),close=bool(np.allclose(new[name],old[name],rtol=registered['native_rtol'],atol=registered['native_atol'])))
            comparisons[post]=fields
        assert digest(original)==base
        original_training=read(PRIMARY/'cases'/CASE/'training.json')
        initial_equal=all(trace[name]==original_training[name] for name in ('gauges','units','sample_counts'))
        done=read(PRIMARY/'cases'/CASE/'complete.json')
        adapter={}
        with np.load(RAW/'adapters.npz') as new,np.load(ROOT/'reports/mast3r_neural_transfer/cases'/CASE/'adapters.npz') as old:
            for name in new:
                adapter[name]=dict(exact=bool(np.array_equal(new[name],old[name])),max_difference=float(np.abs(new[name]-old[name]).max()))
        result=dict(plan_sha256=sha(PUBLIC/'plan.json'),execution_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
            original_state_sha256=base,original_state_matches=base==done['original_state_sha256'],
            initial_gauge_and_support_exact=initial_equal,adapter_comparison=adapter,native_comparison=comparisons,
            all_native_exact=all(v['exact'] for f in comparisons.values() for v in f.values()),
            all_native_close=all(v['close'] for f in comparisons.values() for v in f.values()),
            files={p.name:sha(p) for p in sorted(RAW.glob('*.npz'))})
        dump(PUBLIC/'result.json',result)
        print('Fresh-process repeat:',{name:result[name] for name in ('initial_gauge_and_support_exact','all_native_exact','all_native_close')},flush=True)
    except Exception as error:
        dump(PUBLIC/'failure.json',dict(error_type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--weights',type=Path)
    args=p.parse_args()
    if args.prepare:plan();print('Registered one fresh-process same-seed neural case.')
    elif args.weights is None:p.error('--weights required')
    else:run(args.weights)
