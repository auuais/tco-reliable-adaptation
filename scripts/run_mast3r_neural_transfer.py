"""Run the fixed local neural-adaptation and calibrated-alignment development screen."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mast3r.model import AsymmetricMASt3R
import mast3r.cloud_opt.sparse_ga as alignment
from capture_mast3r_transfer import WEIGHTS
from mast3r_network_priors import install,digest,native_samples,objective
from mast3r_network_capture import capture,views_for,training_features,training_pair
from mast3r_calibrated_alignment import CalibratedPriors
from mast3r_alignment_hook import optimizer_hook
from prepare_dense_focal_pilot import ROOT,INPUT,write_arrays
from run_depth_scale_study import PLAN
from collect_relative_focal_windows import FRAMES
from calibration_audit import resized_k
from evaluation_stability import dump,sha
from score_saved_geometry import read

PUBLIC=ROOT/'results/mast3r_neural_transfer'
RAW=ROOT/'reports/mast3r_neural_transfer'
PACKETS={'centered':ROOT/'results/neighbor_sparse_depth/measurements.npz','shifted':ROOT/'results/shifted_sparse_grid/measurements.npz'}
POSES=ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz'
SCENES=('heads','chess','fire')
ARMS={'D':(0.,0.),'K':(.01,0.),'J':(.01,.01),'N':(.01,.01)}
SOURCES=('scripts/run_mast3r_neural_transfer.py','scripts/mast3r_network_priors.py','scripts/mast3r_network_capture.py',
         'scripts/mast3r_calibrated_alignment.py','scripts/mast3r_joint_prior_loss.py','scripts/mast3r_alignment_hook.py',
         'third_party/TCO/peft/lora.py','docs/MAST3R_NEURAL_TRANSFER.md')


def text_sha(path): return hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()


def intrinsic():
    k=resized_k(525);k[[0,1],[0,1]]*=1.12
    resize=np.diag([512/518,384/392,1.]);resize[:2,2]=(resize[[0,1],[0,1]]-1)/2
    return resize@k


def schedule(steps=40):
    values=[]
    for step in range(steps):
        i=(step%20)//4;j=(i+step%4+1)%5;k=(j+1)%5
        if k==i:k=(k+1)%5
        values.append([i,j,k])
    return values


def pose_packets():
    values={}
    with np.load(POSES) as original:
        for index,slug in enumerate(SCENES):
            clean=original[slug+'_base'].astype(float)
            c2w=np.linalg.inv(clean);noisy=c2w.copy()
            diameter=np.linalg.norm(c2w[:,None,:3,3]-c2w[None,:,:3,3],axis=-1).max()
            rng=np.random.default_rng(20261112+index)
            for i in range(1,5):
                axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
                direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
                noisy[i,:3,:3]=c2w[i,:3,:3]@Rotation.from_rotvec(axis*np.deg2rad(5)).as_matrix()
                noisy[i,:3,3]+=.02*diameter*direction
            values[slug+'_clean']=clean;values[slug+'_noisy']=np.linalg.inv(noisy)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    write_arrays(PUBLIC/'pose_packets.npz',values)


def prepare():
    pose_packets()
    jobs=[]
    for slug in SCENES:
        jobs.append(dict(scene=slug,grid='none',arm='zero',case=slug+'_zero_none',post=['ordinary']))
        for grid in PACKETS:
            jobs.append(dict(scene=slug,grid=grid,arm='zero',case=f'{slug}_zero_{grid}',post=['KD','KDP_clean','KDP_noisy']))
            for arm in ARMS:
                post=['ordinary','KD']+(['KDP_clean','KDP_noisy'] if arm in ('D','K') else ['KDP_noisy' if arm=='N' else 'KDP_clean'])
                jobs.append(dict(scene=slug,grid=grid,arm=arm,case=f'{slug}_{grid}_{arm}',post=post))
    rgb=read(PLAN)['inputs']
    manifest=dict(checkpoint_sha256=WEIGHTS,sources={n:text_sha(ROOT/n) for n in SOURCES},
        rgb_sha256={s:sha(INPUT/rgb[s+'_base']['file']) for s in SCENES},
        packets={g:sha(p) for g,p in PACKETS.items()},original_poses_sha256=sha(POSES),
        pose_packets_sha256=sha(PUBLIC/'pose_packets.npz'),intrinsic=intrinsic().tolist(),schedule=schedule(),
        arms=ARMS,jobs=jobs,reconstructions=sum(len(j['post']) for j in jobs),seed=42,learning_rate=.0001)
    path=PUBLIC/'plan.json'
    if path.exists():
        import json
        assert read(path)==json.loads(__import__('json').dumps(manifest)),'Frozen neural plan changed'
    dump(path,manifest)
    assert manifest['reconstructions']==105 and sum(j['arm']!='zero' for j in jobs)==24
    return manifest


def make_gauges(model,features,samples):
    support=[native_samples(s,'cuda') for s in samples]
    z={}
    with torch.no_grad():
        for i in range(5):
            x,y=support[i][0]
            for j in range(5):
                if i!=j:z[i,j]=training_pair(model,features,i,j)[0]['pts3d'][0,y,x,2].detach()
    units=[];scales={};evidence={}
    for i in range(5):
        xy,measured=support[i]
        good=torch.stack([torch.isfinite(z[i,j])&(z[i,j]>0) for j in range(5) if i!=j]).all(dim=0)
        xy,measured=xy[:,good],measured[good]
        assert len(measured)>=8
        support[i]=(xy,measured);units.append(measured.median().detach())
        evidence[f'xy_{i}']=xy.cpu().numpy();evidence[f'measured_{i}']=measured.cpu().numpy()
        for j in range(5):
            if i==j:continue
            initial=z[i,j][good]
            scales[i,j]=(measured/initial).median().detach()
            evidence[f'initial_{i}_{j}']=initial.cpu().numpy()
    return support,units,scales,evidence


def train(model,features,samples,poses,arm,steps):
    support,units,scales,evidence=make_gauges(model,features,samples)
    c2w=np.linalg.inv(poses)
    relative={(i,j):torch.as_tensor(poses[j]@c2w[i],dtype=torch.float32,device='cuda') for i in range(5) for j in range(5) if i!=j}
    y,x=np.mgrid[8:384:16,8:512:16]
    rays=torch.as_tensor(np.stack([x,y,np.ones_like(x)],axis=-1)@np.linalg.inv(intrinsic()).T,dtype=torch.float32,device='cuda')
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=.0001)
    iw,pw=ARMS[arm]
    trace=[]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    for step,(i,j,k) in enumerate(schedule(steps)):
        optimizer.zero_grad(set_to_none=True)
        out=[training_pair(model,features,i,j),training_pair(model,features,i,k)]
        loss,components=objective(out,[scales[i,j],scales[i,k]],*support[i],units[i],
                                  [relative[i,j],relative[i,k]],rays,pw,iw)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite neural objective')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if norm<=0:raise ValueError('Zero neural gradient')
        trace.append(dict(step=step,pairs=[[i,j],[i,k]],loss=float(loss.detach()),gradient_norm=float(norm),
                          **{name:float(value.detach()) for name,value in components.items()}))
        optimizer.step()
        del out,loss,components
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    timing=dict(training_seconds=time.perf_counter()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    gauges={f'{i}_{j}':float(value) for (i,j),value in scales.items()}
    return dict(rows=trace,gauges=gauges,units=[float(u) for u in units],sample_counts=[len(m) for xy,m in support],**timing),evidence


def run(weights,smoke=False):
    manifest=prepare()
    assert read(PUBLIC/'zero_adapter_bridge.json')['pass_all']
    if not smoke:
        proof=read(PUBLIC/'cases/two_pair_smoke/complete.json')
        assert proof['plan_sha256']==sha(PUBLIC/'plan.json')
        for name,h in proof['files'].items():assert sha(RAW/'cases/two_pair_smoke'/name)==h
        for name,h in proof['metadata_sha256'].items():assert sha(PUBLIC/'cases/two_pair_smoke'/name)==h
    assert sha(weights/'model.safetensors')==WEIGHTS
    model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda().requires_grad_(False)
    base=list(model.state_dict().items());base_hash=digest(base)
    torch.manual_seed(42);install(model,ROOT/'third_party/TCO/peft/lora.py')
    adapters={n:p for n,p in model.named_parameters() if p.requires_grad}
    initial={n:p.detach().clone() for n,p in adapters.items()}
    def reset():
        with torch.no_grad():
            for n,p in adapters.items():p.copy_(initial[n]);p.grad=None
        torch.manual_seed(42);np.random.seed(42)
    current=None;features=None
    execution_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    jobs=manifest['jobs'] if not smoke else [dict(scene='heads',grid='centered',arm='J',case='two_pair_smoke',post=[])]
    for job in jobs:
        case=job['case'];public=PUBLIC/'cases'/case;raw=RAW/'cases'/case
        if (public/'failure.json').exists():raise ValueError('Preserved failed case requires engineering review: '+case)
        if (public/'complete.json').exists():
            done=read(public/'complete.json')
            assert done['plan_sha256']==sha(PUBLIC/'plan.json')
            for name,h in done['files'].items():assert sha(raw/name)==h
            for name,h in done['metadata_sha256'].items():assert sha(public/name)==h
            print(case,'already verified',flush=True);continue
        try:
            reset()
            slug=job['scene'];key=slug+'_base'
            if current!=slug:
                _,_,views=views_for(key);features=training_features(model,views);current=slug
            if job['grid']!='none':
                with np.load(PACKETS[job['grid']]) as p:samples=[p[f'{slug}_{f}'] for f in FRAMES]
            with np.load(PUBLIC/'pose_packets.npz') as p:
                poses={kind:p[slug+'_'+kind] for kind in ('clean','noisy')}
            raw.mkdir(parents=True,exist_ok=True)
            if job['arm']!='zero':
                kind='noisy' if job['arm']=='N' else 'clean'
                record,evidence=train(model,features,samples,poses[kind],job['arm'],1 if smoke else 40)
                if digest(base)!=base_hash:raise ValueError('Frozen original model state changed')
                write_arrays(raw/'gauge_support.npz',evidence)
                write_arrays(raw/'adapters.npz',{n:p.detach().cpu().numpy() for n,p in adapters.items()})
                dump(public/'training.json',dict(job=job,original_state_sha256=base_hash,**record))
            cache=RAW/'cache'/(slug+'_zero' if job['arm']=='zero' else case)
            for post in job['post']:
                if post=='ordinary':capture(model,key,cache,raw,public,post)
                else:
                    kind='noisy' if post=='KDP_noisy' else 'clean'
                    weight=0. if post=='KD' else .01
                    def factory(arguments):return CalibratedPriors(alignment,arguments,samples,poses[kind],intrinsic(),weight)
                    with optimizer_hook(alignment,factory) as callbacks:capture(model,key,cache,raw,public,post)
                    assert len(callbacks)==1 and len(callbacks[0].trace)==300
                    assert all(v.get('finite_gradient') and v.get('received_gradient')==1. for v in callbacks[0].trace)
                    dump(public/(post+'_trace.json'),callbacks[0].trace)
            if digest(base)!=base_hash:raise ValueError('Original state changed after alignment')
            files={p.name:sha(p) for p in sorted(raw.glob('*.npz'))}
            dump(public/'complete.json',dict(job=job,plan_sha256=sha(PUBLIC/'plan.json'),execution_commit=execution_commit,
                original_state_sha256=base_hash,files=files,metadata_sha256={p.name:sha(p) for p in sorted(public.glob('*.json'))}))
            print(case,'completed and sealed',flush=True)
        except Exception as error:
            dump(public/'failure.json',dict(job=job,error_type=type(error).__name__,message=str(error),execution_commit=execution_commit))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.prepare:print('Registered reconstructions:',prepare()['reconstructions'])
    elif args.weights is None:p.error('--weights required')
    else:run(args.weights,args.smoke)
