"""Run a predeclared 21-case local MASt3R prior-alignment factorial without evaluation labels."""
import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import torch
import mast3r.cloud_opt.sparse_ga as alignment
from mast3r.model import AsymmetricMASt3R

from capture_mast3r_transfer import capture,WEIGHTS
from evaluation_stability import dump,sha
from mast3r_alignment_hook import optimizer_hook,EXPECTED_SOURCE
from mast3r_joint_prior_loss import JointPriors
from prepare_dense_focal_pilot import ROOT,INPUT
from run_depth_scale_study import PLAN
from collect_relative_focal_windows import FRAMES
from score_saved_geometry import read

PUBLIC=ROOT/'results/mast3r_joint_priors'
RAW=ROOT/'reports/mast3r_joint_priors'
PACKETS={'centered':ROOT/'results/neighbor_sparse_depth/measurements.npz','shifted':ROOT/'results/shifted_sparse_grid/measurements.npz'}
POSES=ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz'
ARMS={'depth':(.01,0.),'pose':(0.,.01),'joint':(.01,.01)}


def source_sha(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()


def plan():
    jobs=[]
    for slug in ('heads','chess','fire'):
        jobs.append(dict(scene=slug,grid='none',arm='standard',run_id=slug+'_standard'))
        for grid in PACKETS:
            for arm in ARMS:
                jobs.append(dict(scene=slug,grid=grid,arm=arm,run_id=f'{slug}_{grid}_{arm}'))
    rgb=read(PLAN)['inputs']
    manifest=dict(checkpoint_sha256=WEIGHTS,upstream_optimizer_sha256=EXPECTED_SOURCE,
        rgb_sha256={s:sha(INPUT/rgb[s+'_base']['file']) for s in ('heads','chess','fire')},
        cache_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p)
                      for s in ('heads','chess','fire') for p in sorted((ROOT/'reports/mast3r_transfer/cache'/(s+'_base')).rglob('*')) if p.is_file()},
        packet_sha256={g:sha(p) for g,p in PACKETS.items()},poses_sha256=sha(POSES),weights=ARMS,
        source_sha256={name:source_sha(ROOT/name) for name in ('scripts/mast3r_alignment_hook.py','scripts/mast3r_joint_prior_loss.py','scripts/capture_mast3r_transfer.py','scripts/run_mast3r_joint_priors.py','docs/MAST3R_JOINT_PRIORS.md')},jobs=jobs)
    path=PUBLIC/'alignment_plan.json'
    if path.exists():
        # JSON stores coefficient tuples as arrays.
        import json
        if read(path)!=json.loads(json.dumps(manifest)):
            raise ValueError('Registered alignment plan changed')
    dump(path,manifest)
    return manifest


def run(weights):
    manifest=plan()
    execution_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if not read(PUBLIC/'disabled_hook_bridge.json')['pass_all'] or not read(PUBLIC/'joint_gradient_probe.json')['gradient_and_output_check_passed']:
        raise ValueError('Both integration checks must pass first')
    assert sha(weights/'model.safetensors')==WEIGHTS
    model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda()
    with np.load(POSES) as poses:
        for job in manifest['jobs']:
            case=job['run_id']
            key=job['scene']+'_base'
            raw=RAW/'full'/case
            public=PUBLIC/'full'/case
            done=public/'complete.json'
            if done.exists():
                record=read(done)
                assert record['plan_sha256']==sha(PUBLIC/'alignment_plan.json')
                assert record['native_sha256']==sha(raw/(key+'.npz'))
                assert record['trace_sha256']==sha(public/'prior_trace.json')
                print(case,'already verified',flush=True)
                continue
            try:
                if job['arm']=='standard':
                    capture(model,key,raw,public)
                    traces=[]
                    counts=[]
                else:
                    with np.load(PACKETS[job['grid']]) as values:
                        samples=[values[f'{job["scene"]}_{frame}'] for frame in FRAMES]
                    dw,pw=ARMS[job['arm']]
                    def factory(arguments):
                        return JointPriors(alignment,arguments,samples,poses[key],dw,pw)
                    with optimizer_hook(alignment,factory) as callbacks:
                        capture(model,key,raw,public)
                    assert len(callbacks)==1
                    traces=callbacks[0].trace
                    counts=callbacks[0].counts
                    if len(traces)!=300 or not all(v.get('finite_gradient') and v.get('received_gradient')==1 for v in traces):
                        raise ValueError('Incomplete prior backward trace')
                dump(public/'prior_trace.json',dict(job=job,native_sample_counts=counts,rows=traces))
                record=dict(job=job,execution_commit=execution_commit,plan_sha256=sha(PUBLIC/'alignment_plan.json'),native_sha256=sha(raw/(key+'.npz')),
                            capture_sha256=sha(public/(key+'.json')),trace_sha256=sha(public/'prior_trace.json'))
                dump(done,record)
                print(case,'completed and sealed',flush=True)
            except Exception as error:
                dump(public/'failure.json',dict(job=job,error_type=type(error).__name__,message=str(error)))
                raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path)
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare:
        print('Registered',len(plan()['jobs']),'local alignment runs.')
    else:
        if args.weights is None:
            p.error('--weights required')
        run(args.weights)
