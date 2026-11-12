"""Verify joint-prior gradients through a real, matched MASt3R fine-alignment update."""
import argparse
from pathlib import Path

import numpy as np
import torch
import mast3r.cloud_opt.sparse_ga as alignment
from mast3r.model import AsymmetricMASt3R

from capture_mast3r_transfer import capture,WEIGHTS
from evaluation_stability import dump,sha
from mast3r_alignment_hook import optimizer_hook
from mast3r_joint_prior_loss import JointPriors
from prepare_dense_focal_pilot import ROOT
from collect_relative_focal_windows import FRAMES
from score_saved_geometry import read

PUBLIC=ROOT/'results/mast3r_joint_priors'
RAW=ROOT/'reports/mast3r_joint_priors'


def run(weights):
    if not read(PUBLIC/'disabled_hook_bridge.json')['pass_all']:
        raise ValueError('Disabled hook bridge must pass first')
    assert sha(weights/'model.safetensors')==WEIGHTS
    key='heads_base'
    with np.load(ROOT/'results/neighbor_sparse_depth/measurements.npz') as data:
        samples=[data[f'heads_{frame}'] for frame in FRAMES]
    with np.load(ROOT/'results/supplied_pose_sparse_depth/supplied_poses.npz') as data:
        cameras=data[key]
    model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda()
    # Upstream returns geometry evaluated before its last optimizer update.
    # Two fine iterations expose the first update in the returned geometry.
    options=dict(niter2=2)
    capture(model,key,RAW/'fine_standard',PUBLIC/'fine_standard',options)
    original=alignment.sparse_scene_optimizer
    def factory(arguments):
        return JointPriors(alignment,arguments,samples,cameras)
    with optimizer_hook(alignment,factory) as callbacks:
        capture(model,key,RAW/'fine_joint',PUBLIC/'fine_joint',options)
    assert alignment.sparse_scene_optimizer is original and len(callbacks)==1
    callback=callbacks[0]
    trace=callback.trace
    if len(trace)!=2 or not all(v.get('finite_gradient') and v.get('received_gradient')==1. for v in trace):
        raise ValueError('The prior was not included in both fine-stage backward passes')
    changes={}
    with np.load(RAW/'fine_standard'/(key+'.npz')) as a,np.load(RAW/'fine_joint'/(key+'.npz')) as b:
        for name in a:
            difference=np.abs(a[name].astype(float)-b[name].astype(float))
            changes[name]=dict(max_absolute_difference=float(difference.max()),mean_absolute_difference=float(difference.mean()))
    influenced=any(v['max_absolute_difference']>1e-10 for v in changes.values())
    result=dict(checkpoint_sha256=WEIGHTS,standard_sha256=sha(RAW/'fine_standard'/(key+'.npz')),
                joint_sha256=sha(RAW/'fine_joint'/(key+'.npz')),native_sample_counts=callback.counts,
                fine_iterations=2,returned_after_fine_updates=1,prior_trace=trace,output_changes=changes,
                optimizer_restored=True,gradient_and_output_check_passed=influenced)
    dump(PUBLIC/'joint_gradient_probe.json',result)
    print('Joint-prior gradient/output integration:',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    run(p.parse_args().weights)
