"""Compare ordinary MASt3R alignment with the disabled research hook on identical inputs."""
import argparse
from pathlib import Path

import numpy as np
import torch
import mast3r.cloud_opt.sparse_ga as alignment
from mast3r.model import AsymmetricMASt3R

from capture_mast3r_transfer import capture,WEIGHTS
from evaluation_stability import dump,sha
from mast3r_alignment_hook import optimizer_hook
from prepare_dense_focal_pilot import ROOT

PUBLIC=ROOT/'results/mast3r_joint_priors'
RAW=ROOT/'reports/mast3r_joint_priors'


def run(weights):
    if sha(weights/'model.safetensors')!=WEIGHTS:
        raise ValueError('Unexpected checkpoint')
    model=AsymmetricMASt3R.from_pretrained(str(weights),local_files_only=True).eval().cuda()
    key='heads_base'
    capture(model,key,RAW/'standard',PUBLIC/'standard')
    original=alignment.sparse_scene_optimizer
    with optimizer_hook(alignment):
        capture(model,key,RAW/'disabled_hook',PUBLIC/'disabled_hook')
    assert alignment.sparse_scene_optimizer is original
    differences={}
    with np.load(RAW/'standard'/(key+'.npz')) as a,np.load(RAW/'disabled_hook'/(key+'.npz')) as b:
        for field in a:
            assert a[field].shape==b[field].shape and np.isfinite(b[field]).all()
            delta=np.abs(a[field].astype(float)-b[field].astype(float))
            differences[field]=dict(exact=bool(np.array_equal(a[field],b[field])),max_absolute_difference=float(delta.max()),
                                   mean_absolute_difference=float(delta.mean()),close=bool(np.allclose(a[field],b[field],rtol=1e-5,atol=1e-6)))
    report=dict(checkpoint_sha256=WEIGHTS,
        standard_metadata_sha256=sha(PUBLIC/'standard'/(key+'.json')),
        disabled_metadata_sha256=sha(PUBLIC/'disabled_hook'/(key+'.json')),
        optimizer_restored=True,differences=differences,pass_all=all(v['close'] for v in differences.values()))
    dump(PUBLIC/'disabled_hook_bridge.json',report)
    print('Disabled-hook bridge:',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    run(p.parse_args().weights)
