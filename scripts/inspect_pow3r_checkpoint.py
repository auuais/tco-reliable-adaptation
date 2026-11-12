"""Validate the pinned official checkpoint's model schema without GPU allocation."""
import argparse
import ast
from collections import Counter
from pathlib import Path
import torch
from pow3r.model import Pow3R
from evaluation_stability import dump
from prepare_dense_focal_pilot import ROOT
from prepare_prior_aware_baselines import digest
from score_saved_geometry import read


def inspect(path):
    expected=read(ROOT/'results/prior_aware_baselines/preparation.json')['pow3r_checkpoint']
    assert path.stat().st_size==expected['bytes'] and digest(path)==expected['sha256']
    checkpoint=torch.load(path,map_location='meta',mmap=True,weights_only=False)
    definition=checkpoint['definition']
    tree=ast.parse(definition,mode='eval').body
    assert isinstance(tree,ast.Call) and isinstance(tree.func,ast.Name) and tree.func.id=='Pow3R'
    assert all(k.arg is not None for k in tree.keywords)
    args=[ast.literal_eval(a) for a in tree.args]
    kwargs={k.arg:ast.literal_eval(k.value) for k in tree.keywords}
    with torch.device('meta'):
        model=Pow3R(*args,**kwargs)
    status=model.load_state_dict(checkpoint['weights'],strict=True,assign=True)
    assert not status.missing_keys and not status.unexpected_keys
    value=dict(checkpoint_sha256=expected['sha256'],definition=definition,epoch=checkpoint['epoch'],
        parameter_count=sum(p.numel() for p in model.parameters()),weight_tensors=len(checkpoint['weights']),
        tensor_dtypes=dict(Counter(str(t.dtype) for t in checkpoint['weights'].values())),
        strict_schema_match=True,device='meta',gpu_inference_performed=False)
    dump(ROOT/'results/prior_aware_baselines/pow3r_structure.json',value)
    print('Pow3R checkpoint schema matched:',value['parameter_count'],'parameters; no GPU allocation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    inspect(p.parse_args().checkpoint)
