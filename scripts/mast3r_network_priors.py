"""Differentiable pair-frame constraints and decoder-only adaptation utilities."""
import hashlib
import importlib.util
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def camera_points(first,other,relative,scale):
    return scale*first,(scale*other)@relative[:3,:3].T+relative[:3,3]


def huber(residual):
    return F.huber_loss(residual,torch.zeros_like(residual),delta=1.)


def bearing(points,rays):
    direction=F.normalize(points,dim=-1,eps=1e-6)
    return huber((direction-F.normalize(rays,dim=-1,eps=1e-6))/.02)


def objective(outputs,scales,support,measured,unit,relatives,rays,pose_weight,intrinsic_weight=0.):
    x,y=support
    metric=[camera_points(a[0]['pts3d'][0],a[1]['pts3d_in_other_view'][0],r,s)
            for a,r,s in zip(outputs,relatives,scales)]
    depth=torch.stack([huber((a[y,x,2]-measured)/(.02*unit)) for a,b in metric]).mean()
    # A fixed image grid includes every pixel selected before any optimization.
    consistency=huber((metric[0][0][8::16,8::16]-metric[1][0][8::16,8::16])/(.02*unit))
    intrinsic=torch.stack([bearing(a[8::16,8::16],rays) for a,b in metric]).mean()
    camera=torch.stack([bearing(b[8::16,8::16],rays) for a,b in metric]).mean()
    total=depth+.1*consistency+intrinsic_weight*intrinsic+pose_weight*camera
    return total,dict(depth=depth,consistency=consistency,intrinsic=intrinsic,bearing=camera)


def install(model,lora_path):
    model.requires_grad_(False)
    spec=importlib.util.spec_from_file_location('tco_decoder_lora',lora_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    selected=[]
    for name,layer in list(model.named_modules()):
        parts=name.split('.')
        if parts[0] not in ('dec_blocks','dec_blocks2') or not isinstance(layer,torch.nn.Linear):
            continue
        if parts[-2] not in ('attn','cross_attn'):
            continue
        parent=model.get_submodule('.'.join(parts[:-1]))
        setattr(parent,parts[-1],module.LoRALayer(layer,rank=4,alpha=1,dropout=0.))
        selected.append(name)
    for block in list(model.dec_blocks)+list(model.dec_blocks2):
        original=block.forward
        def checked(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        block.forward=checked
    return selected


def digest(state):
    result=hashlib.sha256()
    for name,tensor in state:
        a=tensor.detach().cpu().contiguous().numpy()
        result.update(name.encode());result.update(str(a.dtype).encode());result.update(str(a.shape).encode());result.update(a.tobytes())
    return result.hexdigest()


def native_samples(values,device):
    pixels=np.rint((values[:,:2]+.5)*[512/518,384/392]-.5).astype(int)
    inside=((pixels>=0)&(pixels<[512,384])).all(axis=1)
    nearest={}
    for j in np.argsort(values[:,2]):
        if inside[j]: nearest.setdefault(tuple(pixels[j]),j)
    selected=np.array(list(nearest.values()))
    if len(selected)<8: raise ValueError('Insufficient native sparse support')
    xy=torch.as_tensor(pixels[selected].T,dtype=torch.long,device=device)
    measured=torch.as_tensor(values[selected,2],dtype=torch.float32,device=device)
    return xy,measured
