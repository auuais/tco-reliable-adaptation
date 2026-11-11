"""Choose a supplied focal prior using the original TCO forward pass."""
from contextlib import contextmanager
import time

import numpy as np
import torch

from focal_metadata_control import focal_proposal
from relative_focal_confirmation import confirmation


@contextmanager
def initial_focal_policy(model, arm, method, geometric_support, observe):
    """Install a one-batch experimental policy without changing model weights or inputs."""
    if arm not in ("retain","consensus","adaptive") or method not in ("relative","relative_with_folds"):
        raise ValueError("Unknown dense focal policy")
    if method == "relative_with_folds" and type(geometric_support) is not bool:
        raise ValueError("The geometry-dependent policy requires image support")
    state = {}
    original = model.forward_original_vggt
    previous = model.__dict__.get("forward_original_vggt")

    def isolate(module, args, kwargs):
        if state:
            raise RuntimeError("Use a fresh policy context for each batch")
        supplied = kwargs["gt_intrinsics"]
        if supplied.shape != (1,5,3,3):
            raise ValueError("Expected one five-view TCO input")
        state["k"] = supplied.detach().clone()
        return args, dict(kwargs,gt_intrinsics=state["k"])

    def initial(images):
        tokens, heads = original(images)
        if "decision" in state:
            raise RuntimeError("The baseline head ran twice")
        started = time.perf_counter()
        k = state["k"]
        supplied = k[0].detach().float().cpu().numpy().copy()
        predicted = heads["intrinsics"][0].detach().float().cpu().numpy()[:,[0,1],[0,1]]
        v, opts = confirmation(supplied[:,[0,1],[0,1]],predicted,bool(geometric_support))
        edit = (arm == "consensus" and v is not None) or (arm == "adaptive" and opts[method])
        if edit:
            _, replacement = focal_proposal(supplied)
            with torch.no_grad():
                for axis in (0,1):
                    k[0,v,axis,axis] = float(replacement[axis,axis])
        actual = k[0].detach().float().cpu().numpy()
        result = dict(arm=arm, method=method, view=v, edited=bool(edit), model_choices=opts,
                      geometric_support=geometric_support, model_focal=predicted.tolist(),
                      supplied_focal=supplied[:,[0,1],[0,1]].tolist(), used_focal=actual[:,[0,1],[0,1]].tolist(),
                      decision_seconds=time.perf_counter()-started)
        state["decision"] = result
        observe(heads,result)
        return tokens,heads

    hook = model.register_forward_pre_hook(isolate,with_kwargs=True)
    model.forward_original_vggt = initial
    try:
        yield
        if "decision" not in state:
            raise RuntimeError("The normal TCO initial head was never observed")
    finally:
        hook.remove()
        if previous is None:
            del model.forward_original_vggt
        else:
            model.forward_original_vggt = previous
