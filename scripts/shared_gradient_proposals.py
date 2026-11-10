"""Transactional optimizer proposals from once-captured component gradients.

Engineering primitive for an unvalidated research hypothesis. This does not
compute a visual validation signal or decide which proposal is correct. Scope:
optimizer-owned parameters/gradients/state and Python/NumPy/Torch RNG. Model
buffers and training/eval mode must be fixed by the caller.
"""
import copy
import random
import numpy as np
import torch


def frozen(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:frozen(item) for key,item in value.items()}
    if isinstance(value,list):return [frozen(item) for item in value]
    if isinstance(value,tuple):return tuple(frozen(item) for item in value)
    return copy.deepcopy(value)


class OptimizerSnapshot:
    def __init__(self,optimizer):
        self.optimizer=optimizer
        self.parameters=[p for group in optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in self.parameters})!=len(self.parameters):raise ValueError("Duplicate optimizer parameter")
        self.values=[frozen(p) for p in self.parameters]
        self.gradients=[None if p.grad is None else frozen(p.grad) for p in self.parameters]
        self.state=frozen(optimizer.state_dict())
        self.python_rng=random.getstate();self.numpy_rng=copy.deepcopy(np.random.get_state())
        self.torch_rng=torch.get_rng_state().clone()
        self.cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None

    def restore(self):
        current=[p for group in self.optimizer.param_groups for p in group["params"]]
        if [id(p) for p in current]!=[id(p) for p in self.parameters]:raise ValueError("Optimizer structure changed")
        with torch.no_grad():
            for parameter,value,gradient in zip(self.parameters,self.values,self.gradients):
                parameter.copy_(value.to(parameter))
                parameter.grad=None if gradient is None else gradient.to(parameter).clone()
        # Never lend mutable snapshot moment tensors to the live optimizer.
        self.optimizer.load_state_dict(copy.deepcopy(self.state))
        random.setstate(self.python_rng);np.random.set_state(copy.deepcopy(self.numpy_rng))
        torch.set_rng_state(self.torch_rng.clone())
        if self.cuda_rng is not None:torch.cuda.set_rng_state_all([s.clone() for s in self.cuda_rng])


def propose(optimizer,gradient_groups,proposals,evaluate,max_norm=1.0):
    """Return candidate states/scores, leaving the original state exactly restored.

    Groups already include their loss weights. Clipping and optimizer updates
    are recomputed for each group combination from the same original moments.
    An empty proposal performs no update. All evaluations/proposals count toward
    inference cost. Accepting a returned state requires separate validated logic.
    """
    if not np.isfinite(max_norm) or max_norm<=0:raise ValueError("Invalid clipping norm")
    anchor=OptimizerSnapshot(optimizer);parameters=anchor.parameters
    groups={}
    for name,gradients in gradient_groups.items():
        if len(gradients)!=len(parameters):raise ValueError("Gradient/parameter count mismatch")
        groups[name]=[]
        for gradient,parameter in zip(gradients,parameters):
            if gradient is not None and (gradient.shape!=parameter.shape or not torch.isfinite(gradient).all()):
                raise ValueError("Invalid component gradient")
            groups[name].append(None if gradient is None else frozen(gradient))
    for names in proposals.values():
        if len(names)!=len(set(names)) or any(name not in groups for name in names):raise ValueError("Invalid proposal groups")
    results={}
    try:
        for name,names in proposals.items():
            anchor.restore()
            for index,parameter in enumerate(parameters):
                components=[groups[key][index] for key in sorted(names) if groups[key][index] is not None]
                parameter.grad=sum((value.to(parameter) for value in components),torch.zeros_like(parameter)) if components else None
            if any(p.grad is not None for p in parameters):
                torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=True)
                optimizer.step()
            candidate=OptimizerSnapshot(optimizer)
            with torch.no_grad():value=frozen(evaluate(name))
            results[name]=dict(state=candidate,evaluation=value)
    finally:
        anchor.restore()
    return results
