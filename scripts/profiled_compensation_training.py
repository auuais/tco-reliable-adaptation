"""Two-pass MASt3R training operations shared by the redesign and its acceptance."""
import torch

from profiled_calibration import assemble_system
from profiled_compensation_penalty import compensation_loss, prepare_fold, query_matrix
from profiled_network import PAIRS, fuse, prepare_training, training_loss

PENALTIES = {"R0": "matched_support", "R1": "compensation", "R2": "compensation"}


def prepare(case, pristine, confidence):
    prepared = prepare_training(case, pristine, confidence)
    depth, _ = fuse(pristine, confidence)
    prepared["compensation_references"] = [prepare_fold(depth, state, query)
        for state, query in zip(prepared["states"], prepared["queries"])]
    return prepared


def loss_for(points, prepared, arm, step, *, measure=False):
    if arm == "P1":
        loss, scales = training_loss(points, prepared, arm, step)
        return loss, scales, None
    if arm != "P3" and arm not in PENALTIES:
        raise ValueError("Unsupported compensation training arm")
    fold = step % 2
    state, query = prepared["states"][fold], prepared["queries"][fold]
    reference = prepared["compensation_references"][fold]
    depths, _ = fuse(points, prepared["confidence"])
    loss, scales, terms = compensation_loss(depths, state, query, reference,
        penalty=PENALTIES.get(arm, "none"), detach_solve=arm == "R2")
    metrics = None
    if measure:
        metrics = measure_terms(depths.detach(), scales.detach(), state, query, reference, terms)
    return loss, scales, metrics


@torch.no_grad()
def measure_terms(depths, scales, state, query, reference, terms):
    a, _ = assemble_system(depths, state, graph_weight=0.)
    q = query_matrix(depths, state, query)
    change = scales - reference["s0"]
    # One active camera coefficient per row, so column norms partition the energy.
    c2 = q.square().sum(dim=0) * change.square()
    x = (a - reference["a0"]) * reference["s0"]
    ordinary = .5 * x.square().sum(dim=0) * reference["damping"]
    return dict(base=float(terms["base"]), penalty=float(terms["regularizer"]),
                compensation_l2_by_view=c2.sqrt().cpu().tolist(),
                compensation_l2=float(c2.sum().sqrt()),
                matched_support_penalty_by_view=ordinary.cpu().tolist())


def output_adjoints(points, prepared, arm, step, varying_pairs=PAIRS, *, measure=False):
    leaves = dict(points)
    variables = []
    for pair in varying_pairs:
        leaves[pair] = tuple(value.detach().requires_grad_(True) for value in points[pair])
        variables.extend(leaves[pair])
    loss, scales, metrics = loss_for(leaves, prepared, arm, step, measure=measure)
    if not bool(torch.isfinite(loss)) or (metrics and
            any(not bool(torch.isfinite(torch.as_tensor(v)).all()) for v in metrics.values())):
        raise FloatingPointError("Nonfinite redesign training objective or trace")
    gradients = torch.autograd.grad(loss, variables, allow_unused=True)
    adjoints = {pair: tuple(gradients[2*i:2*i+2]) for i, pair in enumerate(varying_pairs)}
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Nonfinite redesign output adjoint")
    return loss.detach(), None if scales is None else scales.detach(), adjoints, metrics
