"""CPU checks of production canonical fusion, fixed calibration and autograd."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from profiled_calibration import (assemble_system, canonical_depth, prepare_calibration,
                                  profiled_objective, query_residual, solve_scales, system_diagnostics)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/lora_profiled_v1/solver_checks.json"


def fixture():
    height, width = 8, 10
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x, y = x.double(), y.double()
    view = torch.arange(5, dtype=torch.float64)[:, None, None]
    depth = 2.*(1.+.025*(view-2)+.02*torch.sin(.5*x+.4*y+.2*view))
    direction = 2.*(.06+.03*torch.cos(.7*x+.2*y+.3*view))
    intrinsic = torch.tensor([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]], dtype=torch.float64)
    camera = torch.eye(4, dtype=torch.float64).repeat(5,1,1)
    camera[:,0,3] = torch.arange(-2,3,dtype=torch.float64)*.2
    support_xy = torch.tensor([[xx,yy] for yy in (1,3,5) for xx in (1,3,5,7)])
    query_xy = torch.tensor([[xx,yy] for yy in (2,4,6) for xx in (2,4,6,8)])
    support = [dict(xy=support_xy.clone(), depth=torch.full((12,), 2., dtype=torch.float64)) for _ in range(5)]
    query = [dict(xy=query_xy.clone(), depth=torch.full((12,), 2., dtype=torch.float64)) for _ in range(5)]
    matches = []
    for i,j in ((0,1),(1,2),(2,3),(3,4),(0,4)):
        xy = torch.tensor([[xx,yy] for yy in (1,3,5) for xx in range(j-i, width)])
        other = xy.clone()
        other[:,0] -= j-i
        matches.append(dict(i=i,j=j,xy_i=xy,xy_j=other))
    confidence = 1. + .1*torch.cos(x+y)[None].repeat(5,1,1)
    return depth, direction, intrinsic, camera, support, query, matches, confidence


def upstream_canonical():
    path = ROOT / "third_party/mast3r/mast3r/cloud_opt/sparse_ga.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "canonical_view"]
    if len(functions) != 1:
        raise ValueError("Expected one pinned upstream canonical_view")
    namespace = dict(torch=torch, F=functional)
    compiled = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(compiled), str(path), "exec"), namespace)
    return namespace["canonical_view"], hashlib.sha256(source.encode()).hexdigest()


def expected_failure(operation):
    try:
        operation()
    except ValueError as error:
        return str(error)
    raise AssertionError("Expected an explicit invalid-input failure")


def assert_frozen(value):
    if isinstance(value, torch.Tensor):
        assert not value.requires_grad and value.grad_fn is None
    elif isinstance(value, dict):
        for child in value.values():
            assert_frozen(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_frozen(child)


def checks():
    torch.set_num_threads(2)
    torch.manual_seed(101)
    # Run the actual pinned fusion function in an isolated CPU namespace.
    points = torch.randn(4, 8, 16, 3, dtype=torch.float32)
    points[...,2] = points[...,2].abs()+2.
    points.requires_grad_()
    conf = (1.1+torch.rand(4,8,16,dtype=torch.float32)).requires_grad_()
    original, source_hash = upstream_canonical()
    reference, _, reference_conf = original(points, conf.detach(), 8)
    depth, fused = canonical_depth(points, conf)
    torch.testing.assert_close(depth, reference[...,2], rtol=0., atol=0.)
    torch.testing.assert_close(fused, reference_conf, rtol=0., atol=0.)
    sample_gradient = torch.autograd.grad(depth.square().sum(), (points,conf), allow_unused=True)
    assert sample_gradient[0].isfinite().all() and sample_gradient[1] is None
    upstream_gradient = torch.autograd.grad(reference[...,2].square().sum(), points)[0]
    torch.testing.assert_close(sample_gradient[0],upstream_gradient,rtol=0.,atol=0.)
    partner_order = torch.tensor([2,0,3,1])
    permuted_depth, permuted_conf = canonical_depth(points[partner_order],conf[partner_order])
    torch.testing.assert_close(permuted_depth, depth,rtol=2e-7,atol=5e-7)
    torch.testing.assert_close(permuted_conf,fused,rtol=2e-7,atol=5e-7)

    initial, direction, k, cameras, support, query, matches, confidence = fixture()
    state = prepare_calibration(initial,k,cameras,support,matches,confidence)
    assert_frozen(state)
    # Mutations of caller-owned observations or geometry cannot change the state.
    mutable_initial, mutable_k, mutable_camera = initial.clone(),k.clone(),cameras.clone()
    mutable_support = copy.deepcopy(support)
    cloned = prepare_calibration(mutable_initial,mutable_k,mutable_camera,mutable_support,matches,confidence)
    old_matrix, old_target = assemble_system(initial,cloned)
    mutable_initial.add_(10.)
    mutable_k.add_(20.)
    mutable_camera.add_(30.)
    mutable_support[0]["depth"].add_(40.)
    new_matrix,new_target = assemble_system(initial,cloned)
    torch.testing.assert_close(old_matrix,new_matrix,rtol=0.,atol=0.)
    torch.testing.assert_close(old_target,new_target,rtol=0.,atol=0.)

    theta = torch.tensor(.17,dtype=torch.float64,requires_grad=True)
    current = initial+theta*direction
    loss, scales = profiled_objective(current,state,query)
    full_gradient = torch.autograd.grad(loss,theta)[0]
    stopped_loss, stopped_scales = profiled_objective(initial+theta*direction,state,query,detach_solve=True)
    stopped_gradient = torch.autograd.grad(stopped_loss,theta)[0]
    torch.testing.assert_close(loss,stopped_loss,rtol=0.,atol=0.)
    torch.testing.assert_close(scales,stopped_scales,rtol=0.,atol=0.)
    assert not stopped_scales.requires_grad
    assert abs(full_gradient-stopped_gradient)>1e-4
    epsilon = 1e-6
    def outer(t):
        return profiled_objective(initial+t*direction,state,query)[0]
    finite = (outer(float(theta.detach())+epsilon)-outer(float(theta.detach())-epsilon))/(2*epsilon)
    torch.testing.assert_close(full_gradient,finite,rtol=2e-6,atol=2e-8)

    a,b = assemble_system(current,state)
    da,_ = assemble_system(initial+direction,state)
    a0,_ = assemble_system(initial,state)
    da = da-a0
    h = a.T@a+state["ridge"]*torch.eye(5,dtype=torch.float64)
    ds = torch.linalg.solve(h,da.T@(b-a@scales)-a.T@da@scales)
    r = query_residual(current,state,query,scales)
    fixed_one = query_residual(current,state,query,torch.ones(5,dtype=torch.float64))
    fixed_zero = query_residual(current,state,query,torch.zeros(5,dtype=torch.float64))
    q_diag = fixed_one-fixed_zero
    rows_per_view = len(query[0]["depth"])
    q = torch.block_diag(*[v[:,None] for v in q_diag.reshape(5,rows_per_view)])
    rq1 = query_residual(initial+direction,state,query,torch.ones(5,dtype=torch.float64))
    rq0 = query_residual(initial,state,query,torch.ones(5,dtype=torch.float64))
    dq = torch.block_diag(*[v[:,None] for v in (rq1-rq0).reshape(5,rows_per_view)])
    analytical = r@(dq@scales+q@ds)
    torch.testing.assert_close(full_gradient,analytical,rtol=2e-10,atol=2e-10)
    stationarity = a.T@(a@scales-b)+state["ridge"]*(scales-1.)
    assert stationarity.abs().max()<1e-9
    sensor_a,_ = assemble_system(initial,state,graph_weight=0.)
    torch.testing.assert_close(state["ridge"],.001*sensor_a.square().sum()/5,rtol=0.,atol=0.)

    # A physically exact plane must satisfy every sensor and 3D graph row.
    plane = torch.full_like(initial,2.)
    exact = prepare_calibration(plane,k,cameras,support,matches,confidence)
    exact_a,exact_b = assemble_system(plane,exact)
    row_error = float((exact_a@torch.ones(5,dtype=torch.float64)-exact_b).abs().max())
    assert row_error<1e-12
    # Verify equal-view and equal-edge normalization with explicit independent rows.
    expected_sensor = 2./(.04*np.sqrt(5*12))
    np.testing.assert_allclose(float(exact_a[0,0]),expected_sensor,rtol=1e-12)
    edge = exact["edges"][0]
    first = edge["xy_i"][0]
    second = edge["xy_j"][0]
    ray_i = torch.linalg.solve(k,torch.cat([first.double(),torch.ones(1,dtype=torch.float64)]))
    ray_j = torch.linalg.solve(k,torch.cat([second.double(),torch.ones(1,dtype=torch.float64)]))
    expected_factor = torch.sqrt(.1*edge["confidence"][0]/(len(exact["edges"])*len(edge["xy_i"]))) / .04
    expected_rows = torch.zeros(3,5,dtype=torch.float64)
    expected_rows[:,0] = 2.*ray_i*expected_factor
    expected_rows[:,1] = -2.*ray_j*expected_factor
    torch.testing.assert_close(exact_a[60:63],expected_rows,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(exact_b[60:63],(cameras[1,:3,3]-cameras[0,:3,3])*expected_factor,rtol=1e-12,atol=1e-12)

    # Reorder views and graph orientation consistently; physical solution is invariant.
    permutation = [3,0,4,1,2]
    inverse = {old:new for new,old in enumerate(permutation)}
    reordered = []
    for edge in matches:
        i,j = inverse[edge["i"]],inverse[edge["j"]]
        xi,xj = edge["xy_i"],edge["xy_j"]
        if i>j:
            i,j,xi,xj = j,i,xj,xi
        reordered.append(dict(i=i,j=j,xy_i=xi,xy_j=xj))
    ordered = prepare_calibration(initial[permutation],k,cameras[permutation],
                                  [support[i] for i in permutation],reordered,confidence[permutation])
    perm_loss,perm_scales = profiled_objective(current[permutation],ordered,[query[i] for i in permutation])
    torch.testing.assert_close(perm_scales,scales[permutation],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(perm_loss,loss,rtol=1e-12,atol=1e-12)

    failures = {}
    missing = copy.deepcopy(support)
    missing[2] = dict(xy=torch.empty((0,2),dtype=torch.long),depth=torch.empty(0))
    failures["missing_support"] = expected_failure(lambda:prepare_calibration(initial,k,cameras,missing,matches,confidence))
    failures["negative_scale"] = expected_failure(lambda:solve_scales(torch.eye(5,dtype=torch.float64),-torch.ones(5,dtype=torch.float64),.001))
    failures["query_support_overlap"] = expected_failure(lambda:profiled_objective(initial,state,support))
    invalid_support_depth = initial.clone()
    invalid_support_depth[0,1,1] = -1.
    failures["initial_negative_support_depth"] = expected_failure(lambda:prepare_calibration(invalid_support_depth,k,cameras,support,matches,confidence))
    invalid_query = copy.deepcopy(query)
    invalid_query[0]["xy"][0,0] = 100
    failures["out_of_bounds_query"] = expected_failure(lambda:profiled_objective(initial,state,invalid_query))

    # Invalid pixels outside support do not invalidate all geometry. Fixed graph
    # endpoint filtering records its removals before any adapter change.
    invalid_dense = initial.clone()
    invalid_dense[0,0,0] = -1.
    prepare_calibration(invalid_dense,k,cameras,support,matches,confidence)
    invalid_conf = confidence.clone()
    invalid_conf[0,1,2] = 0.
    filtered = prepare_calibration(initial,k,cameras,support,matches,invalid_conf)
    assert sum(e["invalid_confidence"] for e in filtered["graph_filtering"])>0
    invalid_endpoint = initial.clone()
    invalid_endpoint[0,1,2] = -1.
    filtered_depth = prepare_calibration(invalid_endpoint,k,cameras,support,matches,confidence)
    assert sum(e["nonpositive_depth"] for e in filtered_depth["graph_filtering"])>0
    disconnected = prepare_calibration(initial,k,cameras,support,matches[:2]+[matches[3]],confidence)
    disconnected_a,disconnected_b = assemble_system(initial,disconnected)
    disconnected_scale = solve_scales(disconnected_a,disconnected_b,disconnected["ridge"])
    disconnected_report = system_diagnostics(disconnected_a,disconnected,disconnected_scale)
    assert disconnected_report["data_rank"]==5 and not disconnected_report["graph_connected"]
    report = system_diagnostics(a,state,scales)
    assert report["data_rank"]==5 and report["graph_connected"] and report["every_view_supported"]
    # Ridge may solve an unobserved scale, but must not make the data rank look
    # full. The returned scale on this deliberately zero column is only a prior.
    deficient_a = a.detach().clone()
    deficient_a[:,2] = 0.
    deficient_scales = solve_scales(deficient_a,b,state["ridge"])
    deficient_report = system_diagnostics(deficient_a,state,deficient_scales)
    assert deficient_report["data_rank"]==4 and deficient_report["singular_ratio"]<1e-14
    torch.testing.assert_close(deficient_scales[2],torch.tensor(1.,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    return dict(scope="CPU production-layer validation on synthetic geometry. No model adapter update or accuracy measurement.",
                torch_version=torch.__version__, device="cpu", dtype="float64 solve / float32 canonical bridge",
                upstream_source_sha256=source_hash, upstream_canonical_values_exact=True, upstream_canonical_gradient_exact=True,
                confidence_gradient_blocked=True, full_detached_forward_exact=True,
                implicit_gradient=float(full_gradient), detached_gradient=float(stopped_gradient),
                analytical_gradient=float(analytical), finite_difference_gradient=float(finite),
                finite_difference_absolute_error=float(abs(full_gradient-finite)),
                normal_equation_max_residual=float(stationarity.abs().max()), exact_plane_row_error=row_error,
                view_permutation_pass=True, frozen_state_mutation_pass=True, ridge_sensor_only_pass=True,
                unrelated_negative_pixel_permitted=True, invalid_confidence_filtered=True,
                initial_nonpositive_match_filtered=True, rank_deficiency_detected_despite_positive_ridge=True,
                diagnostics=report, disconnected_diagnostics=disconnected_report,
                rank_deficient_diagnostics=deficient_report, expected_failures=failures,
                pass_all=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    result = checks()
    if args.verify:
        old = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for name in ("implicit_gradient","detached_gradient","analytical_gradient","finite_difference_gradient"):
            np.testing.assert_allclose(result[name],old[name],rtol=2e-6,atol=2e-8)
        if not old["pass_all"]:
            raise ValueError("Stored validation is not passing")
    else:
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        with OUTPUT.open("w",encoding="utf-8",newline="\n") as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:result[k] for k in ("pass_all","implicit_gradient","detached_gradient","finite_difference_absolute_error")}))
