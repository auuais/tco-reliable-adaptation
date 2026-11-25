"""Qualify the renderer, lifetime switches and full TCO update receipts."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

from run_profiled_d0 import read, sha, write_new


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "reports" / "tco_render_lifetime_v1"
PUBLIC = ROOT / "results" / "tco_render_lifetime_v1"
LEAVES = ("depth", "depth_conf", "camera_poses", "intrinsics")
FIXED_RTOL = 1e-4
OUTPUT_RTOL = 1e-5
ATOL = 1e-6


def upstream():
    """Import the same modules and identities as the reconstruction runner."""
    path = ROOT / "third_party" / "TCO"
    for directory in (path, path / "base_models" / "vggt"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import objective
    import tco_loss

    return objective, tco_loss


def capture_renderer_inputs(kwargs, path):
    """Save the positive-step loss arguments without changing CUDA RNG or keeping its graph."""
    import torch

    def copy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {name: copy(item) for name, item in value.items()}
        return value

    predictions = kwargs["predictions"]
    if kwargs.get("lambda_depth") or kwargs.get("lambda_redundancy"):
        raise ValueError("The renderer probe expects only pose, intrinsics and RGB terms")
    inputs = {name: copy(value) for name, value in kwargs.items() if name != "predictions"}
    inputs["predictions"] = {name: copy(predictions[name]) for name in LEAVES}
    if "pose_type" in predictions:
        inputs["predictions"]["pose_type"] = predictions["pose_type"]
    saved = dict(kwargs=inputs, rng=torch.cuda.get_rng_state(), autocast=torch.is_autocast_enabled("cuda"),
                 autocast_dtype=str(torch.get_autocast_dtype("cuda")).removeprefix("torch."))
    with Path(path).open("xb") as stream:
        torch.save(saved, stream)


def synthetic_inputs():
    import torch

    height, width, views = 18, 24, 5
    y, x = torch.meshgrid(torch.linspace(-1., 1., height), torch.linspace(-1., 1., width), indexing="ij")
    index = torch.arange(views, dtype=torch.float32).view(1, views, 1, 1)
    depth = 2.4 + 0.11 * x + 0.09 * y + 0.035 * torch.sin(4 * x + 2 * y) + 0.014 * index
    confidence = 2.2 + 0.32 * x - 0.25 * y + 0.13 * index
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
    shift = torch.linspace(-0.08, 0.08, views)
    poses[0, :, 0, 3] = shift
    poses[0, :, 1, 3] = shift.square()
    intrinsics = torch.tensor([[22., 0., (width - 1) / 2], [0., 22., (height - 1) / 2], [0., 0., 1.]])
    intrinsics = intrinsics.view(1, 1, 3, 3).repeat(1, views, 1, 1)
    rgb = torch.stack((0.5 + 0.2 * x + 0.04 * index + 0 * y, 0.5 + 0.3 * y + 0.08 * x + 0 * index,
                       0.5 + 0.15 * torch.sin(6 * x + y) - 0.03 * index), dim=2)
    prior = poses.clone()
    prior[0, :, 0, 3] = -shift.flip(0) + shift.square()
    prior[0, :, 1, 3] = -shift.square()
    return dict(predictions=dict(zip(LEAVES, (depth[..., None], confidence, poses, intrinsics))), images=rgb,
                gt_extrinsics=prior, gt_intrinsics=intrinsics * 0.99, lambda_pose=1.,
                pose_rot_loss_type="angle", pose_trans_loss_type="normed_l1", pose_translation_weight=2.,
                lambda_intrinsics=0.01, lambda_mv_consistency=0.2, mv_rgb_weight=1.,
                mv_depth_weight=0., mv_normal_weight=0., num_view_groups=100,
                meta_info=dict(sample_idx=0, tco_step=0, scene_name="renderer_fixture"))


def compare_tensor(actual, expected, rtol, atol=ATOL):
    import torch

    if actual is None or expected is None:
        return dict(passed=actual is None and expected is None, present=actual is not None,
                    expected_present=expected is not None)
    if actual.shape != expected.shape or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        return dict(passed=False, shape=list(actual.shape), expected_shape=list(expected.shape),
                    invalid_values=True)
    a, b = actual.double(), expected.double()
    delta = float(torch.linalg.vector_norm((a - b).reshape(-1)))
    ref = float(torch.linalg.vector_norm(b.reshape(-1)))
    return dict(passed=delta <= atol + rtol * ref, l2=delta, ref_l2=ref,
                zero=not bool(torch.count_nonzero(actual)), expected_zero=not bool(torch.count_nonzero(expected)))


def cpu_fixture():
    """Probe the actual guard with instrumented CPU renderer inputs and the actual early clear helper."""
    import torch
    from tco_render_lifetime import clear_gradients, patched_runtime

    objective, tco_loss = upstream()
    unproject = objective.unproject_depth_to_points

    def run(policy, normal_weight, visualize, sparse):
        kwargs = synthetic_inputs()
        kwargs["mv_normal_weight"] = normal_weight
        predictions = kwargs["predictions"]
        for value in predictions.values():
            value.requires_grad_(True)
        height, width = predictions["depth"].shape[2:4]
        state = torch.Generator().manual_seed(20261125).get_state()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state)
            perm = torch.randperm(5, generator=torch.Generator().set_state(state))
            target = predictions["depth"][0, perm, ..., 0].detach()
            boundary = target + target.mean() * 0.0001
            counts, log = [], []

            def tracked_unproject(depth, intrinsics):
                counts.append(int(depth.shape[1]))
                return unproject(depth, intrinsics)

            def raster(**render):
                alpha = render["opacities"].view(height, width)[None, ..., None].expand(5, -1, -1, -1)
                if render["render_mode"] == "ED":
                    offset = torch.full_like(target, 0.2)
                    offset[0, 0, 0] = 0.
                    if not sparse:
                        offset[1:, ::2] = -0.2
                    depth = (boundary + offset)[..., None]
                else:
                    depth = render["colors"].reshape(5, height, width, 3) * 0.91 + 0.03
                normals = alpha.expand(-1, -1, -1, 3) * 0.3
                return depth, alpha, normals, None, None, None, {}

            wb = SimpleNamespace(run=object() if visualize else None, log=lambda *a, **k: log.append(1),
                                 Image=lambda data: data)
            with patched_runtime(policy), patch.dict(sys.modules, {"gsplat": SimpleNamespace(rasterization_2dgs=raster),
                                                                    "wandb": wb}):
                with patch.object(objective, "unproject_depth_to_points", tracked_unproject):
                    loss, terms = tco_loss.compute_tco_losses(**kwargs)
                gradients = torch.autograd.grad(loss, tuple(predictions.values()), allow_unused=True)
        return dict(loss=loss.detach(), terms=terms, gradients=gradients, unproject=counts, visuals=len(log))

    cases = {}
    for normal, visuals, sparse in ((0., False, False), (1., False, False), (0., True, False), (0., False, True)):
        ref = run("native", normal, visuals, sparse)
        changed = run("n", normal, visuals, sparse)
        expected_calls = [1, 5] if normal > 0 or visuals else [1]
        result = dict(native_unproject=ref["unproject"], candidate_unproject=changed["unproject"],
                      native_visuals=ref["visuals"], candidate_visuals=changed["visuals"],
                      loss=compare_tensor(changed["loss"], ref["loss"], OUTPUT_RTOL),
                      terms={name: compare_tensor(changed["terms"][name], value, OUTPUT_RTOL)
                             for name, value in ref["terms"].items()},
                      gradients={name: compare_tensor(a, b, OUTPUT_RTOL)
                                 for name, a, b in zip(LEAVES, changed["gradients"], ref["gradients"])})
        result["passed"] = (ref["unproject"] == [1, 5] and changed["unproject"] == expected_calls
                            and ref["visuals"] == changed["visuals"] == int(visuals)
                            and result["loss"]["passed"]
                            and all(row["passed"] for field in ("terms", "gradients") for row in result[field].values()))
        cases[f"normal_{normal}_vis_{int(visuals)}_sparse_{int(sparse)}"] = result

    def adam(early, steps):
        factor = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float32))
        optim = torch.optim.Adam([factor], lr=0.0003)
        audit, grad_before_forward = [], []
        for _ in range(steps):
            if early:
                clear_gradients(optim, None)
            grad_before_forward.append(factor.grad is not None)
            loss = (factor.square() + factor * 0.3).sum()
            if not early:
                optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([factor], 1.)
            optim.step()
            state = optim.state[factor]
            audit.append({key: value.detach().tolist() for key, value in
                          dict(factor=factor, grad=factor.grad, step=state["step"], moment=state["exp_avg"],
                               square=state["exp_avg_sq"]).items()})
        return dict(audit=audit, grad_before_forward=grad_before_forward,
                    final_grad=factor.grad is not None, optimizer_states=len(optim.state))

    early, late, zero = adam(True, 2), adam(False, 2), adam(True, 0)
    grad_pass = (early["audit"] == late["audit"] and early["grad_before_forward"] == [False, False]
                 and late["grad_before_forward"] == [False, True] and early["final_grad"] and late["final_grad"]
                 and zero == dict(audit=[], grad_before_forward=[], final_grad=False, optimizer_states=0))
    return dict(torch=torch.__version__, normals=cases, early_clear=dict(passed=grad_pass, early=early, late=late,
                                                                         zero_steps=zero),
                passed=grad_pass and all(row["passed"] for row in cases.values()))


def render_replay(data, policy, negative=False):
    """Use gsplat twice and retain the actual objective VJP on identical leaves."""
    import gsplat
    import torch
    from tco_render_lifetime import patched_runtime

    _, tco_loss = upstream()
    inputs = {}
    for name, value in data["kwargs"].items():
        if name == "predictions":
            inputs[name] = {key: item.to("cuda").detach().requires_grad_(True)
                            if key in LEAVES else item for key, item in value.items()}
        else:
            inputs[name] = value.to("cuda") if isinstance(value, torch.Tensor) else value
    leaves = inputs["predictions"]
    depth_input = leaves["depth"].detach()
    if depth_input.ndim == 5:
        depth_input = depth_input.squeeze(-1)
    if depth_input.ndim != 4 or depth_input.shape[0] != 1:
        raise ValueError("Expected one scene of model prediction leaves")
    rng = torch.Generator(device="cuda").set_state(data["rng"])
    perm = torch.randperm(depth_input.shape[1], device="cuda", generator=rng)
    post_perm_state = rng.get_state()
    stats, calls = {}, []
    native_raster = gsplat.rasterization_2dgs

    def raster(**kwargs):
        result = native_raster(**kwargs)
        mode = kwargs["render_mode"]
        calls.append(dict(mode=mode, height=kwargs["height"], width=kwargs["width"],
                          views=kwargs["viewmats"].shape[-3], gaussians=kwargs["means"].shape[-2]))
        with torch.no_grad():
            if mode == "ED":
                target = depth_input[0, perm]
                visible = result[0][..., 0] <= target + target.mean() * 0.0001
                eps = 1e-6 if depth_input.dtype == torch.float32 else 1e-4
                alpha = result[1][..., 0]
                stats.update(visibility=visible.cpu(), alpha_denom=((alpha * visible.float()).sum() + eps).cpu(),
                             expected_depth=result[0].detach().cpu(), alpha=alpha.detach().cpu())
            elif mode == "RGB":
                stats["rgb"] = result[0].detach().cpu()
        if negative and mode == "ED":
            result = (result[0], result[1].detach(), *result[2:])
        return result

    with patched_runtime(policy), patch.object(gsplat, "rasterization_2dgs", raster):
        original_multiview = tco_loss.depth_multiview_consistency_energy
        multiview = []

        def loss_tap(*args, **kwargs):
            value = original_multiview(*args, **kwargs)
            multiview.append(value)
            return value

        with patch.object(tco_loss, "depth_multiview_consistency_energy", loss_tap):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.cuda.set_rng_state(data["rng"])
                with torch.autocast("cuda", enabled=data["autocast"], dtype=getattr(torch, data["autocast_dtype"])):
                    loss, terms = tco_loss.compute_tco_losses(**inputs)
                if not torch.equal(torch.cuda.get_rng_state(), post_perm_state):
                    raise ValueError("Renderer qualification found a changed RNG draw")
                if len(multiview) != 1 or [call["mode"] for call in calls] != ["ED", "RGB"]:
                    raise ValueError("Expected exactly one ED/RGB pair for one TCO scene")
                args = tuple(leaves[name] for name in LEAVES)
                mv_grads = torch.autograd.grad(multiview[0], args, allow_unused=True, retain_graph=True)
                total_grads = torch.autograd.grad(loss, args, allow_unused=True)
    cpu = lambda value: None if value is None else value.detach().cpu()
    return dict(values=dict(loss=cpu(loss), **{name: cpu(value) for name, value in terms.items()}),
                mv_gradients=dict(zip(LEAVES, map(cpu, mv_grads))),
                total_gradients=dict(zip(LEAVES, map(cpu, total_grads))), calls=calls,
                permutation=perm.tolist(), **stats)


def compare_render(actual, expected):
    import torch

    exact = (actual["calls"] == expected["calls"] and actual["permutation"] == expected["permutation"]
             and torch.equal(actual["visibility"], expected["visibility"]))
    counts = dict(visible=int(actual["visibility"].sum()), total=actual["visibility"].numel())
    result = dict(exact_discrete=exact, counts=counts)
    for field in ("values", "mv_gradients", "total_gradients"):
        tolerance = FIXED_RTOL if field.endswith("_gradients") else OUTPUT_RTOL
        result[field] = {name: compare_tensor(actual[field][name], reference, tolerance)
                         for name, reference in expected[field].items()}
    result["outputs"] = {name: compare_tensor(actual[name], expected[name], OUTPUT_RTOL)
                         for name in ("alpha_denom", "expected_depth", "alpha", "rgb")}
    result["passed"] = (exact and counts["visible"] > 0 and all(row["passed"] for field in
                        ("values", "mv_gradients", "total_gradients", "outputs") for row in result[field].values()))
    return result


def renderer_fixture(capture=None):
    """Check repeated native CUDA arithmetic, skip-normal effects and a detached-alpha negative control."""
    import torch

    def qualify(data):
        expected = render_replay(data, "native")
        repeats = [compare_render(render_replay(data, "native"), expected) for _ in range(2)]
        negative = compare_render(render_replay(data, "native", negative=True), expected)
        detected = (negative["exact_discrete"] and all(row["passed"] for row in negative["values"].values())
                    and any(not row["passed"] for row in negative["mv_gradients"].values()))
        policies = {policy: compare_render(render_replay(data, policy), expected) for policy in ("camera", "n", "g", "ng")}
        return dict(native_repeats=repeats, native_repeats_pass=all(row["passed"] for row in repeats),
                    negative=negative, negative_detected=detected, policies=policies, calls=expected["calls"],
                    permutation=expected["permutation"],
                    loss_values={name: float(value) for name, value in expected["values"].items()},
                    alpha_denom=float(expected["alpha_denom"]))

    generator = torch.Generator(device="cuda").manual_seed(20261125)
    synthetic = qualify(dict(kwargs=synthetic_inputs(), rng=generator.get_state(), autocast=False,
                             autocast_dtype="float16"))
    real = []
    if capture is not None:
        capture = Path(capture)
        paths = [capture, capture.with_name("renderer_inputs_1.pt")]
        for step, path in enumerate(paths):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            real.append(dict(step=step, capture_sha256=sha(path), **qualify(payload)))
    checks = [synthetic, *real]
    # The renderer itself changes only for the N switch. DPT fallback uses the NG renderer.
    policy_map = {"native": "camera", "camera": "camera", "n": "n", "g": "g", "ng": "ng",
                  "ngd": "ng", "camera_d": "camera"}
    return dict(synthetic=synthetic, real=real, qualified={
        policy: len(real) == 2 and all(row["native_repeats_pass"] and row["negative_detected"]
                                    and row["policies"][render_policy]["passed"] for row in checks)
        for policy, render_policy in policy_map.items()})


def completed_job(identifier):
    """Find one successful attempt and verify its declaration against its terminal receipt."""
    matches = []
    for terminal_path in sorted((PRIVATE / "launches" / identifier).glob("attempt*/terminal.json")):
        terminal = read(terminal_path)
        if terminal["verdict"] != "completed":
            continue
        case = PRIVATE / "cases" / identifier / terminal_path.parent.name / "case"
        receipt = case / "complete.json"
        if sha(receipt) != terminal["complete_sha256"]:
            raise ValueError(f"Receipt bytes changed for {identifier}")
        result = read(receipt)
        if result["id"] != identifier or result["rental_compute_usd"] != 0:
            raise ValueError(f"Invalid receipt for {identifier}")
        matches.append((case, result, terminal["complete_sha256"]))
    if len(matches) > 1:
        raise ValueError(f"Multiple completed receipts for {identifier}")
    return matches[0] if matches else None


def compare_npz(actual, expected, *, families=()):
    import numpy as np

    groups = {}
    with np.load(actual, allow_pickle=False) as left, np.load(expected, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError(f"Tensor inventories differ in {actual.name}: {actual.parent}")
        for key in sorted(right.files):
            a, b = left[key].astype(np.float64), right[key].astype(np.float64)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Invalid tensor {key} in {actual.name}")
            pieces = key.split("/")
            group = "/".join(pieces[:2]) if families else "prediction"
            delta, ref = a - b, max(1e-12, float(np.linalg.norm(b.reshape(-1))))
            relative = float(np.linalg.norm(delta.reshape(-1))) / ref
            absolute = float(np.abs(delta).max())
            row = groups.setdefault(group, dict(count=0, max_abs=0., max_relative_l2=0., worst_tensor=None))
            row["count"] += 1
            row["max_abs"] = max(row["max_abs"], absolute)
            if row["worst_tensor"] is None or relative > row["max_relative_l2"]:
                row.update(max_relative_l2=relative, worst_tensor=key)
    if families:
        required = {f"{kind}/{index}" for kind in families for index in (1, 2)}
        if set(groups) != required or any(row["count"] != 384 for row in groups.values()):
            raise ValueError(f"Unexpected LoRA receipt groups in {actual}: {list(groups)}")
    return groups


def compare_factors(left, right, fixed):
    families = ("grad", "preclip", "parameter")
    if not fixed:
        families += ("exp_avg", "exp_avg_sq", "adam_step")
    fields = compare_npz(left / "lora_audit.npz", right / "lora_audit.npz", families=families)
    for key, row in fields.items():
        limit = 0. if key.startswith("adam_step/") else FIXED_RTOL
        row["passed"] = row["max_relative_l2"] <= limit
    return dict(steps={
        str(index): dict(passed=all(row["passed"] for key, row in fields.items() if key.endswith(f"/{index}")))
        for index in (1, 2)}, fields=fields)


def same_audit(actual, expected, fields):
    return all(field in actual and field in expected and actual[field] == expected[field] for field in fields)


def compare_run(actual, expected, *, fixed=False, matched=False):
    left, observed, _ = actual
    right, reference, _ = expected
    common = ("frozen_start_sha256", "frozen_end_sha256", "adapter_initial", "render_orders")
    audit_l, audit_r = observed["audit"], reference["audit"]
    valid = (observed["input_sha256"] == reference["input_sha256"]
             and observed["updates"] == reference["updates"] == 2
             and observed["lora_tensors"] == reference["lora_tensors"] == 384
             and observed["native_depth_batches"] == reference["native_depth_batches"] == [[5]]
             and same_audit(audit_l, audit_r, common)
             and all(row["frozen_start_sha256"] == row["frozen_end_sha256"] for row in (audit_l, audit_r)))
    factors = compare_factors(left, right, fixed)
    result = dict(common=valid, factors=factors)
    if fixed:
        forward = same_audit(audit_l, audit_r, ("initial_prediction", "taps", "update_predictions"))
        predictions = compare_npz(left / "prediction.npz", right / "prediction.npz")["prediction"]
        result.update(forward_exact=forward, predictions=predictions)
        result["passed"] = (valid and forward and predictions["max_relative_l2"] <= OUTPUT_RTOL
                            and all(row["passed"] for row in factors["steps"].values()))
    else:
        predictions_l, predictions_r = audit_l["update_predictions"], audit_r["update_predictions"]
        same_inputs = [a == b for a, b in zip(predictions_l, predictions_r)]
        result.update(forward_exact_by_update=same_inputs, losses=dict(candidate=observed["losses"],
                                                                      native=reference["losses"]))
        # The second comparison is a matched derivative only when it starts at the reference state.
        updates = (1, 2) if matched else (1,)
        restored = same_audit(audit_l, audit_r, ("matched_state_sha256", "matched_restored")) if matched else True
        result["matched_restore_valid"] = restored if matched else None
        result["passed"] = (valid and len(predictions_l) == len(predictions_r) == 2
                            and restored
                            and all(same_inputs[index - 1] for index in updates)
                            and all(factors["steps"][str(index)]["passed"] for index in updates))
    return result


def compare_identity(actual, expected):
    _, observed, _ = actual
    _, reference, _ = expected
    return dict(passed=observed["updates"] == reference["updates"] == 0
                and observed["native_depth_batches"] == reference["native_depth_batches"] == [[5]]
                and observed["input_sha256"] == reference["input_sha256"]
                and observed["output_sha256"] == reference["output_sha256"])


def compare_leak(actual, expected):
    _, observed, _ = actual
    _, reference, _ = expected
    return dict(passed=observed["input_sha256"] == reference["input_sha256"]
                and observed["reference_sha256"] != reference["reference_sha256"]
                and observed["output_sha256"] == reference["output_sha256"]
                and same_audit(observed["audit"], reference["audit"], ("adapter_initial", "adapter_final")))


def qualify(include_fallback=False):
    """Build the release authority from local receipts and full tensor comparisons."""
    from tco_render_lifetime_protocol import FALLBACK, INITIAL, MAIN, verify

    verify("initial")
    jobs = {}

    def get(identifier):
        if identifier not in jobs:
            jobs[identifier] = completed_job(identifier)
        return jobs[identifier]

    cpu = get("cpu_fixture")
    renderer = get("renderer_replay")
    capture = get("renderer_capture")
    native = get("real_native")
    native_repeat = get("real_native_2")
    native_agreement = compare_run(native_repeat, native) if native_repeat and native else dict(passed=False)
    capture_hashes = []
    if capture is not None and renderer is not None:
        capture_hashes = [sha(capture[0] / name) for name in ("renderer_inputs.pt", "renderer_inputs_1.pt")]
        if capture_hashes != [row["capture_sha256"] for row in renderer[1]["renderer"]["real"]]:
            raise ValueError("Renderer replay and captured loss arguments disagree")
    fixtures = dict(cpu=dict(passed=bool(cpu and cpu[1]["fixture"]["passed"])),
                    renderer=dict(real_captures=capture_hashes, qualified=renderer[1]["renderer"]["qualified"]
                                  if renderer else {}), native_repeat=native_agreement)
    fixed_native = get("fixed_native")
    matched_native = get("matched_native")
    identity_native = get("identity_native")
    results = {}
    for policy in (*MAIN, *(FALLBACK if include_fallback else ())):
        combination = dict(fixed=(get(f"fixed_{policy}"), fixed_native),
                           matched=(get(f"matched_{policy}"), matched_native),
                           real=(get(f"real_{policy}"), native),
                           identity=(get(f"identity_{policy}"), identity_native))
        if policy != "native":
            combination["leak"] = (get(f"leak_{policy}"), get(f"fixed_{policy}"))
        checks = dict(cpu=fixtures["cpu"], renderer=dict(passed=fixtures["renderer"]["qualified"].get(policy, False)),
                      native_repeat=native_agreement)
        for kind, (left, right) in combination.items():
            if left is None or right is None:
                checks[kind] = dict(passed=False, missing=True)
            elif kind == "identity":
                checks[kind] = compare_identity(left, right)
            elif kind == "leak":
                checks[kind] = compare_leak(left, right)
            else:
                checks[kind] = compare_run(left, right, fixed=kind == "fixed", matched=kind == "matched")
        reasons = [name for name, check in checks.items() if not check["passed"]]
        results[policy] = dict(qualified=not reasons, reasons=reasons, **checks)

    return dict(qualified={policy: row["qualified"] for policy, row in results.items()}, fixtures=fixtures,
                policies=results, jobs={identifier: job[2] for identifier, job in sorted(jobs.items()) if job},
                initial_sha256=sha(INITIAL, text=True), rental_compute_usd=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cpu", "qualify"))
    parser.add_argument("--include-fallback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "cpu":
        import torch

        torch.set_num_threads(1)
        record = cpu_fixture()
        print(json.dumps(record, indent=2, allow_nan=False))
        if not record["passed"]:
            raise SystemExit(1)
    else:
        record = qualify(args.include_fallback)
        target = args.output or PUBLIC / "qualification.json"
        if args.include_fallback and target == PUBLIC / "qualification.json" and target.exists():
            archive = PUBLIC / "qualification_primary.json"
            if archive.exists():
                raise FileExistsError(f"Preliminary qualification is already archived at {archive}")
            pending = PUBLIC / "qualification_with_fallback.json"
            write_new(pending, record)
            shutil.copyfile(target, archive)
            pending.replace(target)
        else:
            write_new(target, record)
        print(json.dumps(record["qualified"], sort_keys=True))


if __name__ == "__main__":
    main()
