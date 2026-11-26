"""Matched-state vector-Jacobian products at the real five-view TCO loss boundary.

All tensor bundles belong in the private case directory. Clean performance runs never
enter this module's instrumentation. The inherited runner owns clipping and Adam.
"""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


LEAVES = ("depth", "depth_conf", "camera_poses", "intrinsics")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, allow_nan=False, indent=2)
        stream.write("\n")


def save(path, data):
    with Path(path).open("xb") as stream:
        torch.save(data, stream)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def cpu_copy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().to(device="cpu", copy=True)
    if isinstance(data, dict):
        return {key: cpu_copy(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(cpu_copy(value) for value in data)
    return data


def tensor_hash(tensor):
    if tensor is None:
        return None
    contiguous = tensor.detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256(repr((str(tensor.dtype), tuple(tensor.shape))).encode())
    digest.update(contiguous.view(torch.uint8).numpy())
    return digest.hexdigest()


def tree_hash(tree):
    """Canonicalize the optimizer's integer keys, group order, and tensor bytes."""
    def encode(value):
        if isinstance(value, torch.Tensor):
            return ["tensor", tensor_hash(value)]
        if isinstance(value, dict):
            return ["dict", [[repr(key), encode(value[key])] for key in sorted(value, key=repr)]]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [encode(item) for item in value]]
        return value

    return hashlib.sha256(json.dumps(encode(tree), allow_nan=False).encode()).hexdigest()


def tensor_meta(tensor):
    if tensor is None:
        return None
    return dict(sha256=tensor_hash(tensor), shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype))


def boundary(y):
    return {name: tensor_meta(y[name]) for name in LEAVES}


def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def set_rng(rng):
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state(rng["cuda"])


def view_draw(rng, count):
    generator = torch.Generator(device="cuda").set_state(rng["cuda"])
    order = torch.randperm(count, device="cuda", generator=generator).tolist()
    return order, generator.get_state()


def check_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def verify_boundary(y, expected):
    for name in LEAVES:
        value = y[name]
        if not value.requires_grad or not value.isfinite().all():
            raise ValueError(f"Inactive or nonfinite model boundary: {name}")
        check_equal(tensor_meta(value), expected[name], f"live Y {name}")


def paired_vectors(y, q):
    if set(q) != set(LEAVES):
        raise ValueError("Missing or misnamed boundary derivative")
    outputs, vectors = [], []
    for key in LEAVES:
        value, derivative = y[key], q[key]
        if derivative is None:
            continue
        if derivative.shape != value.shape or derivative.dtype != value.dtype:
            raise ValueError(f"Shape/dtype mismatch at {key}")
        if not bool(torch.isfinite(derivative).all()):
            raise ValueError(f"Nonfinite derivative at {key}")
        outputs.append(value)
        vectors.append(derivative.to(value.device))
    if not vectors:
        raise ValueError("All four derivatives are absent")
    return tuple(outputs), tuple(vectors)


def backend_settings():
    return dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                sdpa_allowed={name: getattr(torch.backends.cuda, name + "_sdp_enabled")()
                              for name in ("flash", "mem_efficient", "math", "cudnn")
                              if hasattr(torch.backends.cuda, name + "_sdp_enabled")})


class GradientProbe:
    def __init__(self, job, out, source, q_source):
        self.mode = job["gradient_mode"]
        self.out = Path(out)
        self.source = Path(source) if source is not None else None
        self.q_source = Path(q_source) if q_source is not None else self.source
        self.rows = {}
        self.step = 0
        self.pending = None
        self.anchor = None
        self.selected = {}
        self.q = {}
        self.hooks = []
        self.injected = 0
        self.capture = self.mode == "capture"
        self.job = job
        if self.mode not in ("capture", "replay", "negative") or job["steps"] != 2:
            raise ValueError("Expected two captured or two individually restored optimizer updates")
        if not self.capture and self.source is None:
            raise ValueError("Signed gradient replay requires a captured trajectory")

    def snapshot(self):
        return dict(names=self.names, factors=cpu_copy(self.selected),
                    old_grad={name: cpu_copy(p.grad) for name, p in self.selected.items()},
                    optimizer=cpu_copy(self.optimizer.state_dict()), rng=rng_state())

    def preforward(self):
        self.step += 1
        index = self.step
        if index not in (1, 2):
            raise ValueError("Unexpected extra gradient-bearing adaptation forward")
        if self.capture:
            snapshot = self.snapshot()
            self.anchor = {"step": index, "snapshot": snapshot, "state_hash": tree_hash(snapshot)}
            return
        path = self.source / f"anchor_step{index}.pt"
        expected_file = source_digest(self.source, path.name)
        self.anchor = load(path)
        check_equal(self.anchor["step"], index, "anchor update")
        before = self.anchor["snapshot"]
        check_equal(before["names"], self.names, "optimizer parameter order")
        with torch.no_grad():
            for name, parameter in self.selected.items():
                parameter.copy_(before["factors"][name])
                previous = before["old_grad"][name]
                parameter.grad = None if previous is None else previous.to(parameter.device, copy=True)
        self.optimizer.load_state_dict(before["optimizer"])
        set_rng(before["rng"])
        check_equal(tree_hash(self.snapshot()), self.anchor["state_hash"], "restored pre-forward state")
        self.rows[str(index)] = {"anchor_sha256": expected_file, "pre_state_sha256": self.anchor["state_hash"]}

    def bind(self, model):
        setup, aggregate = model.backbone_peft_setup, model.aggregate_with_patch_tokens

        def initialize(*args, **kwargs):
            optimizer, params = setup(*args, **kwargs)
            self.optimizer = optimizer
            by_id = {id(value): name for name, value in model.named_parameters()}
            self.names = [by_id[id(value)] for group in optimizer.param_groups for value in group["params"]]
            self.selected = dict(zip(self.names, params))
            if len(self.selected) != 384 or any(by_id[id(value)] != name for name, value in self.selected.items()):
                raise ValueError("Expected the 384 ordered native LoRA tensors")
            optimizer.register_step_post_hook(lambda *_: self.poststep())
            return optimizer, params

        def forward(*args, **kwargs):
            if kwargs.get("use_checkpoint"):
                self.preforward()
            return aggregate(*args, **kwargs)

        model.backbone_peft_setup = initialize
        model.aggregate_with_patch_tokens = forward

    def hook_loss(self, ordinary, state, audit, args, kwargs):
        from check_tco_render_lifetime import capture_renderer_inputs

        y = {key: kwargs["predictions"][key] for key in LEAVES}
        if state["updates"] + 1 != self.step or self.pending is not None:
            raise ValueError("Loss observed out of order")
        key = str(self.step)
        if self.capture:
            path = self.out / f"renderer_inputs_step{self.step}.pt"
            capture_renderer_inputs(kwargs, path)
            self.anchor["boundary"] = boundary(y)
            verify_boundary(y, self.anchor["boundary"])
            self.anchor["loss_rng"] = rng_state()
            self.anchor["loss_autocast"] = (torch.is_autocast_enabled("cuda"), str(torch.get_autocast_dtype("cuda")))
            self.q = dict.fromkeys(LEAVES)

            def record(name):
                def hook(derivative):
                    if self.q[name] is not None:
                        raise ValueError(f"Repeated derivative callback for {name}")
                    self.q[name] = cpu_copy(derivative)
                    return None
                return hook

            self.hooks = [value.register_hook(record(name)) for name, value in y.items()]
            scalar, terms = ordinary(*args, **kwargs)
            self.anchor.update(post_permutation=rng_state(), loss=cpu_copy(scalar), terms=cpu_copy(terms),
                               permutation=audit["render_orders"][-1])
            self.rows[key] = {"pre_state_sha256": self.anchor["state_hash"]}
        else:
            verify_boundary(y, self.anchor["boundary"])
            check_equal(tree_hash(rng_state()), tree_hash(self.anchor["loss_rng"]), "pre-loss RNG")
            check_equal((torch.is_autocast_enabled("cuda"), str(torch.get_autocast_dtype("cuda"))),
                        self.anchor["loss_autocast"], "loss autocast")
            perm, post_draw = view_draw(self.anchor["loss_rng"], kwargs["images"].shape[1])
            check_equal(perm, self.anchor["permutation"], "replay view order")
            if not torch.equal(post_draw, self.anchor["post_permutation"]["cuda"]):
                raise ValueError("Original renderer consumed more than one permutation")
            set_rng(self.anchor["post_permutation"])
            audit.setdefault("render_orders", []).append(perm)
            from run_tco_one_percent import FIELDS, tensor_hash as inherited_hash
            audit.setdefault("update_predictions", []).append(
                {name: inherited_hash(kwargs["predictions"][name]) for name in FIELDS})
            scalar = self.anchor["loss"].to(y["depth"].device)
            terms = {name: value.to(scalar.device) for name, value in self.anchor["terms"].items()}
            state.setdefault("first_loss", scalar.detach())
            state["last_loss"] = scalar.detach()
            q_file = self.q_source / f"q_step{self.step}.pt"
            self.rows[key]["q_artifact_sha256"] = source_digest(self.q_source, q_file.name)
            bundle = load(q_file)
            verify_q_bundle(bundle, self.anchor)
            self.q = dict(bundle["q"])
            if self.mode == "negative":
                flipped = next((name for name in LEAVES if self.q[name] is not None and self.q[name].any()), None)
                if flipped is None:
                    raise ValueError("No active derivative for signed negative")
                self.q[flipped] = -self.q[flipped]
                self.rows[key]["flipped"] = flipped
        self.pending = scalar
        self.y = y
        return scalar, terms

    def backward(self, ordinary, scalar, *args, **kwargs):
        if scalar is not self.pending:
            return ordinary(scalar, *args, **kwargs)
        if args or kwargs:
            raise ValueError("The pinned loss expects an unscaled scalar backward")
        if self.capture:
            self.anchor["backward_rng"] = rng_state()
            ordinary(scalar)
            for handle in self.hooks:
                handle.remove()
            self.hooks.clear()
            paired_vectors(self.y, self.q)
        else:
            check_equal(tree_hash(rng_state()), tree_hash(self.anchor["backward_rng"]), "pre-backward RNG")
            outputs, gradients = paired_vectors(self.y, self.q)
            torch.autograd.backward(outputs, grad_tensors=gradients)
        self.injected += 1
        self.pending = None
        self.y = {}

    def poststep(self):
        index = self.step
        if self.injected != index:
            raise ValueError("Expected one matched backward and Adam step")
        row = self.rows[str(index)]
        if self.capture:
            self.anchor["gradient_present"] = {name: value is not None for name, value in self.q.items()}
            for name, data in ((f"anchor_step{index}.pt", self.anchor),
                               (f"q_step{index}.pt", make_q_bundle(self.anchor, self.q))):
                save(self.out / name, data)
            row["anchor_sha256"] = file_sha(self.out / f"anchor_step{index}.pt")
        row.update(boundary=self.anchor["boundary"], derivatives={key: tensor_meta(val) for key, val in self.q.items()},
                   source_views=self.anchor["permutation"][:1], target_views=self.anchor["permutation"],
                   pre_forward_rng_sha256=tree_hash(self.anchor["snapshot"]["rng"]),
                   pre_permutation_rng_sha256=tree_hash(self.anchor["loss_rng"]),
                   post_permutation_rng_sha256=tree_hash(self.anchor["post_permutation"]),
                   pre_backward_rng_sha256=tree_hash(self.anchor["backward_rng"]))
        if index == 2:
            self.summary()

    def summary(self):
        artifacts = {path.name: file_sha(path) for path in self.out.glob("*.pt")
                     if path.name.startswith(("anchor_step", "q_step", "renderer_inputs_step"))}
        write_json(self.out / "gradient_summary.json", dict(mode=self.mode, signed_total_loss=True,
                   backend=backend_settings(), matched_updates=self.rows, raw_artifact_sha256=artifacts))


@contextmanager
def gradient_context(lifetime, job, out, source=None, q_source=None):
    mode = job.get("gradient_mode")
    if mode in (None, "control"):
        if mode == "control":
            write_json(Path(out) / "gradient_summary.json", dict(mode="control", fresh_ordinary_trajectory=True,
                       backend=backend_settings()))
        yield
        return
    if job.get("matched_state_from"):
        raise ValueError("Do not combine independent state restoration with signed-gradient replay")
    probe = GradientProbe(job, out, source, q_source)
    setup, factory, backward = lifetime.setup_execution, lifetime.observe_losses, torch.Tensor.backward

    def setup_with_gradients(model, *args, **kwargs):
        installed = setup(model, *args, **kwargs)
        probe.bind(model)
        return installed

    def observe(state, audit, fixed):
        origin, observed = factory(state, audit, fixed)
        if audit is None or fixed:
            raise ValueError("Real-gradient replay requires the two-update native optimizer audit")
        return origin, lambda *args, **kwargs: probe.hook_loss(observed, state, audit, args, kwargs)

    def dispatch(scalar, *args, **kwargs):
        return probe.backward(backward, scalar, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(lifetime, "setup_execution", setup_with_gradients))
        stack.enter_context(patch.object(lifetime, "observe_losses", observe))
        stack.enter_context(patch.object(torch.Tensor, "backward", dispatch))
        yield
        if probe.step != 2 or probe.injected != 2 or probe.pending is not None:
            raise ValueError("Gradient experiment did not finish both matched updates")


def source_digest(folder, filename):
    summary = json.loads((folder / "gradient_summary.json").read_text(encoding="utf-8"))
    actual = file_sha(folder / filename)
    check_equal(actual, summary["raw_artifact_sha256"][filename], "source artifact " + filename)
    return actual


def make_q_bundle(anchor, q):
    return dict(step=anchor["step"], anchor_state_sha256=anchor["state_hash"],
                boundary=anchor["boundary"], q=q, present={name: value is not None for name, value in q.items()})


def verify_q_bundle(bundle, anchor):
    check_equal(bundle["step"], anchor["step"], "q update")
    check_equal(bundle["anchor_state_sha256"], anchor["state_hash"], "q anchor state")
    check_equal(bundle["boundary"], anchor["boundary"], "q boundary")
    check_equal(set(bundle["q"]), set(LEAVES), "q names")
    expected = anchor["gradient_present"]
    check_equal({name: value is not None for name, value in bundle["q"].items()}, expected, "q presence")
    check_equal(bundle["present"], expected, "q declared presence")


def array_stats(actual, reference, rtol=1e-4):
    """Float64 accumulation; zero reference/angle is undefined, not infinity."""
    if actual is None or reference is None:
        return {"present": actual is not None, "reference_present": reference is not None,
                "passed": actual is None and reference is None}
    if isinstance(actual, torch.Tensor):
        if actual.dtype != reference.dtype:
            raise ValueError("Derivative dtype mismatch")
        a, b = actual.detach().cpu().double().numpy(), reference.detach().cpu().double().numpy()
    else:
        if actual.dtype != reference.dtype:
            raise ValueError("Optimizer array dtype mismatch")
        a, b = actual.astype(np.float64), reference.astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Array shape or finiteness mismatch")
    a, b = a.ravel(), b.ravel()
    # Elementwise reductions avoid a threaded BLAS dispatch per small LoRA tensor.
    aa, bb = float(np.sum(a*a)), float(np.sum(b*b))
    diff = a - b
    dd, ab = float(np.sum(diff*diff)), float(np.sum(a*b))
    delta, ref = math.sqrt(dd), math.sqrt(bb)
    return dict(present=True, reference_present=True, passed=delta <= 1e-6 + rtol * ref,
                diff_norm=delta, ref_norm=ref, norm=math.sqrt(aa),
                max_abs=float(np.max(np.abs(diff), initial=0)), relative_l2=delta/ref if ref else None,
                cosine=ab/math.sqrt(aa*bb) if aa and bb else None, dot=ab)


def summarize_differences(rows):
    active = {name: stats for name, stats in rows.items() if "diff_norm" in stats}
    valid = {name: stats for name, stats in active.items() if stats["relative_l2"] is not None}
    worst = max(valid, key=lambda name: valid[name]["relative_l2"], default=None)
    abs_worst = max(active, key=lambda name: active[name]["diff_norm"], default=None)
    delta = math.sqrt(sum(row["diff_norm"]**2 for row in active.values()))
    ref = math.sqrt(sum(row["ref_norm"]**2 for row in active.values()))
    actual = math.sqrt(sum(row["norm"]**2 for row in active.values()))
    return dict(count=len(rows), passed=bool(rows) and all(row["passed"] for row in rows.values()),
                failed=sum(not row["passed"] for row in rows.values()), diff_norm=delta,
                actual_norm=actual, reference_norm=ref, relative_l2=delta/ref if ref else None,
                cosine=sum(row["dot"] for row in active.values())/(actual*ref) if actual and ref else None,
                worst_relative_name=worst, worst_relative=active.get(worst),
                largest_difference_name=abs_worst, largest_difference=active.get(abs_worst))


def compare_q(left, right):
    check_equal(set(left), set(LEAVES), "observed derivative names")
    check_equal(set(right), set(LEAVES), "reference derivative names")
    rows = {name: array_stats(left[name], right[name]) for name in LEAVES}
    return dict(leaves=rows, overall=summarize_differences(rows))


def regenerate(source, out, replica=1):
    """One fresh detached-leaf evaluation for each of the two fixed states."""
    from check_tco_render_lifetime import compare_render, render_replay

    source, out = Path(source), Path(out)
    rows, hashes = {}, {}
    for step in (1, 2):
        state_name, input_name, q_name = (f"anchor_step{step}.pt", f"renderer_inputs_step{step}.pt",
                                          f"q_step{step}.pt")
        source_hashes = {name: source_digest(source, name) for name in (state_name, input_name, q_name)}
        anchor, data, original = (load(source / name) for name in (state_name, input_name, q_name))
        verify_q_bundle(original, anchor)
        check_equal({k: tensor_hash(v) for k, v in data["kwargs"]["predictions"].items() if k in LEAVES},
                    {k: v["sha256"] for k, v in anchor["boundary"].items()}, "detached-leaf boundary")
        check_equal(tensor_hash(data["rng"]), tensor_hash(anchor["loss_rng"]["cuda"]), "detached-leaf RNG")
        actual = render_replay(data, "native")
        check_equal(actual["permutation"], anchor["permutation"], "detached-leaf view order")
        q = actual["total_gradients"]
        bundle = make_q_bundle(anchor, q)
        verify_q_bundle(bundle, anchor)
        q_comparison = compare_q(q, original["q"])
        value_comparisons = {name: array_stats(actual["values"][name], expected, rtol=1e-5)
                             for name, expected in {"loss": anchor["loss"], **anchor["terms"]}.items()}
        row = dict(anchor_sha256=source_hashes[state_name], pre_state_sha256=anchor["state_hash"],
                   source_artifact_sha256=source_hashes, comparisons=q_comparison,
                   loss_comparisons=value_comparisons, visible=int(actual["visibility"].sum()),
                   alpha_denominator=float(actual["alpha_denom"]), permutation=actual["permutation"],
                   renderer_calls=actual["calls"])
        if int(replica) == 1:
            changed = render_replay(data, "native", negative=True)
            negative = compare_render(changed, actual)
            row["detached_alpha_detected"] = (negative["exact_discrete"]
                and all(part["passed"] for part in negative["values"].values())
                and any(not part["passed"] for part in negative["mv_gradients"].values()))
        save(out / q_name, bundle)
        hashes[q_name] = file_sha(out / q_name)
        rows[str(step)] = row
    result = dict(mode="regenerate", replica=replica, backend=backend_settings(), matched_updates=rows,
                  raw_artifact_sha256=hashes)
    write_json(out / "gradient_summary.json", result)
    return result


def compare_q_cases(actual, reference):
    """Compare previously exported native captured or detached-leaf q bundles."""
    left, right = Path(actual), Path(reference)
    result = {}
    for step in (1, 2):
        name = f"q_step{step}.pt"
        source_digest(left, name)
        source_digest(right, name)
        a, b = load(left/name), load(right/name)
        for field in ("step", "boundary", "anchor_state_sha256"):
            check_equal(a[field], b[field], f"paired q {field}")
        result[str(step)] = compare_q(a["q"], b["q"])
    return result


def compare_cases(actual, reference, anchor):
    """Report optimizer discrepancies at each matched pre-update state.

    Historical clipped-gradient keys index completed updates from zero. All
    other inherited value families use indices one and two. Convert explicitly.
    """
    actual, reference, anchor = Path(actual), Path(reference), Path(anchor)
    result = {}
    with np.load(actual / "lora_audit.npz", allow_pickle=False) as a, \
         np.load(reference / "lora_audit.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Gradient/parameter audit inventories differ")
        for step in (1, 2):
            step_file = f"anchor_step{step}.pt"
            source_digest(anchor, step_file)
            before = load(anchor / step_file)["snapshot"]["factors"]
            if len(before) != 384:
                raise ValueError("Unexpected number of anchor factors")
            families = {}
            for family in ("preclip", "grad", "parameter", "exp_avg", "exp_avg_sq", "adam_step", "delta"):
                rows = {}
                prefix = "parameter" if family == "delta" else family
                index = step - 1 if family == "grad" else step
                for factor, initial in before.items():
                    name = f"{prefix}/{index}/{factor}"
                    x, y = a[name], b[name]
                    if family == "delta":
                        origin = initial.double().numpy()
                        x, y = x.astype(np.float64) - origin, y.astype(np.float64) - origin
                    row = array_stats(x, y)
                    if family == "adam_step":
                        row["passed"] = np.array_equal(x, y) and bool(np.all(x == step))
                    rows[factor] = row
                families[family] = summarize_differences(rows)
            result[str(step)] = families
    return result
