"""Camera and within-update transformer activation policies for five-view TCO."""

from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from hashlib import sha256
import inspect
from pathlib import Path
import sys
import textwrap
from types import FunctionType, MethodType
import weakref

import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from tco_courtyard_camera import install_camera


SITES = tuple(f"{kind}:{i}" for i in range(24) for kind in ("frame", "global"))
MATMUL = {"aten.mm.default", "aten.bmm.default", "aten.addmm.default"}
SAMPLES = {0, 1, 19, 39}


def _is_attention(name):
    return "scaled_dot_product" in name and "backward" not in name or (
        name.startswith("aten._flash_attention_forward.") or name.startswith("aten._efficient_attention_forward.")
    )


def _matches(name, operation):
    if operation == "attention":
        return _is_attention(name)
    if operation == "matmul":
        return name in MATMUL
    return name == operation


def _compile(native, source):
    local = {}
    exec(compile(source, native.__code__.co_filename + " (five-view overlay)", "exec"),
         native.__globals__, local)
    return local[native.__name__]


def _digest(source):
    return sha256(source.encode("utf-8")).hexdigest()


def _storage_key(value):
    storage = value.untyped_storage()
    return (str(value.device), storage.data_ptr(), storage.nbytes())


class PolicyAudit:
    """Keep scalar observations and weak references only. Clean cases do no profiling."""

    def __init__(self, camera, retention, sites, operation, diagnostic, update_index):
        self.config = dict(camera=camera, retention=retention, sites=list(sites), operation=operation)
        self.diagnostic = diagnostic
        self.update_index = update_index
        self.source_hashes = {}
        self.counts = Counter()
        self.timings = {}
        self.operations = {}
        self.storages = []
        self.caches = []
        self.refinements = []
        self.camera_vjps = {}
        self._pending_events = []
        self._prior_cache_entries = []
        self._last_observed_update = None

    def sample(self):
        if not self.diagnostic:
            return None
        step = self.update_index() if self.update_index is not None else 0
        if step != self._last_observed_update:
            prior = [pair for pair in self._prior_cache_entries if pair[0] < step and pair[1]() is not None]
            self.counts["cached_wrappers_live_at_next_observed_update"] += len(prior)
            self._prior_cache_entries = [pair for pair in self._prior_cache_entries if pair[0] >= step]
            self._last_observed_update = step
        return step if self.update_index is None or step in SAMPLES else None

    @contextmanager
    def time(self, label, device):
        if device.type != "cuda":
            yield
            return
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        try:
            yield
        finally:
            end.record()
            self._pending_events.append((label, begin, end))

    @contextmanager
    def saved_storage(self, label, device):
        unique = set()
        count = 0

        def pack(value):
            nonlocal count
            count += 1
            unique.add(_storage_key(value))
            return value.detach()

        start = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda v: v):
                yield
        finally:
            self.storages.append(dict(label=label, saved_tensor_calls=count,
                                      distinct_storages=len(unique), bytes_including_alias_bases=sum(k[2] for k in unique),
                                      allocated_before=start,
                                      allocated_after=torch.cuda.memory_allocated(device) if start is not None else None))

    def report(self):
        if self._pending_events:
            torch.cuda.synchronize()
            for label, begin, end in self._pending_events:
                self.timings[label] = self.timings.get(label, 0.) + begin.elapsed_time(end)
            self._pending_events.clear()
        return {**self.config, "diagnostic": self.diagnostic, "source_hashes": self.source_hashes,
                "counts": dict(self.counts), "cuda_event_ms": self.timings, "ops": self.operations,
                "saved_storage": self.storages, "selective_cache": self.caches, "camera_graphs": self.refinements,
                "camera_token_vjp": self.camera_vjps}


def _camera_token_capture(head, audit, sink):
    """Capture only the camera branch's VJP before LayerNorm, excluding DPT's use of the shared tap."""
    if audit.update_index is None:
        raise ValueError("Camera-token capture requires an optimizer update index")
    def observe(module, inputs):
        value, = inputs
        if not torch.is_grad_enabled() or not value.requires_grad:
            return
        update = audit.update_index() + 1
        key = str(update)
        if key in audit.camera_vjps:  # one ordinary forward even under non-reentrant replay
            return
        cpu_x = value.detach().to(device="cpu", copy=True)
        def digest(tensor):
            return sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        entry = audit.camera_vjps[key] = dict(shape=list(cpu_x.shape), dtype=str(cpu_x.dtype),
                                             token_bytes_sha256=digest(cpu_x), gradient_received=False)
        def record(gradient):
            if entry["gradient_received"]:
                raise ValueError("Camera tokens received multiple backward calls in one matched update")
            cpu_g = gradient.detach().to(device="cpu", copy=True)
            entry.update(gradient_received=True, gradient_bytes_sha256=digest(cpu_g),
                         gradient_l2=float(cpu_g.double().norm()), finite=bool(torch.isfinite(cpu_g).all()))
            sink(update, cpu_x, cpu_g)
        value.register_hook(record)
    head.token_norm.register_forward_pre_hook(observe)


def _last_camera(head, audit, detach_final=False):
    """Return the opt-in final-output caller. Ordinary head calls keep the native trunk."""
    ordinary = head.trunk_fn
    native = ordinary.__func__
    source = textwrap.dedent(inspect.getsource(native))
    begin = "    for _ in range(num_iterations):\n"
    finish = "    return pred_pose_enc_list"
    if source.count(begin) != 1 or source.count(finish) != 1 or "pred_pose_enc = pred_pose_enc.detach()" not in source:
        raise ValueError("Unrecognized pinned camera refinement")
    before, body = source.split(begin)
    loop, after = body.split(finish)
    fault = "            if iteration == num_iterations - 1:\n                pose_tokens = pose_tokens.detach()\n" if detach_final else ""
    # Use the original arithmetic and raw recurrent state in all iterations.
    source = (before + "    for iteration in range(num_iterations):\n"
              + "        with torch.set_grad_enabled(iteration == num_iterations - 1):\n" + fault
              + textwrap.indent(loop, "    ") + finish + after)
    specialized = MethodType(_compile(native, source), head)
    audit.source_hashes["last_refinement"] = _digest(source)
    last_requested = ContextVar("tco_final_camera", default=False)

    def trunk(tokens, num_iterations):
        if not (last_requested.get() and torch.is_grad_enabled() and tokens.requires_grad):
            return ordinary(tokens, num_iterations)
        step = audit.sample()
        label = f"{step}/camera.last"
        with (audit.saved_storage(label, tokens.device) if step is not None else nullcontext()):
            outputs = specialized(tokens, num_iterations)
        if step is not None:
            audit.refinements.append(dict(update_index=step, iterations=num_iterations,
                                          output_requires_grad=[v.requires_grad for v in outputs],
                                          input_requires_grad=tokens.requires_grad))
        return outputs

    head.trunk_fn = trunk

    def call(taps, num_iterations=4):
        if head.training or not torch.is_grad_enabled() or any(p.requires_grad for p in head.parameters()):
            return head(taps, num_iterations=num_iterations)
        flag = last_requested.set(True)
        try:
            return head(taps, num_iterations=num_iterations)
        finally:
            last_requested.reset(flag)

    return call


def _adaptation_camera(model, audit, forward_source):
    native = model.forward.__func__
    # The historical launcher compiles its in-memory control overlay at a synthetic
    # filename. It supplies the pinned TCO source captured before that scope.
    source = textwrap.dedent(forward_source if forward_source is not None else inspect.getsource(native))
    start, stop = "for step in range(tco_steps):", "# Restore requires_grad flags"
    old = "pose_enc_list = self.camera_head(aggregated_tokens_list)"
    if source.count(start) != 1 or source.count(stop) != 1:
        raise ValueError("Unrecognized pinned adaptation loop")
    prefix, tail = source.split(start)
    loop, suffix = tail.split(stop)
    if loop.count(old) != 1 or "pose_enc_list[-1]" not in loop:
        raise ValueError("Expected a final-pose adaptation consumer")
    loop = loop.replace(old, "pose_enc_list = self._five_view_final_camera(aggregated_tokens_list)")
    source = prefix + start + loop + stop + suffix
    model._five_view_final_camera = _last_camera(model.camera_head, audit)
    model.forward = MethodType(_compile(native, source), model)
    audit.source_hashes["adaptation_forward"] = _digest(source)


class _OperatorTrace(TorchDispatchMode):
    """Below the selective cache mode, so actual executions exclude cache hits."""

    def __init__(self, audit, label, device):
        super().__init__()
        self.audit, self.label, self.device = audit, label, device

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name not in MATMUL and not _is_attention(name) and name != self.audit.config["operation"]:
            return func(*args, **(kwargs or {}))
        key = f"{self.label}/{name}"
        with self.audit.time(key, self.device):
            output = func(*args, **(kwargs or {}))
        row = self.audit.operations.setdefault(key, dict(calls=0, signatures={}))
        row["calls"] += 1
        signature = "/".join(f"{list(x.shape)}:{x.dtype}" for x in tree_leaves(output) if isinstance(x, torch.Tensor))
        row["signatures"][signature] = row["signatures"].get(signature, 0) + 1
        return output


def _cache_snapshot(mode):
    storage = mode.storage
    counts = {str(op): len(entries) for op, entries in storage.items() if entries}
    wrappers = [v for entries in storage.values() for item in entries for v in tree_leaves(item)]
    keys = {_storage_key(v.val) for v in wrappers if isinstance(v.val, torch.Tensor)}
    return counts, wrappers, sum(k[2] for k in keys)


def _contexts(audit, site, step, device, selective, save):
    operation = audit.config["operation"]

    def policy(ctx, op, *args, **kwargs):
        if save and _matches(str(op), operation):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    native_modes = create_selective_checkpoint_contexts(policy) if selective else (nullcontext(), nullcontext())
    if step is None:
        return native_modes
    row = dict(update_index=step, site=site)
    if selective:
        audit.caches.append(row)

    @contextmanager
    def tracked(mode, replay):
        label = f"{step}/{site}/" + ("recompute" if replay else "forward")
        audit.counts[label] += 1
        try:
            with audit.time(label, device), _OperatorTrace(audit, label, device), mode:
                yield
        finally:
            if selective:
                counts, wrappers, size = _cache_snapshot(mode)
                if replay:
                    # Native early-stop can leave unconsumed entries. Only popped
                    # entries saved an actual operation during this backward.
                    row["remaining"] = counts
                    row["hits"] = {op: n - counts.get(op, 0) for op, n in row["saved"].items()}
                else:
                    row.update(saved=counts, bytes_including_alias_bases=size)
                    audit._prior_cache_entries.extend((step, weakref.ref(v)) for v in wrappers)

    return tracked(native_modes[0], False), tracked(native_modes[1], True)


def _install_retention(model, audit):
    retention = audit.config["retention"]
    if retention == "original" and not audit.diagnostic:
        return
    a = model.aggregator
    if a.use_reentrant or a.aa_block_size != 1 or len(a.frame_blocks) != 24 or len(a.global_blocks) != 24:
        raise ValueError("Expected 24 paired non-reentrant checkpoints")
    selected = set(audit.config["sites"])
    sites = {id(block): f"{kind}:{i}" for kind in ("frame", "global")
             for i, block in enumerate(getattr(a, kind + "_blocks"))}

    def dispatch(block, tokens, pos, use_reentrant):
        if use_reentrant or id(block) not in sites:
            raise ValueError("Unexpected checkpoint site or mode")
        site = sites[id(block)]
        step = audit.sample()
        if retention == "block" and site in selected:
            if step is None:
                return block(tokens, pos=pos)
            label = f"{step}/{site}/retained_block"
            audit.counts[label] += 1
            with audit.time(label, tokens.device), audit.saved_storage(label, tokens.device), _OperatorTrace(audit, label, tokens.device):
                return block(tokens, pos=pos)
        save = retention == "op" and site in selected
        selective = save or retention == "zero"
        if not selective and step is None:
            return checkpoint(block, tokens, pos, use_reentrant=False)
        def contexts():
            return _contexts(audit, site, step, tokens.device, selective, save)
        return checkpoint(block, tokens, pos, use_reentrant=False, context_fn=contexts)

    for kind in ("frame", "global"):
        method = f"_process_{kind}_attention"
        native = getattr(model, method).__func__
        if native.__globals__.get("checkpoint") is not checkpoint:
            raise ValueError("Expected the original transformer checkpoint binding")
        local = dict(native.__globals__, checkpoint=dispatch)
        fn = FunctionType(native.__code__, local, native.__name__, native.__defaults__, native.__closure__)
        setattr(model, method, MethodType(fn, model))
        audit.source_hashes[method] = _digest(textwrap.dedent(inspect.getsource(native)))


def install_policy(model, camera="native", retention="original", sites=(), operation="attention",
                   diagnostic=False, update_index=None, forward_source=None, camera_token_sink=None):
    """Install on a fresh `four` model after base setup. Get scalar receipts via report()."""
    if camera not in {"native", "checkpoint", "last"} or retention not in {"original", "zero", "op", "block"}:
        raise ValueError("Unknown policy variant")
    sites = tuple(sites)
    if len(set(sites)) != len(sites) or not set(sites).issubset(SITES):
        raise ValueError(f"Unknown or duplicate transformer sites: {sites}")
    if bool(sites) != (retention in {"op", "block"}):
        raise ValueError("Only op/block retention takes a nonempty site list")
    if operation not in {"attention", "matmul"} and not (operation.startswith("aten.") and operation.count(".") == 2):
        raise ValueError(f"Unknown ATen operation: {operation}")
    audit = PolicyAudit(camera, retention, sites, operation, diagnostic, update_index)
    if camera == "last":
        _adaptation_camera(model, audit, forward_source)
    if camera == "checkpoint":
        install_camera(model, True)
    if camera_token_sink is not None:
        _camera_token_capture(model.camera_head, audit, camera_token_sink)
    _install_retention(model, audit)
    return audit


def _error(actual, reference, r):
    if actual is None or reference is None or actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError("Camera output or VJP presence, shape or dtype differs")
    a, b = actual.detach().double(), reference.detach().double()
    delta, norm = float((a - b).norm()), float(b.norm())
    ok = torch.isfinite(a).all().item() and torch.isfinite(b).all().item() and delta <= 1e-6 + r * norm
    if not ok:
        raise ValueError(f"Camera output or signed VJP exceeded tolerance: {delta} vs {1e-6 + r * norm}")
    return dict(delta_l2=delta, reference_l2=norm, max_abs=float((a - b).abs().max()), pass_tolerance=True)


def camera_fixture(device="cpu", amp=False):
    """Native vs last-only vs whole checkpoint; ordinary intermediate supervision stays differentiable."""
    from copy import deepcopy
    from types import SimpleNamespace

    vggt = Path(__file__).resolve().parents[1] / "third_party/TCO/base_models/vggt"
    sys.path.insert(0, str(vggt))
    try:
        from vggt.heads.camera_head import CameraHead
    finally:
        sys.path.remove(str(vggt))
    device = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261126)
        source = CameraHead(dim_in=32, trunk_depth=4, num_heads=4).eval().requires_grad_(False)
        initial = torch.randn(1, 5, 3, 32)
        cotangents = [torch.randn(1, 5, 9) for _ in range(4)]

    def evaluate(kind, iterations, auxiliary=False, inference=False):
        head = deepcopy(source).to(device)
        audit = PolicyAudit(kind, "original", (), "attention", False, lambda: 0)
        call = head
        if kind in {"last", "detach_final"}:
            specialized = _last_camera(head, audit, detach_final=kind == "detach_final")
            if not auxiliary:
                call = specialized
        elif kind == "checkpoint":
            install_camera(SimpleNamespace(camera_head=head), True)
        capture = []
        _camera_token_capture(head, audit, lambda n, x, g: capture.append((n, x, g)))
        x = initial.to(device).clone().requires_grad_()
        with torch.set_grad_enabled(not inference), torch.autocast(device.type, enabled=amp):
            outputs = call([x], num_iterations=iterations)
        connects = [value.requires_grad for value in outputs]
        if inference or not outputs[-1].requires_grad:
            gradient = None
        else:
            indices = range(iterations) if auxiliary else [iterations - 1]
            y = tuple(outputs[i] for i in indices)
            q = tuple(cotangents[i].to(device=device, dtype=output.dtype) for i, output in zip(indices, y))
            gradient, = torch.autograd.grad(y, x, grad_outputs=q)
            if not bool(gradient[:, :, 0].any()) or bool(gradient[:, :, 1:].any()):
                raise ValueError("Camera VJP must affect only the camera token")
            if len(capture) != 1 or capture[0][0] != 1:
                raise ValueError("Expected one camera branch VJP per optimizer update")
            torch.testing.assert_close(capture[0][1], x[:, :, 0].detach().cpu(), atol=0, rtol=0)
            torch.testing.assert_close(capture[0][2], gradient[:, :, 0].detach().cpu(), atol=0, rtol=0)
        if any(p.grad is not None for p in head.parameters()):
            raise ValueError("Unexpected gradient on a frozen camera parameter")
        return [v.detach() for v in outputs], gradient, connects

    cases = {}
    for iters, aux, inference in ((4, False, False), (1, False, False), (4, True, False), (4, False, True)):
        values, gradient, _ = evaluate("native", iters, aux, inference)
        for kind in ("last", "checkpoint"):
            result, g, graph = evaluate(kind, iters, aux, inference)
            expected = [False] * iters if inference else (
                [False] * (iters - 1) + [True] if kind == "last" and not aux else [True] * iters
            )
            if graph != expected or len(result) != iters:
                raise ValueError("Camera graph history differs from expected consumer")
            record = dict(values=[_error(x, y, 1e-5) for x, y in zip(result, values)], gradient_paths=graph)
            if not inference:
                record["signed_token_vjp"] = _error(g, gradient, 1e-4)
            cases[f"{kind}/{iters}/aux{int(aux)}/inference{int(inference)}"] = record
    _, broken, _ = evaluate("detach_final", 4)
    if broken is not None:
        raise ValueError("Detached last-refinement token negative control escaped")
    return dict(device=str(device), torch=torch.__version__, amp=amp, cases=cases,
                detached_last_token_detected=True)


def retention_fixture():
    """Two optimizer updates must build fresh same-update SAC values and replay both VJPs."""
    from types import SimpleNamespace

    class TinyBlock(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.clone())

        def forward(self, tokens, pos=None):
            return (tokens @ self.weight).sin()

    class TinyModel:
        def _process_frame_attention(self, tokens):
            return checkpoint(self.aggregator.frame_blocks[0], tokens, None, use_reentrant=False)

        def _process_global_attention(self, tokens):
            return checkpoint(self.aggregator.global_blocks[0], tokens, None, use_reentrant=False)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261126)
        weight = torch.randn(4, 4)
        tokens = torch.randn(3, 4)
        cot = torch.randn(3, 4)

    def run(retention):
        model = TinyModel()
        frame = [TinyBlock(weight) for _ in range(24)]
        global_blocks = [TinyBlock(weight.T.contiguous()) for _ in range(24)]
        model.aggregator = SimpleNamespace(frame_blocks=frame, global_blocks=global_blocks,
                                           use_reentrant=False, aa_block_size=1)
        step = 0
        sites = ["frame:0"] if retention in {"op", "block"} else []
        audit = install_policy(model, retention=retention, operation="aten.mm.default", sites=sites,
                               diagnostic=True, update_index=lambda: step)
        parameters = [frame[0].weight, global_blocks[0].weight]
        optim = torch.optim.Adam(parameters, lr=0.01)
        records = []
        for step in range(2):
            optim.zero_grad(set_to_none=True)
            x = tokens.clone().requires_grad_()
            y = model._process_global_attention(model._process_frame_attention(x))
            y.backward(cot)
            records.append([value.detach().clone() for value in (y, x.grad, *(p.grad for p in parameters))])
            optim.step()
            records[-1].extend(p.detach().clone() for p in parameters)
        return records, audit.report()

    native, _ = run("original")
    if torch.equal(native[0][0], native[1][0]):
        raise ValueError("Fixture's Adam update did not change the next forward")
    evidence = {}
    for name in ("zero", "op", "block"):
        actual, report = run(name)
        for ref_row, target_row in zip(native, actual):
            for reference, result in zip(ref_row, target_row):
                torch.testing.assert_close(result, reference, atol=0, rtol=0)
        hits = [row.get("hits", {}).get("aten.mm.default", 0) for row in report["selective_cache"]]
        if name == "op" and hits != [1, 1] or name == "zero" and any(hits):
            raise ValueError(f"Incorrect number of per-update cached GEMM hits: {hits}")
        for step in range(2):
            prefix = f"{step}/frame:0"
            expected = f"{prefix}/retained_block" if name == "block" else f"{prefix}/recompute"
            if report["counts"].get(expected) != 1:
                raise ValueError(f"Unexpected forward/recompute count at {expected}")
        if report["counts"].get("cached_wrappers_live_at_next_observed_update", 0):
            raise ValueError("A cached GEMM entry survived the optimizer update")
        evidence[name] = dict(identical_updates=2, cached_mm_hits=hits,
                              stale_entries=report["counts"]["cached_wrappers_live_at_next_observed_update"])
    return evidence
