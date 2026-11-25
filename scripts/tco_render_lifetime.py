"""Isolated source transformations and memory tracing for the pinned TCO study."""

from contextlib import ExitStack, contextmanager
from hashlib import sha256
import importlib
import inspect
from pathlib import Path
import textwrap
from unittest.mock import patch
import weakref

import torch


POLICIES = ("native", "camera", "n", "g", "ng", "ngd", "camera_d")
NORMALS = {"n", "ng", "ngd"}
EARLY_CLEAR = {"g", "ng", "ngd"}
DEPTH_RECOMPUTE = {"ngd", "camera_d"}


def clear_gradients(*optimizers):
    """Clear completed gradients without touching parameters or optimizer state."""
    for optimizer in optimizers:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Pinned upstream source anchor changed: {old[:100]}")
    return source.replace(old, new, 1)


def _phase_block(source, start, end, phase):
    start_offset = source.index(start)
    end_offset = source.index(end, start_offset)
    body = source[start_offset:end_offset].rstrip("\n")
    prefix = body[: len(body) - len(body.lstrip())]
    wrapped = f'{prefix}with _tco_lifetime_scope("{phase}"):\n{textwrap.indent(body, "    ")}\n'
    return source[:start_offset] + wrapped + source[end_offset:]


def _objective_source(original, skip_normals, profiled):
    source = textwrap.dedent(inspect.getsource(original))
    if skip_normals:
        source = _replace_once(
            source, "\n            if pred_normals is not None:",
            "\n            if pred_normals is not None and (normal_weight > 0 or "
            "(wandb_module is not None and meta_info is not None)):",
        )
    if profiled:
        source = _phase_block(
            source, "            render_depth, render_alp, render_normals,",
            "            # Shapes: depth", "render.expected_depth",
        )
        source = _replace_once(
            source, "            # Shapes: depth",
            '            _tco_lifetime_inventory("ed_outputs", (render_depth, render_alp, render_normals, _))\n'
            "            # Shapes: depth",
        )
        source = _replace_once(
            source, "            # Normal rendering loss across all targets",
            '            _tco_lifetime_inventory("target_normals", ())\n'
            "            # Normal rendering loss across all targets",
        )
        source = _replace_once(
            source, "                # Keep for visualization",
            '                _tco_lifetime_inventory("target_normals", '
            "(tgt_pts_cam_n, tgt_valid_n, v_v_n, v_u_n, norm, tgt_normals))\n"
            "                # Keep for visualization",
        )
        source = _phase_block(
            source, "            if pred_normals is not None",
            "            # RGB rendering across all targets", "render.target_normals",
        )
        source = _phase_block(
            source, "                render_cols, _, _, _, _, _, _ =",
            "                pred_rgb = render_cols", "render.rgb",
        )
        source = _replace_once(
            source, "                pred_rgb = render_cols",
            '                _tco_lifetime_inventory("rgb_outputs", (render_cols, _))\n'
            "                pred_rgb = render_cols",
        )
        source = _replace_once(
            source, "            # Visibility from expected depth across all targets",
            '            _tco_lifetime_inventory("rgb_outputs", ())\n'
            "            # Visibility from expected depth across all targets",
        )
    return source


def _forward_source(original, clear_early, profiled):
    source = textwrap.dedent(inspect.getsource(original))
    if clear_early or profiled:
        before_loop = "    for step in range(tco_steps):\n"
        clear = "        _tco_lifetime_clear(optimizer, heads_optimizer)\n" if clear_early else ""
        if profiled:
            clear = (
                "        with _tco_lifetime_scope('update_entry'):\n"
                '            _tco_lifetime_inventory("cold_outputs", predictions_orig)\n'
                + ("            with _tco_lifetime_scope('zero_grad'):\n"
                   "                _tco_lifetime_clear(optimizer, heads_optimizer)\n" if clear else "")
            )
        source = _replace_once(source, before_loop, before_loop + clear)
        old_clear = (
            "            # Zero gradients for both optimizers\n"
            "            if optimizer is not None:\n"
            "                optimizer.zero_grad(set_to_none=True)\n"
            "            if heads_optimizer is not None:\n"
            "                heads_optimizer.zero_grad(set_to_none=True)\n"
        )
        source = _replace_once(
            source, old_clear,
            "" if clear_early else "            with _tco_lifetime_scope('zero_grad'):\n"
            "                _tco_lifetime_clear(optimizer, heads_optimizer)\n",
        )
    return source


def _patched_function(original, source):
    namespace = {}
    filename = str(Path(original.__code__.co_filename).name) + " (registered lifetime overlay)"
    # Offset lines into the original source so diagnostic stacks remain easy to relate.
    text = "\n" * (original.__code__.co_firstlineno - 1) + source
    exec(compile(text, filename, "exec"), original.__globals__, namespace)
    return namespace[original.__name__]


@contextmanager
def patched_runtime(policy, scope=None, monitor=None):
    """Patch the two native symbols in memory and restore them after the case."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown render-lifetime policy: {policy}")
    obj = importlib.import_module("objective")
    loss = importlib.import_module("tco_loss")
    tco = importlib.import_module("tco_vggt_lora")
    objective = obj.depth_multiview_consistency_energy
    forward = tco.TCO_VGGT_LoRA.forward
    if loss.depth_multiview_consistency_energy is not objective:
        raise ValueError("TCO loss uses a different renderer reference")
    objective_source = _objective_source(objective, policy in NORMALS, scope is not None)
    forward_source = _forward_source(forward, policy in EARLY_CLEAR, scope is not None)
    digests = {
        "objective_callable_sha256": sha256(objective_source.encode("utf-8")).hexdigest(),
        "forward_callable_sha256": sha256(forward_source.encode("utf-8")).hexdigest(),
    }
    helpers = {"_tco_lifetime_clear": clear_gradients}
    if scope is not None:
        if monitor is None:
            raise ValueError("A scope requires a lifetime monitor")
        helpers.update(_tco_lifetime_scope=scope, _tco_lifetime_inventory=monitor.remember)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(objective.__globals__, helpers))
        stack.enter_context(patch.dict(forward.__globals__, helpers))
        objective_overlay = _patched_function(objective, objective_source)
        forward_overlay = _patched_function(forward, forward_source)
        stack.enter_context(patch.object(obj, objective.__name__, objective_overlay))
        stack.enter_context(patch.object(loss, objective.__name__, objective_overlay))
        stack.enter_context(patch.object(tco.TCO_VGGT_LoRA, forward.__name__, forward_overlay))
        yield digests


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


class LifetimeMonitor:
    """Summarize identifiable CUDA storage without keeping graph tensors alive."""

    def __init__(self):
        self.categories = {}
        self.saved_depth = []
        self.optimizer = None

    def remember(self, category, values):
        self.categories[category] = [weakref.ref(value) for value in _tensors(values)]

    def pack_depth(self, value):
        alias = value.detach()
        if alias.is_cuda:
            self.saved_depth.append(weakref.ref(alias))
        return alias

    def inventory(self):
        self.saved_depth = [ref for ref in self.saved_depth if ref() is not None]
        categories = {**self.categories, "saved_depth": self.saved_depth}
        if self.optimizer is not None:
            groups = self.optimizer.param_groups
            gradients = [param.grad for group in groups for param in group["params"]]
            categories["gradients"] = [weakref.ref(grad) for grad in gradients if grad is not None]
            categories["adam_moments"] = [weakref.ref(value) for value in _tensors(self.optimizer.state)]
        all_storages = {}
        result = {}
        for name, refs in categories.items():
            storages = {}
            count = 0
            dtypes = set()
            for ref in refs:
                value = ref()
                if value is None or not value.is_cuda:
                    continue
                count += 1
                dtypes.add(str(value.dtype))
                storage = value.untyped_storage()
                key = (str(storage.device), storage.data_ptr())
                storages[key] = storage.nbytes()
            all_storages.update(storages)
            result[name] = dict(tensors=count, distinct_storages=len(storages),
                                bytes=sum(storages.values()), dtypes=sorted(dtypes))
        result["union"] = dict(distinct_storages=len(all_storages), bytes=sum(all_storages.values()))
        return result
