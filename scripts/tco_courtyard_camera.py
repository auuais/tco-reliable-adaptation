"""Checkpoint VGGT's frozen camera head while preserving native TCO depth batches."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint


def install_camera(model, enabled, diagnostics=None):
    """Keep the original head API and replay only its gradient-bearing forward."""
    if not enabled:
        return
    head = model.camera_head
    native_forward = head.forward

    def forward(aggregated_tokens_list, num_iterations=4):
        if head.training or not torch.is_grad_enabled() or not aggregated_tokens_list:
            return native_forward(aggregated_tokens_list, num_iterations)
        final_tap = aggregated_tokens_list[-1]
        if not final_tap.requires_grad:
            return native_forward(aggregated_tokens_list, num_iterations)

        # The pinned head only reads the last tap. Leave the original iterative
        # detach operations and all camera tokens in the native implementation.
        def compute(tap):
            return tuple(native_forward([tap], num_iterations))

        options = {}
        if diagnostics is not None:
            options["context_fn"] = lambda: (nullcontext(), diagnostics("camera.recompute"))
        poses = checkpoint(compute, final_tap, use_reentrant=False, preserve_rng_state=True, **options)
        return list(poses)

    head.forward = forward


def _compare(actual, expected, atol, rtol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if not bool(torch.isfinite(actual).all()):
        raise ValueError("The camera fixture produced nonfinite values")
    delta = actual.double() - expected.double()
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.double().norm().clamp_min(1e-12)))


def cpu_fixture():
    """Compare native camera outputs and fixed-cotangent VJPs on the CPU."""
    vggt = Path(__file__).resolve().parents[1] / "third_party/TCO/base_models/vggt"
    sys.path.insert(0, str(vggt))
    try:
        from vggt.heads.camera_head import CameraHead
    finally:
        sys.path.remove(str(vggt))

    atol, rtol = 1e-6, 1e-5
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261125)
        source = CameraHead(dim_in=32, trunk_depth=4, num_heads=4).eval().requires_grad_(False)
        tokens = torch.randn(2, 10, 3, 32)
        cotangents = tuple(torch.randn(2, 10, 9) for _ in range(4))

    def run(views, iterations, last_only, enabled):
        head = deepcopy(source)
        events = []

        @contextmanager
        def observe(name):
            events.append(name)
            yield

        original = head.forward
        install_camera(SimpleNamespace(camera_head=head), enabled, observe)
        if not enabled and head.forward != original:
            raise ValueError("Installing the native control changed the camera method")
        final_tap = tokens[:, :views].clone().requires_grad_(True)
        unused_tap = torch.zeros_like(final_tap).requires_grad_(True)
        taps = [unused_tap, final_tap]
        output = head(taps, num_iterations=iterations)
        if (not isinstance(output, list) or len(output) != iterations
                or any(pose.shape != (2, views, 9) for pose in output) or events):
            raise ValueError("The camera fixture changed native forward behavior")
        selected = output[-1:] if last_only else output
        cotangent = cotangents[iterations - 1:iterations] if last_only else cotangents[:iterations]
        gradient, unused = torch.autograd.grad(selected, (final_tap, unused_tap), allow_unused=True,
                                               grad_outputs=tuple(value[:, :views] for value in cotangent))
        expected_events = ["camera.recompute"] if enabled else []
        if events != expected_events or unused is not None or any(p.grad is not None for p in head.parameters()):
            raise ValueError("Camera replay count, tap routing or head freezing changed")
        if not bool(gradient[:, :, 0].any()) or bool(gradient[:, :, 1:].any()):
            raise ValueError("Expected a nonzero cotangent on camera tokens only")

        events.clear()
        with torch.no_grad():
            inference = head(taps, num_iterations=iterations)
        if events or not isinstance(inference, list) or len(inference) != iterations:
            raise ValueError("No-grad camera unexpectedly replayed or changed its API")
        for result, every_refinement in zip(inference, output):
            if result.requires_grad:
                raise ValueError("No-grad camera produced a gradient-bearing output")
            _compare(result, every_refinement, atol, rtol)
        return output, gradient, expected_events

    cases = {}
    for views, iterations, last_only in ((5, 4, True), (5, 4, False), (10, 4, True), (10, 4, False),
                                         (5, 2, False)):
        expected, native_gradient, _ = run(views, iterations, last_only, False)
        actual, gradient, replay = run(views, iterations, last_only, True)
        mode = "last" if last_only else "all"
        cases[f"{views}views_{iterations}iterations_{mode}"] = dict(
            views=views, iterations=iterations, adjoints=mode, replay=replay,
            outputs=[_compare(a, b, atol, rtol) for a, b in zip(actual, expected)],
            last_tap_gradient=_compare(gradient, native_gradient, atol, rtol))

    return dict(device="cpu", torch=torch.__version__, atol=atol, rtol=rtol, cases=cases)


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(cpu_fixture(), indent=2))
