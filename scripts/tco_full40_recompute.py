"""Change how the frozen VGGT depth head stores activations during TCO."""

from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint


TAPS = (4, 11, 17, 23)
NATIVE_CHUNK = 8
CHUNKS = (8, 4, 2, 1)


def install_head(model, chunk, recompute, diagnostics=None):
    """Retain native DPT chunk dispatch, with an optional checkpoint per chunk."""
    if chunk not in CHUNKS:
        raise ValueError(f"Unsupported depth-head chunk size: {chunk}")
    head = model.depth_head
    if tuple(head.intermediate_layer_idx) != TAPS or head.feature_only:
        raise ValueError("Expected the four-tap VGGT depth/confidence head")
    if chunk != NATIVE_CHUNK:
        head.forward = partial(head.forward, frames_chunk_size=chunk)
    if not recompute and diagnostics is None:
        return

    native_chunk = head._forward_impl

    def run_chunk(taps, images, patch_start, first=None, last=None):
        start = 0 if first is None else first
        end = images.shape[1] if last is None else last
        span = f"[{start}:{end}]"
        scope = nullcontext() if diagnostics is None else diagnostics("depth.chunk" + span)
        with scope:
            if not recompute or not torch.is_grad_enabled() or not any(taps[idx].requires_grad for idx in TAPS):
                return native_chunk(taps, images, patch_start, first, last)

            # Pass views as explicit inputs. Slice the image the same way as DPT.
            tap_slices = [taps[idx] for idx in TAPS]
            image_slice = images
            if first is not None:
                tap_slices = [tap[:, start:end] for tap in tap_slices]
                image_slice = images[:, start:end].contiguous()

            def compute(rgb, *features):
                chunk_taps = [None] * (TAPS[-1] + 1)
                for index, feature in zip(TAPS, features):
                    chunk_taps[index] = feature
                return native_chunk(chunk_taps, rgb, patch_start)

            options = {}
            if diagnostics is not None:
                options["context_fn"] = lambda: (nullcontext(), diagnostics("depth.recompute" + span))
            return checkpoint(compute, image_slice, *tap_slices, use_reentrant=False, preserve_rng_state=True,
                              **options)

    head._forward_impl = run_chunk


def _differences(actual, expected):
    delta = actual.double() - expected.double()
    return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.norm() / expected.double().norm().clamp_min(1e-12)))


def fixture():
    """Use the pinned DPT implementation on CPU to qualify outputs and VJPs."""
    from contextlib import contextmanager

    vggt = Path(__file__).resolve().parents[1] / "third_party/TCO/base_models/vggt"
    sys.path.insert(0, str(vggt))
    try:
        from vggt.heads.dpt_head import DPTHead
    finally:
        sys.path.remove(str(vggt))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261124)
        source = DPTHead(dim_in=16, features=16, out_channels=[8] * 4, output_dim=2, activation="exp")
        source.eval().requires_grad_(False)
        images = torch.rand(2, 10, 3, 28, 42)
        inputs = [torch.randn(2, 10, 7, 16) for _ in TAPS]
        adjoints = (torch.randn(2, 10, 28, 42, 1), torch.randn(2, 10, 28, 42))
    order = torch.tensor([7, 2, 8, 1, 9, 5, 0, 6, 3, 4])
    labels = ("depth", "confidence", *(f"grad_tap_{idx}" for idx in TAPS))
    atol, rtol = 1e-6, 1e-5

    def compare(actual, expected, expect_exact=False):
        if len(actual) != len(expected):
            raise AssertionError("Different number of output or gradient tensors")
        measurements = {}
        for label, a, b in zip(labels, actual, expected):
            if not bool(torch.isfinite(a).all()):
                raise AssertionError(f"{label} produced nonfinite values")
            torch.testing.assert_close(a, b, atol=0 if expect_exact else atol, rtol=0 if expect_exact else rtol)
            measurements[label] = _differences(a, b)
        return measurements

    def run(chunk, recompute, permute=False):
        from copy import deepcopy

        head = deepcopy(source)
        phases = []

        @contextmanager
        def observe(phase):
            phases.append(phase)
            yield

        install_head(SimpleNamespace(depth_head=head), chunk, recompute, observe)
        actual_inputs = [tensor[:, order] if permute else tensor.clone() for tensor in inputs]
        actual_images = images[:, order] if permute else images
        taps = [None] * (TAPS[-1] + 1)
        for index, tensor in zip(TAPS, actual_inputs):
            taps[index] = tensor.requires_grad_(True)

        output = head(taps, actual_images, 1)
        grad_outputs = [tensor[:, order] for tensor in adjoints] if permute else adjoints
        gradients = torch.autograd.grad(output, actual_inputs, grad_outputs=grad_outputs)
        expected_spans = [f"[{start}:{min(start + chunk, 10)}]" for start in range(0, 10, chunk)]
        forwarded = [phase.removeprefix("depth.chunk") for phase in phases if phase.startswith("depth.chunk")]
        replayed = [phase.removeprefix("depth.recompute") for phase in phases if phase.startswith("depth.recompute")]
        assert forwarded == expected_spans
        assert sorted(replayed) == (sorted(expected_spans) if recompute else [])
        assert all(bool(gradient.any()) for gradient in gradients)

        phases.clear()
        with torch.no_grad():
            readout = head(taps, actual_images, 1)
        assert phases == ["depth.chunk" + span for span in expected_spans]
        assert not any(tensor.requires_grad for tensor in readout)
        compare(readout, output)

        values = (*output, *gradients)
        if permute:
            values = tuple(value[:, order.argsort()] for value in values)
        return values, replayed

    original_forward = source.forward
    original_chunk = source._forward_impl
    install_head(SimpleNamespace(depth_head=source), NATIVE_CHUNK, False)
    assert source.forward == original_forward and source._forward_impl == original_chunk

    measurements = {}
    for chunk in CHUNKS:
        eager, _ = run(chunk, False)
        repeated, _ = run(chunk, False)
        compare(repeated, eager, expect_exact=True)
        candidate, replayed = run(chunk, True)
        measurements[str(chunk)] = dict(same_chunk=compare(candidate, eager), replayed_spans=replayed)
        if chunk == 4:
            permuted, _ = run(chunk, True, permute=True)
            measurements[str(chunk)]["permutation"] = compare(permuted, candidate)
    return dict(device="cpu", torch=torch.__version__, views=10, batch=2, atol=atol, rtol=rtol, chunks=measurements)


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(fixture(), indent=2))
