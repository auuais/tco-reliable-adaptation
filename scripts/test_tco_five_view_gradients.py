"""CPU regression checks for the real-loss replay harness and its receipts."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import tco_five_view_gradients as g


class ReplayTests(unittest.TestCase):
    def test_raw_dtype_sign_and_unused_channels(self):
        leaf = torch.tensor([1.5, -2.0], dtype=torch.bfloat16, requires_grad=True)
        y = {key: leaf * (idx+1) for idx, key in enumerate(g.LEAVES)}
        q = {"depth": torch.tensor([-3., 2.], dtype=leaf.dtype),
             "depth_conf": None, "camera_poses": torch.zeros_like(leaf), "intrinsics": None}
        outputs, vectors = g.paired_vectors(y, q)
        torch.autograd.backward(outputs, vectors)
        self.assertTrue(torch.equal(leaf.grad, q["depth"]))
        self.assertNotEqual(g.tensor_hash(q["depth"]), g.tensor_hash(q["depth"].float()))
        self.assertNotEqual(g.tensor_hash(q["depth"]), g.tensor_hash(-q["depth"]))
        self.assertFalse(g.array_stats(None, torch.zeros_like(leaf))["passed"])
        for bad_q in ({k: v for k, v in q.items() if k != "depth_conf"},
                      dict(q, depth=q["depth"].float()), dict(q, depth=torch.full_like(leaf, float("nan")))):
            with self.assertRaises(ValueError):
                g.paired_vectors(y, bad_q)

    def test_wrong_update_and_swapped_presence(self):
        y = {key: torch.tensor([1.]) for key in g.LEAVES}
        q = dict(y, intrinsics=None)
        anchor = {"step": 1, "state_hash": "saved1", "boundary": g.boundary(y),
                  "gradient_present": {key: value is not None for key, value in q.items()}}
        b = g.make_q_bundle(anchor, q)
        with self.assertRaisesRegex(ValueError, "q update"):
            g.verify_q_bundle(b, dict(anchor,step=2))
        with self.assertRaisesRegex(ValueError, "q anchor state"):
            g.verify_q_bundle(b, dict(anchor,state_hash="different"))
        b["q"]["intrinsics"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "q presence"):
            g.verify_q_bundle(b, anchor)

    def test_full_capture_replay_with_different_incoming_state(self):
        class Tiny(torch.nn.Module):
            def __init__(self, start):
                super().__init__()
                self.factors = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(float(start)))
                                                       for _ in range(384)])

            def backbone_peft_setup(self):
                params = list(self.parameters())
                return torch.optim.Adam(params, lr=0.0003), params

            def aggregate_with_patch_tokens(self, use_checkpoint):
                z = torch.stack(list(self.factors)).mean()
                return {name: (i+1) * z for i, name in enumerate(g.LEAVES)}

        def fake_rng():
            return dict(cpu=torch.get_rng_state(), cuda=torch.get_rng_state())

        def cpu_draw(rng, count):
            generator = torch.Generator().set_state(rng["cuda"])
            return torch.randperm(count, generator=generator).tolist(), generator.get_state()

        def observe(state, audit, fixed):
            def loss(**kwargs):
                order = torch.randperm(5).tolist()
                audit.setdefault("render_orders", []).append(order)
                y = kwargs["predictions"]
                scalar = y["depth"].square() + y["camera_poses"].square() + 0*y["intrinsics"]
                return scalar, {"fixture": scalar.detach()}
            return None, loss

        def fake_inputs(kwargs, path):
            g.save(path, {"kwargs": g.cpu_copy(kwargs)})

        runtime = SimpleNamespace(setup_execution=lambda model, *a, **k: None, observe_losses=observe)
        cut = SimpleNamespace(capture_renderer_inputs=fake_inputs)
        inherited = SimpleNamespace(FIELDS=g.LEAVES, tensor_hash=g.tensor_hash)
        with tempfile.TemporaryDirectory() as root, patch.multiple(g, rng_state=fake_rng,
                set_rng=lambda v: torch.set_rng_state(v["cpu"]), view_draw=cpu_draw), \
                patch.dict(sys.modules, {"check_tco_render_lifetime": cut, "run_tco_one_percent": inherited}):
            def run(mode, name, source, start):
                out = Path(root)/name
                out.mkdir()
                model, audit, state = Tiny(start), {}, {"updates": 0}
                torch.manual_seed(23 if source is None else 711)
                with g.gradient_context(runtime, dict(gradient_mode=mode, steps=2), out, source=source):
                    runtime.setup_execution(model)
                    opt, params = model.backbone_peft_setup()
                    _, observe_loss = runtime.observe_losses(state, audit, False)
                    gradients, factors = [], []
                    for _ in range(2):
                        ys = model.aggregate_with_patch_tokens(use_checkpoint=True)
                        scalar, _ = observe_loss(predictions=ys, images=torch.zeros(1,5))
                        opt.zero_grad(set_to_none=True)
                        scalar.backward()
                        gradients.append(torch.stack([p.grad for p in params]).clone())
                        torch.nn.utils.clip_grad_norm_(params, 1.)
                        opt.step()
                        state["updates"] += 1
                        factors.append(torch.stack(params).detach().clone())
                return out, gradients, factors

            native, grad, factors = run("capture", "capture", None, 1)
            _, replay, updated = run("replay", "replay", native, 6)
            for step in (0,1):
                self.assertTrue(torch.equal(grad[step], replay[step]))
                self.assertTrue(torch.equal(factors[step], updated[step]))
            negative, flipped, _ = run("negative", "negative", native, 2)
            self.assertFalse(torch.equal(grad[0], flipped[0]))
            summary = json.loads((negative/"gradient_summary.json").read_text())
            self.assertEqual(summary["matched_updates"]["1"]["flipped"], "depth")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
