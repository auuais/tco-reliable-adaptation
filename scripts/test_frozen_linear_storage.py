import unittest

import torch

from frozen_linear_storage import cache_linear


class FrozenStorageTest(unittest.TestCase):
    def test_cpu_autocast_preserves_backward_and_fp32_fallback(self):
        torch.manual_seed(100)
        layer=torch.nn.Linear(17,9).requires_grad_(False)
        x=torch.randn(4,3,17)
        original=layer(x)
        def amp():
            a=x.clone().requires_grad_()
            with torch.autocast("cpu",dtype=torch.bfloat16):
                out=layer(a)
            out.float().square().mean().backward()
            return out.detach(),a.grad
        y0,g0=amp()
        original_bytes,shadow_bytes=cache_linear(layer,torch.bfloat16)
        y1,g1=amp()
        self.assertTrue(torch.equal(y0,y1))
        self.assertTrue(torch.equal(g0,g1))
        self.assertTrue(torch.equal(layer(x),original))
        self.assertEqual(original_bytes,2*shadow_bytes)
        self.assertEqual(set(layer.state_dict()),{"weight","bias"})
        with torch.no_grad():layer.weight.add_(1)
        with self.assertRaises(RuntimeError):layer(x)
