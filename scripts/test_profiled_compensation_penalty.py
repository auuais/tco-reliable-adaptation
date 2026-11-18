"""Check forward matching, exact derivatives and trace normalization."""
import unittest

import torch

from profiled_calibration import prepare_calibration, profiled_objective
from profiled_compensation_penalty import (compensation_loss, objective, prepare_fold,
                                           prepare_reference)


class CompensationPenaltyTests(unittest.TestCase):
    def setUp(self):
        self.eye = torch.eye(5, dtype=torch.float64)
        self.a = self.eye.repeat(4, 1) * torch.linspace(.7, 1.4, 20, dtype=torch.float64)[:, None]
        self.q = self.eye.repeat(3, 1) * torch.linspace(.8, 1.3, 15, dtype=torch.float64)[:, None]
        self.b = torch.linspace(.9, 2.1, 20, dtype=torch.float64)
        self.d = torch.linspace(.6, 1.6, 15, dtype=torch.float64)
        self.ridge = .1
        self.reference = prepare_reference(self.a, self.q, self.b, self.ridge)

    def test_trace_matches_exact_calibration_jacobian_in_support_units(self):
        camera = self.a.abs().argmax(dim=1)
        s0 = self.reference["s0"]
        def compensation(shift):
            perturbed = self.a + self.eye[camera] * (shift / s0[camera])[:, None]
            _, scales, _ = objective(perturbed, self.q, self.b, self.d, self.ridge,
                                      self.reference, penalty="compensation")
            return self.q @ (scales-s0)
        jacobian = torch.autograd.functional.jacobian(compensation, torch.zeros_like(self.b))
        for i in range(5):
            torch.testing.assert_close(jacobian[:, camera == i].square().sum(),
                                       self.reference["damping"][i] * (camera == i).sum(),
                                       rtol=1e-12, atol=1e-12)

    def test_pristine_both_penalties_exactly_zero(self):
        for method in ("compensation", "matched_support"):
            loss, _, terms = objective(self.a, self.q, self.b, self.d, self.ridge,
                                       self.reference, penalty=method)
            self.assertEqual(float(terms["regularizer"]), 0.)
            self.assertTrue(torch.equal(loss, terms["base"]))

    def test_detached_controller_matches_forward_with_distinct_gradient(self):
        candidate = (self.a*1.025).detach().requires_grad_()
        inputs = (candidate, self.q, self.b, self.d, self.ridge, self.reference)
        full, full_scales, _ = objective(*inputs, penalty="compensation")
        full_gradient, = torch.autograd.grad(full, candidate)
        fixed, fixed_scales, _ = objective(*inputs, penalty="compensation", detach_solve=True)
        self.assertTrue(torch.equal(full, fixed))
        self.assertTrue(torch.equal(full_scales, fixed_scales))
        self.assertGreater(float(full_gradient.norm()), 0.)
        self.assertFalse(fixed.requires_grad)
        direction = self.a*.03
        h = 1e-5
        def loss(a):
            return objective(a, *inputs[1:], penalty="compensation")[0]
        fd = (loss(candidate+h*direction)-loss(candidate-h*direction))/(2*h)
        torch.testing.assert_close((full_gradient*direction).sum(), fd, rtol=1e-7, atol=1e-9)

    def test_depth_bridge_keeps_exact_p3_forward_and_gradient_at_pristine(self):
        dtype = torch.float64
        initial = 1. + torch.arange(80, dtype=dtype).reshape(5, 4, 4)/80
        c2w = torch.eye(4, dtype=dtype).repeat(5, 1, 1)
        base_xy = torch.tensor([(x, y) for y in range(2) for x in range(4)])
        test_xy = base_xy + torch.tensor([0, 2])
        depth = torch.linspace(.8, 1.3, 8, dtype=dtype)
        support = [dict(xy=base_xy, depth=depth+i*.1) for i in range(5)]
        query = [dict(xy=test_xy, depth=depth+i*.15) for i in range(5)]
        state = prepare_calibration(initial, torch.eye(3, dtype=dtype), c2w,
                                    support, [], torch.ones_like(initial))
        frozen = prepare_fold(initial, state, query)
        z = initial.clone().requires_grad_()
        base, _ = profiled_objective(z, state, query, graph_weight=0.)
        loss, _, terms = compensation_loss(z, state, query, frozen, penalty="compensation")
        self.assertTrue(torch.equal(base, loss))
        self.assertEqual(float(terms["regularizer"]), 0.)
        original, = torch.autograd.grad(base, z)
        updated, = torch.autograd.grad(loss, z)
        self.assertTrue(torch.equal(original, updated))
        z_changed = (z.detach() * 1.025).requires_grad_()
        full, full_scales, _ = compensation_loss(z_changed, state, query, frozen, penalty="compensation")
        detached, scales, _ = compensation_loss(z_changed, state, query, frozen,
                                                 penalty="compensation", detach_solve=True)
        self.assertTrue(torch.equal(full, detached))
        self.assertTrue(torch.equal(full_scales, scales))
        gradient, = torch.autograd.grad(full, z_changed)
        comparator, = torch.autograd.grad(detached, z_changed)
        self.assertFalse(torch.equal(gradient, comparator))


if __name__ == "__main__":
    unittest.main()
