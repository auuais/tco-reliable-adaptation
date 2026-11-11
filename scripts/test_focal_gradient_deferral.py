import sys
import unittest

import numpy as np
import torch

from focal_gradient_deferral import first_difference, ROOT


class FocalDeferralTests(unittest.TestCase):
    def test_equal_slopes_on_both_axes_are_required(self):
        original = np.array([10., 10.], np.float32)
        candidate = np.array([8., 12.], np.float32)
        prediction = np.array([[13., 13.], [11., 11.]], np.float32)
        self.assertEqual(first_difference(prediction, original, candidate), 1)
        self.assertIsNone(first_difference(prediction[:1], original, candidate))

    def test_lazy_choice_matches_eager_adam_through_first_different_gradient(self):
        sys.path.insert(0, str(ROOT / "third_party/TCO"))
        from constraints import direct_intrinsics_energy

        def trajectory(start, chosen_alt, lazy):
            adapter = torch.nn.Parameter(torch.zeros(2))
            opt = torch.optim.Adam([adapter], lr=.21)
            directions = torch.tensor([[1., .25], [.75, .5], [.5, .7]])
            original = torch.full((3, 2), 10.)
            alternative = original.clone()
            alternative[0] = torch.tensor([8., 9.])
            chosen = alternative if chosen_alt else original
            answered = False
            queried = None
            states = []
            for step in range(14):
                predicted = start + adapter * directions
                if lazy and not answered:
                    first = (predicted.detach()[0] - original[0]).clamp(-1, 1)
                    second = (predicted.detach()[0] - alternative[0]).clamp(-1, 1)
                    if not torch.equal(first, second):
                        answered = True
                        queried = step
                target = chosen if answered or not lazy else original
                def matrices(focal):
                    k = torch.eye(3).repeat(1, 3, 1, 1)
                    k[0, :, 0, 0] = focal[:, 0]
                    k[0, :, 1, 1] = focal[:, 1]
                    return k
                # Both losses share a trainable adapter, as visual and camera terms do in TCO.
                loss = direct_intrinsics_energy(matrices(predicted), matrices(target)) + .01 * adapter.square().sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([adapter], .3)
                grad = adapter.grad.detach().clone()
                opt.step()
                slot = opt.state[adapter]
                states.append(torch.cat((grad, adapter.detach(), slot["exp_avg"], slot["exp_avg_sq"],
                                         slot["step"].reshape(1))).clone())
            return torch.stack(states), queried

        for initial, expected in ((9.8, "immediate"), (12., "later"), (30., "never")):
            for decision in (True, False):
                eager, _ = trajectory(initial, decision, False)
                lazy, step = trajectory(initial, decision, True)
                self.assertTrue(torch.equal(eager, lazy))
                self.assertEqual("never" if step is None else "immediate" if step == 0 else "later", expected)


if __name__ == "__main__":
    unittest.main()
