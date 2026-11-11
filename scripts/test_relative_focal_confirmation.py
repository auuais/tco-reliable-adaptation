import unittest

import numpy as np

from relative_focal_confirmation import confirmation


class RelativeFocalTest(unittest.TestCase):
    def test_shared_model_bias_cancels_per_axis(self):
        uniform = np.tile([500., 530.], (5, 1))
        observed = uniform.copy()
        observed[3] *= 1.12
        for correct, change in ((uniform, True), (observed, False)):
            for model_bias in ([1., 1.], [.8, 1.05], [1.3, 1.25]):
                view, choices = confirmation(observed, correct * model_bias, True)
                self.assertEqual(view, 3)
                self.assertEqual(choices["relative_with_folds"], change)
        self.assertFalse(confirmation(observed, uniform, False)[1]["relative_with_folds"])

    def test_outlier_required(self):
        uniform = np.full((5, 2), 510.)
        view, choices = confirmation(uniform, uniform * 1.2, True)
        self.assertIsNone(view)
        self.assertFalse(any(choices.values()))


if __name__ == "__main__":
    unittest.main()
