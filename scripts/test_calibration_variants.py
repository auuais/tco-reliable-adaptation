import unittest
import numpy as np
from calibration_audit import resized_k, NOMINAL_FOCAL
from calibration_variants import calibration_prior


class CalibrationIsolation(unittest.TestCase):
    def test_prior_changes_do_not_change_original_reference(self):
        original = np.tile(resized_k(NOMINAL_FOCAL, uniform=True).astype(np.float32), (5,1,1))
        before = original.copy()
        released = calibration_prior(original, 518, 392, "released")
        axis = calibration_prior(original, 518, 392, "axis_nominal")
        registered = calibration_prior(original, 518, 392, "axis_registration")
        np.testing.assert_array_equal(original, before)
        np.testing.assert_array_equal(released, original)
        self.assertFalse(np.shares_memory(original, released))
        self.assertAlmostEqual(float(axis[0,0,0]), 448.601159, places=4)
        self.assertAlmostEqual(float(registered[0,0,0]), 424.921875, places=5)
        np.testing.assert_array_equal(axis[:,1,1], original[:,1,1])
        np.testing.assert_array_equal(axis[:,:2,2], registered[:,:2,2])
        with self.assertRaises(AssertionError):
            calibration_prior(np.eye(3)[None], 518, 392, "axis_registration")


if __name__ == "__main__":
    unittest.main()
