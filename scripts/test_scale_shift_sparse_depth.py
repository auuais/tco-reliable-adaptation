"""Check sparse calibration bounds, measurement units, and failure handling."""
import unittest

import numpy as np

from scale_shift_sparse_depth import fit


class SparseScaleShiftTest(unittest.TestCase):
    def test_depth_units_do_not_change_fit(self):
        x=np.linspace(.5,4.,30)
        y=1.1*x+.04
        first=fit(x,y)
        second=fit(7*x,7*y)
        np.testing.assert_allclose(second,first*[1,7],rtol=1e-10,atol=1e-10)
        self.assertLess(np.mean(abs(first[0]*x+first[1]-y)),np.mean(abs(x-y)))

    def test_constant_depth_has_regularized_solution(self):
        p=fit(np.ones(10),np.ones(10))
        np.testing.assert_allclose(p,[1,0],atol=1e-12)

    def test_bounds_and_bad_measurements(self):
        x=np.linspace(1,2,30)
        p=fit(x,10*x)
        self.assertTrue(.5<=p[0]<=2.)
        self.assertLessEqual(abs(p[1]),.25*np.median(10*x)+1e-12)
        with self.assertRaises(ValueError):
            fit(np.ones(8),np.zeros(8))
