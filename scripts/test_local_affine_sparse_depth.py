"""Check the affine ridge solver against an independent augmented least-squares problem."""
import unittest

import numpy as np

from local_affine_sparse_depth import estimate,intercept


class LocalAffineDepthTest(unittest.TestCase):
    def test_weighted_ridge_matches_augmented_fit(self):
        design=np.array([[[1.,-1.,0.],[1.,1.,0.],[1.,0.,1.],[1.,0.,-1.]]])
        weight=np.array([[.5,1.,.8,.2]])
        residual=np.array([[.02,.04,-.01,.05]])
        center=.03
        matrix=np.vstack([design[0]*np.sqrt(weight[0,:,None]),np.diag([1.,2.,2.])])
        target=np.r_[residual[0]*np.sqrt(weight[0]),center,0.,0.]
        expected=np.linalg.lstsq(matrix,target,rcond=None)[0][0]
        self.assertAlmostEqual(intercept(design,weight,residual,center)[0],expected,places=13)

    def test_common_sensor_scale_preserves_model_shape(self):
        depth=np.ones((1,16,16))
        xy=np.array([[0,0],[15,0],[0,15],[15,15],[4,4],[10,4],[4,10],[10,10]])
        samples=np.column_stack([xy,np.full(8,2.)])
        output,gauge=estimate(depth,[samples])
        self.assertAlmostEqual(gauge,2.)
        for corrected in output.values():
            np.testing.assert_allclose(corrected,depth,atol=1e-12)

    def test_insufficient_anchors_is_explicit(self):
        with self.assertRaises(ValueError):
            estimate(np.ones((1,16,16)),[np.ones((7,3))])
