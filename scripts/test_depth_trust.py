import unittest
import numpy as np
from depth_trust import move_along_rays
class BoundedGeometry(unittest.TestCase):
    def test_scale_covariance_and_no_shift_of_in_bound_rays(self):
        z0=np.ones((1,1,5));z1=np.array([[[.5,.98,1.,1.02,2.]]]);c=np.array([[2.,-1,3.]])
        rays=np.array([1.,-.4,1.]);x=c[:,None,None,:]+z1[...,None]*rays
        y,_=move_along_rays(x,c,z0,z1,"bounded_5pct")
        np.testing.assert_array_equal(y[:,:,1:4],x[:,:,1:4])
        np.testing.assert_allclose((y-c[:,None,None,:])[...,2],np.clip(z1,1/1.05,1.05))
        shift=np.array([3.,4.,-1]);z,_=move_along_rays(7*x+shift,7*c+shift,4*z0,7*z1,"bounded_5pct")
        np.testing.assert_allclose(z,7*y+shift)
    def test_invalid_depth_rejected(self):
        with self.assertRaises(ValueError):move_along_rays(np.ones((1,2,2,3)),np.zeros((1,3)),np.zeros((1,2,2)),np.ones((1,2,2)),"half_shape")
