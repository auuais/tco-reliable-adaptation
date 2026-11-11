"""Check RGB/depth pose direction, axial depth units, and foreground visibility."""
import unittest
import numpy as np
from prepare_tum_sparse_reference import compensate


class MotionCompensationTest(unittest.TestCase):
    def test_identity_keeps_integer_rays(self):
        depth=np.array([[1.,2.,0.],[3.,0.,4.]])
        output,valid=compensate(depth,np.eye(3),np.eye(3),np.eye(4),depth.shape)
        np.testing.assert_array_equal(output,depth)
        np.testing.assert_array_equal(valid,depth>0)

    def test_target_camera_translation(self):
        transform=np.eye(4)
        transform[0,3]=-2
        output,valid=compensate(np.full((1,3),2.),np.eye(3),np.eye(3),transform,(1,3))
        np.testing.assert_array_equal(output,[[2.,2.,0.]])
        np.testing.assert_array_equal(valid,[[True,True,False]])

    def test_frontmost_depth_wins(self):
        intrinsic=np.diag([.1,.1,1.])
        output,valid=compensate(np.array([[3.,1.]]),np.eye(3),intrinsic,np.eye(4),(1,1))
        np.testing.assert_array_equal(output,[[1.]])
        self.assertTrue(valid[0,0])
