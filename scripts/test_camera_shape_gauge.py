import unittest
import numpy as np
from camera_shape_v10 import compare

class TestGauge(unittest.TestCase):
    def test_global_similarity_does_not_change_relative_camera_fit(self):
        E=np.repeat(np.eye(4)[None],5,0)
        E[:,:3,3]=[[0,0,0],[1,0,0],[0,2,0],[0,0,3],[1,1,0]]
        K=np.repeat(np.diag([500.,500.,1.])[None],5,0)
        Q=np.array([[0.,-1,0],[1,0,0],[0,0,1.]])
        M=E.copy();M[:,:3,:3]=Q.T
        M[:,:3,3]=3*E[:,:3,3]-Q.T@np.array([1.,2.,3.])
        r=compare(M,K,E,K)
        self.assertLess(max(r.values()),1e-8)
        bad=K.copy();bad[:,0,0]*=2
        self.assertAlmostEqual(compare(M,bad,E,K)['focal_mean_abs_log'],np.log(2)/2)
