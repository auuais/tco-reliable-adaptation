import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from visual_focal_baseline import centered_pixels, estimate, fundamental_pairs


class VisualFocalTest(unittest.TestCase):
    def test_rectification_produces_equal_focal_axes(self):
        k=np.array([[502.,0.,258.],[0.,491.,196.],[0.,0.,1.]])
        rays=np.array([[.08,-.15],[.24,.11]])
        px=rays*k[[0,1],[0,1]]+k[:2,2]
        np.testing.assert_allclose(centered_pixels(px,k),rays*np.sqrt(502*491))

    def test_noise_free_solver_sees_the_supplied_camera_order(self):
        try:
            import poselib
        except ImportError:
            self.skipTest("PoseLib is needed for the external solver")
        rng=np.random.default_rng(941)
        xyz=rng.uniform([-1.4,-.9,3.2],[1.3,1.2,7.4],(160,3))
        k=np.repeat(np.array([[[500.,0.,258.],[0.,510.,196.],[0,0,1.]]]),5,axis=0)
        k[2,[0,1],[0,1]]*=.82
        uv=[]
        for i,intr in enumerate(k):
            rotation=Rotation.from_rotvec(rng.normal(size=3)*.19).as_matrix()
            x=(xyz-rng.normal(size=3)*.45)@rotation.T
            uv.append((x[:,:2]/x[:,2:])*intr[[0,1],[0,1]]+intr[:2,2])
        pairs=[(i,j,uv[i],uv[j]) for i in range(5) for j in range(i+1,5)]
        fits=fundamental_pairs(pairs,k)
        continuous,snapped,stats=estimate(fits,k)
        self.assertEqual(stats["valid_neighbors"],[4]*5)
        self.assertFalse(stats["snapped_edit"])
        self.assertLess(np.max(abs(np.log(continuous[:,0,0]/k[:,0,0]))),.01)
        np.testing.assert_array_equal(snapped,k)
        a,b,stats=estimate([],k)
        self.assertEqual(stats["valid_neighbors"],[0]*5)
        np.testing.assert_array_equal(a,k)
        np.testing.assert_array_equal(b,k)


if __name__=="__main__":
    unittest.main()
