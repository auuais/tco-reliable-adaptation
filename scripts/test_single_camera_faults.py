import unittest

import numpy as np

from localize_camera_prior_faults import inject, score


class SingleCameraFaultTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5725)
        world = rng.uniform([-1,-1,3], [1,1,6], (80,3))
        self.cameras = np.repeat(np.eye(4)[None], 5, axis=0)
        self.cameras[:,:3,3] = [[0,0,0],[-.1,0,0],[0,-.15,0],[.1,-.1,.01],[-.2,-.2,.02]]
        self.k = np.repeat(np.array([[[500.,0,259],[0,500,196],[0,0,1]]]),5,axis=0)
        projections = []
        for e, k in zip(self.cameras,self.k):
            p = (world @ e[:3,:3].T + e[:3,3]) @ k.T
            projections.append(p[:,:2]/p[:,2:])
        self.pairs = [(i,j,projections[i],projections[j]) for i in range(5) for j in range(i+1,5)]

    def test_exact_scene_and_all_five_possible_fault_locations(self):
        clean = score(self.cameras,self.k,self.pairs)
        self.assertLess(clean["best"],1e-10)
        self.assertEqual(clean["winners"],list(range(5)))
        for view in range(5):
            for fault in ("rotation","translation","focal"):
                e,k = inject(self.cameras,self.k,view,fault,43,"synthetic")
                self.assertEqual(score(e,k,self.pairs)["winners"],[view],(view,fault))
                np.testing.assert_array_equal(np.delete(e,view,axis=0),np.delete(self.cameras,view,axis=0))
                np.testing.assert_array_equal(np.delete(k,view,axis=0),np.delete(self.k,view,axis=0))

    def test_common_metric_scale_cannot_change_sampson_error(self):
        e,k = inject(self.cameras,self.k,2,"rotation",17,"synthetic")
        before = score(e,k,self.pairs)
        e[:,:3,3] *= 3.7
        after = score(e,k,self.pairs)
        np.testing.assert_allclose(before["scores"],after["scores"],rtol=1e-9,atol=1e-10)

    def test_sparse_camera_and_missing_support_remain_in_denominator(self):
        partial = score(self.cameras,self.k,self.pairs[:2])
        self.assertEqual(partial["winners"],[0])
        self.assertIsNone(partial["scores"][1])
        self.assertEqual(score(self.cameras,self.k,[])["winners"],[])


if __name__ == "__main__":
    unittest.main()
