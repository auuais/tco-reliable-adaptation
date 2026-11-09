"""Geometry and leakage invariants for the scientific corruption experiment."""
import unittest
import numpy as np
from prior_corruptions import PROFILES, axis_rotation, corrupt_priors


class PriorTests(unittest.TestCase):
    def setUp(self):
        self.e = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
        self.e[:, :3, 3] = -np.array([[0,0,0], [1,0,0], [0,2,0], [1,2,1], [-1,0,1]])
        self.k = np.repeat(np.array([[500,0,250],[0,500,190],[0,0,1]], dtype=np.float32)[None],5,axis=0)

    def test_clean_exact_and_inputs_immutable(self):
        before_e, before_k = self.e.copy(), self.k.copy()
        e,k,_ = corrupt_priors(self.e,self.k,"clean",1729,"test")
        self.assertTrue(np.array_equal(e,self.e) and np.array_equal(k,self.k))
        for profile in PROFILES:
            corrupt_priors(self.e,self.k,profile,1729,"test")
        self.assertTrue(np.array_equal(self.e,before_e) and np.array_equal(self.k,before_k))

    def test_geometry_and_component_pairing(self):
        center0 = -np.linalg.solve(self.e[:,:3,:3],self.e[:,:3,3:4])[...,0]
        for profile,settings in PROFILES.items():
            e,k,m = corrupt_priors(self.e,self.k,profile,1729,"test")
            np.testing.assert_array_equal(e[0],self.e[0])
            np.testing.assert_allclose(np.linalg.det(e[:,:3,:3]),1,atol=1e-6)
            delta=e[1:,:3,:3] @ np.swapaxes(self.e[1:,:3,:3],1,2)
            angles=np.rad2deg(np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1)))
            np.testing.assert_allclose(angles,settings['rotation_deg'],atol=1e-3)
            centers=-np.linalg.solve(e[:,:3,:3],e[:,:3,3:4])[...,0]
            np.testing.assert_allclose(np.linalg.norm(centers[1:]-center0[1:],axis=1),m['center_displacement_magnitude'],atol=5e-7)
            np.testing.assert_array_equal(k[:,:2,2],self.k[:,:2,2])
            np.testing.assert_allclose(k[:,0,0]/self.k[:,0,0],settings['focal_scale'])

    def test_reproducibility_and_global_rng_isolation(self):
        np.random.seed(43); expected=np.random.random(4); np.random.seed(43)
        a=corrupt_priors(self.e,self.k,'mixed_severe',1729,'test')
        np.testing.assert_array_equal(np.random.random(4),expected)
        b=corrupt_priors(self.e,self.k,'mixed_severe',1729,'test')
        np.testing.assert_array_equal(a[0],b[0])
        self.assertEqual(a[2],b[2])
        c=corrupt_priors(self.e,self.k,'mixed_severe',1730,'test')
        self.assertFalse(np.array_equal(a[0],c[0]))

    def test_translation_origin_invariance(self):
        shifted=self.e.copy(); shifted[:,:3,3]-=np.array([10,-5,3])
        a=corrupt_priors(self.e,self.k,'translation10',1729,'test')
        b=corrupt_priors(shifted,self.k,'translation10',1729,'test')
        self.assertAlmostEqual(a[2]['trajectory_diameter'],b[2]['trajectory_diameter'])
        np.testing.assert_allclose(a[0][:,:3,3]-self.e[:,:3,3],b[0][:,:3,3]-shifted[:,:3,3],atol=1e-6)

    def test_preserve_nonorthogonality_and_reject_zero_baseline(self):
        self.e[1,0,0] *= 1.0001
        result=corrupt_priors(self.e,self.k,'rotation5',1729,'test')[0]
        np.testing.assert_allclose(result[1,:3,:3].T@result[1,:3,:3],self.e[1,:3,:3].T@self.e[1,:3,:3],atol=2e-7)
        self.e[:,:3,3]=0
        with self.assertRaises(ValueError): corrupt_priors(self.e,self.k,'translation10',1729,'test')


if __name__ == '__main__': unittest.main()
