import unittest

import numpy as np

from fixed_camera_depth_scale import correct, fit, point_map, sample


class DepthScaleTest(unittest.TestCase):
    def setUp(self):
        self.k = np.tile([[35.,0.,31.5],[0.,35.,23.5],[0.,0.,1.]], (5,1,1))
        self.c = np.array([[0,0,0],[.19,.05,0],[.35,-.06,0],[.57,.02,0],[.78,.08,0]])
        self.e = np.tile(np.eye(4), (5,1,1))
        self.e[:,:3,3] = -self.c
        self.truth = np.array([.95,1.06,.98,1.03,1.09])
        self.depth = np.broadcast_to((2/self.truth)[:,None,None], (5,48,64))

    def tracks(self, seed, z=2., outliers=0):
        rng = np.random.default_rng(seed)
        xyz = rng.uniform([-.1,-.5,z], [.8,.5,z], (60,3))
        p = xyz[None]-self.c[:,None]
        uv = 35*p[:,:,:2]/p[:,:,2:] + [31.5,23.5]
        fold = []
        for i in range(5):
            for j in range(i+1,5):
                b = uv[j].copy()
                b[:outliers] += [10,-7]
                fold.append((i,j,uv[i],b))
        return fold

    def test_recovery_outliers_and_similarity(self):
        folds = [self.tracks(2,outliers=2), self.tracks(3)]
        scales, info = correct(folds, self.depth, self.k, self.e)
        self.assertEqual(info['decision'], 'accepted')
        np.testing.assert_allclose(scales['guarded'], self.truth, atol=.008, rtol=0)
        q = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        offset = np.array([4.,-6.,1.])
        shifted = self.e.copy()
        shifted[:,:3,:3] = q.T
        shifted[:,:3,3] = 3*self.e[:,:3,3] - q.T @ offset
        rescaled,_ = correct(folds, 3*self.depth, self.k, shifted)
        np.testing.assert_allclose(rescaled['guarded'], scales['guarded'], atol=1e-10)
        original = point_map(self.depth, self.k, self.e, scales['guarded'])
        transformed = point_map(3*self.depth, self.k, shifted, rescaled['guarded'])
        np.testing.assert_allclose(transformed, 3*original @ q.T + offset, atol=2e-6)

    def test_identity_and_veto(self):
        flat = np.ones_like(self.depth)*2
        scales, info = correct([self.tracks(2),self.tracks(3)], flat, self.k, self.e)
        self.assertEqual(info['decision'],'validation_veto')
        np.testing.assert_array_equal(scales['guarded'], np.ones(5))
        scales, info = correct([self.tracks(2,z=2.16),self.tracks(3)], flat, self.k, self.e)
        self.assertEqual(info['decision'],'validation_veto')
        self.assertGreater(min(scales['five']),1.05)
        np.testing.assert_array_equal(scales['guarded'], np.ones(5))

    def test_invalid_depth_missing_pairs_and_rank(self):
        depth = np.tile(np.arange(1.,5.), (4,1))
        depth[2,2] = np.nan
        rays, valid = sample(np.array([[.5,.4],[1.1,1.2],[-.1,1.],[3,3],[np.nan,1.]]),
                             depth,self.k[0],np.eye(3))
        np.testing.assert_array_equal(valid,[True,False,False,True,False])
        self.assertAlmostEqual(rays[0,2],1.5)
        bad = [p for p in self.tracks(1) if 4 not in p[:2]]
        scales, info = correct([bad,self.tracks(3)], self.depth,self.k,self.e)
        self.assertEqual(info['decision'],'insufficient_support')
        np.testing.assert_array_equal(scales['guarded'],np.ones(5))
        _,info = correct([self.tracks(3),bad], self.depth,self.k,self.e)
        self.assertEqual(info['decision'],'missing_validation')
        _,status = fit((np.ones((6,3,5)),np.zeros((6,3)),np.ones(6)/6,[]),5)
        self.assertEqual(status,'rank_deficient')


if __name__ == '__main__':
    unittest.main()
