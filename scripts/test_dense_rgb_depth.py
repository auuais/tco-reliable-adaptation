import unittest

import numpy as np

from dense_rgb_depth import ARMS, choose, estimate


class DenseRgbTest(unittest.TestCase):
    def test_known_plane_and_constant_texture(self):
        h,w = 52,100
        f,z = 75.,2.
        k = np.tile([[f,0,w/2],[0,f,h/2],[0,0,1.]],(5,1,1))
        poses = np.tile(np.eye(4),(5,1,1))
        offsets = np.array([-14,-7,0,8,16])
        poses[:,0,3] = -offsets*z/f
        tex = np.random.default_rng(42).uniform(size=(h,w+32,3)).astype(np.float32)
        images = np.stack([tex[:,16+u:16+u+w].transpose(2,0,1) for u in offsets])
        depths = np.full((5,h,w),z)
        depths[2] *= np.exp(-.08)
        maps,_ = estimate(images,depths,k,poses)
        for arm in ARMS:
            self.assertGreater(np.mean(maps[arm][2,9:-9,24:-24]==4),.98)
            self.assertGreater(np.mean(maps[arm][1,9:-9,24:-24]==0),.98)
        maps,_ = estimate(np.zeros_like(images),depths,k,poses)
        self.assertTrue(all(not m.any() for m in maps.values()))

    def test_disagreement_and_missing_validation(self):
        costs = np.ones((4,5,2,3))
        costs[[0,2],1] = .5
        valid = np.ones((4,2,3),dtype=bool)
        valid[3,:,1] = False
        costs[[1,3],1,:,0] = .6
        d,_ = choose(costs,valid)
        np.testing.assert_array_equal(d['two_source'],np.ones((2,3)))
        np.testing.assert_array_equal(d['guarded'],[[1,0,0],[1,0,0]])
        self.assertEqual(d['four_source'][0,1],0)


if __name__=='__main__':
    unittest.main()
