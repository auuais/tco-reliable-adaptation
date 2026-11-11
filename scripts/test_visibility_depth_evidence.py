import unittest

import numpy as np

from dense_rgb_depth import source_cost
from visibility_depth_evidence import pair_costs,screen


class VisibilityDepthTest(unittest.TestCase):
    def test_long_start_preserves_correct_hypothesis(self):
        h,w=28,45
        y,x=np.indices((h,w),dtype=float)
        rays=np.stack(((x-22)/30,(y-14)/30,np.ones_like(x)),axis=-1)
        k=np.array([[30.,0,22],[0,30,14],[0,0,1.]])
        target=np.random.default_rng(7).uniform(size=(h,w,3)).astype(np.float32)
        source=np.concatenate((target,np.full((h,w,1),2.,dtype=np.float32)),axis=2)
        depth=np.full((h,w),2*np.exp(.08))
        a,ok=pair_costs(target,source,depth,rays,np.array([0,0,0]),k)
        old,valid=source_cost(target,source,depth,rays,np.array([0,0,0]),k)
        np.testing.assert_array_equal(valid,ok['all_candidates'])
        np.testing.assert_array_equal(a['all_candidates','rgb'],old)
        self.assertFalse(valid.any())
        on=ok['starting_depth']
        self.assertGreater(on.mean(),.75)
        np.testing.assert_array_equal(a['starting_depth','depth'].argmin(axis=0)[on],np.full(on.sum(),3))
        self.assertTrue(np.allclose(a['starting_depth','depth'][4][on],.2))

    def test_witnesses_follow_support(self):
        losses=np.ones((4,5,3,3))
        losses[1,1]=.2
        losses[3,1]=.4
        mask=np.zeros((4,3,3),bool)
        mask[1]=True
        mask[3,:,:2]=True
        d=screen(losses,mask)
        np.testing.assert_array_equal(d['single'][0],np.ones((3,3)))
        np.testing.assert_array_equal(d['two_witness'][0],np.tile([1,1,0],(3,1)))
        self.assertFalse(d['four_view'][1].any())


if __name__=='__main__':
    unittest.main()
