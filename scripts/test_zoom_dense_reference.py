import unittest

import numpy as np

from zoom_dense_reference import zoom_reference


class DenseZoomTest(unittest.TestCase):
    def test_asymmetric_reference_principal_point_moves_with_the_image_crop(self):
        k=np.array([[10.,0,3.],[0,11.,2.],[0,0,1.]])
        center=np.array([4.,3.])
        e=np.eye(4)
        e[:3,3]=[.1,-.2,.4]
        z=np.full((7,9),2.)
        mask=np.ones_like(z,dtype=bool)
        mask[3,4]=False
        xyz,valid,cal=zoom_reference(z,mask,k,e,center,1.25)
        np.testing.assert_allclose(cal[:2,2],center+1.25*(k[:2,2]-center))
        self.assertFalse(valid[3,4])
        for x,y in ((1,1),(7,5)):
            self.assertTrue(valid[y,x])
            cam=xyz[y,x]+e[:3,3]
            original=k@cam
            np.testing.assert_allclose(original[:2]/original[2],center+(np.array([x,y])-center)/1.25)
            self.assertAlmostEqual(cam[2],2.)

    def test_wider_output_has_no_fabricated_depth_at_the_border(self):
        k=np.array([[10.,0,2.],[0,10.,2.],[0,0,1.]])
        z=np.ones((5,5))
        _,mask,_=zoom_reference(z,z>0,k,np.eye(4),np.array([2.,2.]),.5)
        self.assertEqual(mask.sum(),9)
        self.assertFalse(mask[2,0])
        self.assertTrue(mask[2,2])


if __name__=="__main__":
    unittest.main()
