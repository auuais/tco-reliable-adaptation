import unittest

import numpy as np

from calibration_audit import resized_k
from focal_zoom_gradient import noisy_poses, predict_edits
from focal_zoom_observations import batches, SCALE


class ZoomGradientTest(unittest.TestCase):
    def test_crop_maps_registered_rays_without_padding(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is required for the full image run")
        x=np.broadcast_to(np.arange(518,dtype=np.float32),(392,518))
        y=np.broadcast_to(np.arange(392,dtype=np.float32)[:,None],(392,518))
        native=np.repeat(np.stack([x,y,x*0+1])[None],5,axis=0)
        views=dict(batches(native))
        self.assertEqual(len(views),11)
        center=resized_k(525.,518,392,uniform=False)[:2,2]
        for label,factor in (("base",SCALE),("v3_wide",1),("v3_tight",SCALE**2)):
            img=views[label]
            for col,row in ((50,55),(470,310)):
                expected=center+(np.array([col,row])-center)/factor
                np.testing.assert_allclose(img[3,:2,row,col],expected,atol=.02,rtol=0)
            np.testing.assert_array_equal(img[:,2],np.ones_like(img[:,2]))
            np.testing.assert_array_equal(img[[0,1,2,4]],views["base"][[0,1,2,4]])
        np.testing.assert_array_equal(views["v3_wide"][3],native[3])

    def test_unchanged_metadata_abstains_even_when_poses_are_noisy(self):
        rig=np.repeat(np.eye(4)[None],5,axis=0)
        rig[:,0,3]=np.arange(5)
        k=np.repeat(resized_k(525.,518,392,uniform=False)[None],5,axis=0)
        noise=noisy_poses(rig,"synthetic",137,1.,.02)
        np.testing.assert_array_equal(noise[0],rig[0])
        np.testing.assert_array_equal(noise,noisy_poses(rig,"synthetic",137,1.,.02))
        centers=lambda e:np.linalg.inv(e)[:,:3,3]
        np.testing.assert_allclose(np.linalg.norm(centers(noise)[1:]-centers(rig)[1:],axis=1),.08)
        edits,gains,proposed=predict_edits(noise,k,k[:,[0,1],[0,1]]*1.2,[[],[]])
        self.assertFalse(any(edits.values()))
        self.assertEqual(gains,[None,None])
        np.testing.assert_array_equal(proposed,k)


if __name__=="__main__":
    unittest.main()
