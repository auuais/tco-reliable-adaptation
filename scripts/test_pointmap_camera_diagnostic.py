import unittest
import numpy as np
from pointmap_camera_diagnostic import recover,fit_camera


class CameraRecoveryTests(unittest.TestCase):
    def make(self,flat=False):
        y,x=np.meshgrid(np.arange(24),np.arange(32),indexing="ij")
        depth=np.full(x.shape,2.) if flat else 2+.5*np.sin(x*.2)*np.cos(y*.3)
        intrinsic=np.array([[32.,0.,15.5],[0.,31.,11.5],[0.,0.,1.]])
        camera=np.stack([(x-15.5)*depth/32,(y-11.5)*depth/31,depth],axis=-1)
        c,s=np.cos(.3),np.sin(.3);rotation=np.array([[c,0,s],[0,1,0],[-s,0,c]]);translation=np.array([.2,-.1,.4])
        return ((camera-translation)@rotation).astype(np.float32),intrinsic,rotation,translation,depth

    def test_camera_and_depth_recover_under_rotation_translation(self):
        points,k,r,t,depth=self.make();result,z=recover(points)
        np.testing.assert_allclose(result["intrinsics"],k,atol=2e-5)
        np.testing.assert_allclose(result["extrinsics"][:3,:3],r,atol=1e-6)
        np.testing.assert_allclose(result["extrinsics"][:3,3],t,atol=1e-6)
        np.testing.assert_allclose(z,depth,atol=1e-6)

    def test_global_similarity_preserves_intrinsics_and_scaled_depth(self):
        points,k,r,t,depth=self.make();result,z=recover(points*3.1+np.array([7.,-4.,2.]))
        np.testing.assert_allclose(result["intrinsics"],k,atol=2e-5)
        np.testing.assert_allclose(z,depth*3.1,atol=3e-6)

    def test_planar_camera_ambiguity_is_rejected(self):
        points,*_=self.make(flat=True)
        with self.assertRaisesRegex(ValueError,"Ambiguous"):
            recover(points)

    def test_nonfinite_input_is_rejected(self):
        points=np.ones((12,3));points[0,0]=np.nan
        with self.assertRaises(ValueError):fit_camera(points,np.ones((12,2)))


if __name__=="__main__":unittest.main()
