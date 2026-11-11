"""Check association uniqueness and pose interpolation against known camera motion."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from stage_tum_sparse_inputs import associations,interpolate


class TumTimestampTests(unittest.TestCase):
    def test_depth_cannot_be_reused(self):
        pairs=associations([(1.,'a'),(1.005,'b'),(1.04,'c')],[(1.003,'d'),(1.04,'e')])
        self.assertEqual([(i,j) for i,j,_ in pairs],[(1,0),(2,1)])
        np.testing.assert_allclose([d for _,_,d in pairs],[.002,0.],atol=1e-12)

    def test_translation_and_rotation_interpolation(self):
        q=Rotation.from_euler('z',[0,90],degrees=True).as_quat()
        trajectory=np.column_stack(([1.,1.04],[[0,0,0],[2,0,0]],q))
        pose=interpolate(trajectory,1.02)
        np.testing.assert_allclose(pose[:3,3],[1,0,0],atol=1e-12)
        np.testing.assert_allclose(pose[:3,:3],Rotation.from_euler('z',45,degrees=True).as_matrix(),atol=1e-12)
        self.assertIsNone(interpolate(trajectory,.9))
        trajectory[1,0]=1.1
        self.assertIsNone(interpolate(trajectory,1.02))
