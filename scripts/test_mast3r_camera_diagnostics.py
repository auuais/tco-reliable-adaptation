"""Check relative-camera diagnostic gauge invariance and undefined translation handling."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from diagnose_mast3r_neural_cameras import relative_errors


class CameraDiagnosticTests(unittest.TestCase):
    def test_independent_world_frame_and_positive_scale_do_not_change_errors(self):
        reference=np.repeat(np.eye(4)[None],3,axis=0)
        reference[1,0,3]=.5;reference[2,1,3]=.3
        transformed=reference.copy()
        rotation=Rotation.from_euler('xyz',[30,20,-10],degrees=True).as_matrix()
        transformed[:,:3,:3]=rotation@transformed[:,:3,:3]
        transformed[:,:3,3]=2*(transformed[:,:3,3]@rotation.T)+np.array([1.,2.,3.])
        errors=relative_errors(transformed,reference)
        self.assertLess(errors['relative_rotation_mean_deg'],1e-10)
        self.assertLess(errors['translation_direction_mean_deg'],1e-5)
        self.assertEqual(errors['valid_translation_directions'],2)

    def test_known_rotation_and_zero_baseline_are_reported(self):
        reference=np.repeat(np.eye(4)[None],2,axis=0)
        predicted=reference.copy()
        predicted[1,:3,:3]=Rotation.from_euler('z',10,degrees=True).as_matrix()
        errors=relative_errors(predicted,reference)
        self.assertAlmostEqual(errors['relative_rotation_mean_deg'],10,places=8)
        self.assertIsNone(errors['translation_direction_mean_deg'])
        self.assertEqual(errors['valid_translation_directions'],0)
