import unittest
import numpy as np
from calibration_audit import resized_k, mapping_findings, projection, NOMINAL_FOCAL


class CalibrationTests(unittest.TestCase):
    def test_anisotropic_resize_preserves_camera_rays(self):
        k=np.array([[500,0,320],[0,500,240],[0,0,1]],dtype=float)
        xyz=np.array([[.1,-.2,1],[-.4,.7,2],[.5,.3,3]])
        xy=projection(xyz,np.eye(4),k)
        expected=(xy+.5)*np.array([518/640,392/480])-.5
        actual=projection(xyz,np.eye(4),resized_k(500))
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        self.assertGreater(np.max(np.abs(projection(xyz,np.eye(4),resized_k(500,uniform=True))-expected)),1.)

    def test_uniform_resize_and_identity_cases(self):
        np.testing.assert_allclose(resized_k(500,320,240),resized_k(500,320,240,True))
        np.testing.assert_allclose(resized_k(500,640,480),[[500,0,320],[0,500,240],[0,0,1]])

    def test_expected_protocol_discrepancies_are_detected(self):
        r=mapping_findings()
        self.assertGreater(r["nominal_focal_excess_percent"],5.)
        self.assertLess(r["nominal_focal_excess_percent"],6.)
        self.assertGreater(r["released_fx_excess_vs_axis_percent"],.8)
        self.assertGreater(r["max_probe_mapping_displacement_pixels"],4.)


if __name__=="__main__":unittest.main()
