"""Validate the relaxed depth-family projector with known representable fields."""
import unittest
import numpy as np
from audit_mast3r_depth_family import projection


class DepthFamilyTests(unittest.TestCase):
    def test_known_cell_depth_and_focal_family_is_representable(self):
        y,x=np.indices((16,24));base=2+.03*x+.02*y+.001*x*y
        anchors=base[4::8,4::8].repeat(8,0).repeat(8,1)
        target=(1.3*anchors+.8*(base-anchors))*1.4
        result=projection(base,target)
        self.assertLess(result['relative_rms_outside_relaxed_family'],1e-12)

    def test_local_shape_change_outside_family_is_detected(self):
        y,x=np.indices((16,16));base=2+.01*x+.02*y
        target=base.copy();target[1,2]+=.1
        result=projection(base,target)
        self.assertGreater(result['relative_rms_outside_relaxed_family'],1e-4)
        self.assertTrue(0<result['outside_energy_fraction']<=1)
