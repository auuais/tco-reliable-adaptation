"""Check units, pose direction, crop visibility, and native sample identity."""
import unittest

import numpy as np

from neighbor_sparse_depth import project_samples


class NeighborProjectionTest(unittest.TestCase):
    def test_translated_plane(self):
        uv = np.array([[320,240],[400,240],[320,300],[560,300]])
        raw = np.array([2000,65535,0,10001])
        rays = np.column_stack([(uv[:,0]-320)/525,(uv[:,1]-240)/525,np.ones(4)])
        camera = np.array([[400.,0,210],[0,400,175],[0,0,1]])
        # A target camera displaced right by 10 cm sees the plane 20 pixels to the left.
        pose = np.eye(4)
        pose[0,3] = -.1
        anchors = project_samples(raw,uv,rays,pose,camera,21)
        np.testing.assert_allclose(anchors,[[190,175,2,21,320,240,2000]])

    def test_projection_excludes_outside_and_behind(self):
        uv = np.zeros((2,2))
        rays = np.array([[600,0,1.],[0,0,1.]])
        pose = np.eye(4)
        pose[2,3] = -1
        anchors = project_samples(np.array([2000,500]),uv,rays,pose,np.eye(3),29)
        self.assertEqual(anchors.shape,(0,7))
