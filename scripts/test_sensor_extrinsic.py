import unittest

import numpy as np

from fit_sensor_extrinsic import edge, homogeneous, signed_distances


class SensorFrameTest(unittest.TestCase):
    def test_metric_sensor_transform_composes_on_the_left_in_both_views(self):
        pose = homogeneous(np.repeat(np.eye(4)[None,:3], 2, axis=0))
        angle = .3
        c, s = np.cos(angle), np.sin(angle)
        pose[1,:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
        pose[1,:3,3] = [-.3,.1,.1]
        k = np.repeat(np.array([[400.,0,150],[0,410.,120],[0,0,1]])[None],2,axis=0)
        h = np.eye(4)
        h[:3,3] = [.04,.01,-.02]
        h[:3,:3] = [[1,0,0],[0,c,-s],[0,s,c]]
        points = np.random.default_rng(42).uniform([-.5,-.7,2],[.6,.5,4],(100,3))
        matches = []
        for cam,ki in zip(h@pose,k):
            pixels = (points@cam[:3,:3].T+cam[:3,3])@ki.T
            matches.append(pixels[:,:2]/pixels[:,2:])
        data = [edge(pose,k,0,1,*matches)]
        self.assertLess(np.max(np.abs(signed_distances(h,data))),1e-10)
        self.assertGreater(np.median(np.abs(signed_distances(np.linalg.inv(h),data))),1.)
