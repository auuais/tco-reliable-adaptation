from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from masked_camera_supervision import without_camera

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/TCO"))
import tco_loss


class MaskedCameraLossTest(unittest.TestCase):
    def setUp(self):
        self.poses=torch.eye(4,dtype=torch.float64).repeat(1,5,1,1)
        self.poses[0,:,0,3]=torch.tensor([0.,.3,1.1,1.8,2.5])
        self.poses[0,:,1,3]=torch.tensor([0.,.2,.1,-.1,.3])
        self.k=torch.tensor([[450.,0,259],[0,455,196],[0,0,1.]],dtype=torch.float64).repeat(1,5,1,1)
        self.images=torch.zeros(1,5,3,4,4,dtype=torch.float64)

    def test_omitted_anchor_with_corrupt_prior_and_different_world_gauges(self):
        source=self.poses.clone()
        world=torch.tensor([[0.,-1,0,4],[1,0,0,3],[0,0,1.,-2],[0,0,0,1]],dtype=source.dtype)
        supplied=torch.linalg.inv(world @ source)
        supplied[0,0,:3,3]+=7
        predicted=self.poses.clone()
        predicted[:,:,:3,3]*=3.5
        original=tco_loss.compute_tco_losses
        self.assertIs(without_camera(original,None),original)
        masked=without_camera(original,0)
        def energy(loss):
            return loss(dict(camera_poses=predicted,intrinsics=self.k), self.images,gt_extrinsics=supplied,
                        gt_intrinsics=self.k,lambda_pose=1.,lambda_intrinsics=1.)[0]
        self.assertGreater(float(energy(original)),.01)
        self.assertLess(abs(float(energy(masked))),1e-7)

    def test_only_external_gradients_are_removed_from_the_suspect_view(self):
        pose=self.poses.clone().requires_grad_()
        k=self.k.clone().requires_grad_()
        depth=torch.ones(1,5,4,4,dtype=pose.dtype,requires_grad=True)
        g=torch.linalg.inv(self.poses)
        g[:,2,0,3]+=.2
        target=self.k.clone()
        target[:,:,0,0]+=10
        def visual(**kw):
            self.assertEqual(kw["depth"].shape[1],5)
            return kw["depth"].square().mean()
        with patch.object(tco_loss,"depth_multiview_consistency_energy",side_effect=visual):
            masked=without_camera(tco_loss.compute_tco_losses,3)
            loss,_=masked(dict(camera_poses=pose,intrinsics=k,depth=depth),self.images,gt_extrinsics=g,
                          gt_intrinsics=target,lambda_pose=1.,lambda_intrinsics=1.,lambda_mv_consistency=.2)
        dp,dk,dd=torch.autograd.grad(loss,(pose,k,depth))
        for grad in (dp,dk):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertEqual(float(grad[:,3].abs().sum()),0.)
            self.assertGreater(float(grad[:,2].abs().sum()),0.)
        self.assertGreater(float(dd[:,3].abs().sum()),0.)


if __name__=="__main__":
    unittest.main()
