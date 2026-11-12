"""Verify camera gauge invariance and sparse-depth gradients before alignment integration."""
import unittest
import torch
from mast3r_joint_prior_loss import terms


class JointPriorTest(unittest.TestCase):
    def cameras(self):
        cameras=torch.eye(4).repeat(2,1,1)
        cameras[1,0,3]=.5
        return cameras

    def test_consistent_measurements_have_zero_loss(self):
        cameras=self.cameras()
        depth=[torch.ones(8)*2,torch.ones(8)*2]
        losses=terms(depth,cameras,depth,cameras,torch.tensor(1.),torch.tensor(2.))
        self.assertTrue(all(v.item()==0 for v in losses.values()))

    def test_global_rigid_frame_and_scale_gauge(self):
        cameras=self.cameras()
        reference=cameras.clone()
        reference[1,0,3]=.6
        measured=[torch.ones(8)*2.1,torch.ones(8)*2.1]
        depth=[torch.ones(8)*2,torch.ones(8)*2]
        first=terms(depth,cameras,measured,reference,torch.tensor(1.),torch.tensor(2.))
        transform=torch.tensor([[0.,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]])
        transformed=transform@cameras
        second=terms(depth,transformed,measured,reference,torch.tensor(1.),torch.tensor(2.))
        scaled=cameras.clone()
        scaled[:,:3,3]*=2
        third=terms([d*2 for d in depth],scaled,measured,reference,torch.tensor(.5),torch.tensor(2.))
        for name in first:
            torch.testing.assert_close(first[name],second[name])
            torch.testing.assert_close(first[name],third[name])

    def test_depth_and_translation_receive_gradients(self):
        cameras=self.cameras().requires_grad_()
        depth=torch.full((2,8),1.8,requires_grad=True)
        reference=self.cameras()
        reference[1,0,3]=.6
        loss=terms(list(depth),cameras,[torch.ones(8)*2]*2,reference,torch.tensor(1.),torch.tensor(2.))
        sum(loss.values()).backward()
        self.assertTrue(torch.isfinite(depth.grad).all() and depth.grad.abs().sum()>0)
        self.assertTrue(torch.isfinite(cameras.grad).all() and cameras.grad[:,:3,3].abs().sum()>0)
