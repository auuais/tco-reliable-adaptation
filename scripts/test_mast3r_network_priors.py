"""Check pairwise camera conventions, scale handling and bearing gradients."""
import unittest
import torch
from mast3r_network_priors import camera_points,bearing


class PairPriorTests(unittest.TestCase):
    def test_known_relative_transform_recovers_second_camera_rays(self):
        relative=torch.tensor([[0.,-1,0,.2],[1,0,0,-.1],[0,0,1,.3],[0,0,0,1.]])
        camera=torch.tensor([[.2,.4,2.],[-.3,.1,1.5]])
        scale=torch.tensor(.7)
        pair_frame=(camera-relative[:3,3])@relative[:3,:3]/scale
        _,restored=camera_points(pair_frame,pair_frame,relative,scale)
        torch.testing.assert_close(restored,camera)
        self.assertLess(bearing(restored,camera).item(),1e-8)
        wrong=relative.clone();wrong[0,3]+=.1
        _,perturbed=camera_points(pair_frame,pair_frame,wrong,scale)
        self.assertGreater(bearing(perturbed,camera).item(),.01)

    def test_model_scale_change_preserves_metric_predictions(self):
        relative=torch.eye(4);relative[0,3]=.3
        points=torch.tensor([[.2,.1,2.]])
        a=camera_points(points,points,relative,torch.tensor(.7))
        b=camera_points(2*points,2*points,relative,torch.tensor(.35))
        for x,y in zip(a,b): torch.testing.assert_close(x,y)

    def test_behind_camera_point_has_finite_nonzero_gradient(self):
        point=torch.tensor([[.1,0.,-1.]],requires_grad=True)
        loss=bearing(point,torch.tensor([[.2,.1,1.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(point.grad).all() and point.grad.abs().sum()>0)
