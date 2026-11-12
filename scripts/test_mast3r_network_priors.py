"""Check pairwise camera conventions, scale handling and bearing gradients."""
import unittest
import torch
from mast3r_network_priors import camera_points,bearing,objective


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

    def test_two_stream_gradients_and_pose_free_baseline(self):
        first=torch.tensor([.1,0.,1.9]).expand(1,16,16,3).clone().requires_grad_()
        other=first.detach().clone().requires_grad_()
        pair=({'pts3d':first},{'pts3d_in_other_view':other})
        xy=torch.zeros((2,8),dtype=torch.long)
        poses=[torch.eye(4),torch.eye(4)]
        args=([pair,pair],[torch.tensor(1.)]*2,xy,torch.full((8,),2.),torch.tensor(2.))
        rays=torch.tensor([[[0.,0.,1.]]])
        wrong=[p.clone() for p in poses];wrong[0][0,3]=5
        for intrinsic_weight in (0.,.01):
            baseline,_=objective(*args,poses,rays,0.,intrinsic_weight)
            altered,_=objective(*args,wrong,rays,0.,intrinsic_weight)
            torch.testing.assert_close(baseline,altered,rtol=0,atol=0)
            a=torch.autograd.grad(baseline,(first,other),retain_graph=True)
            b=torch.autograd.grad(altered,(first,other),retain_graph=True)
            for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
        joint,_=objective(*args,poses,rays,.01,.01)
        joint.backward()
        for grad in (first.grad,other.grad):
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum()>0)
