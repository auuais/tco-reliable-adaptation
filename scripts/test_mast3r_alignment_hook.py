"""Check disabled-objective fidelity, additive gradients and hook restoration behavior."""
import unittest

import torch
from mast3r_alignment_hook import compile_hook


def sparse_scene_optimizer(value):
    def loss_func(K,w2cam,pts3d,pix_loss):
        return K.square()
    K=value
    w2cam=cam2w=value
    depthmaps=[value]
    pts3d=pix_loss=value
    loss_dust3r_w=0.
    def loss_dust3r(cam2w,pts3d,lossd):
        return cam2w*0
    lossd=value
    iter=0
    loss=loss_func(K,w2cam,pts3d,pix_loss)+loss_dust3r_w*loss_dust3r(cam2w,pts3d,lossd)
    return loss


class Mast3rHookTest(unittest.TestCase):
    def test_disabled_hook_preserves_gradient(self):
        hooked=compile_hook(sparse_scene_optimizer)
        x=torch.tensor(2.,requires_grad=True)
        loss=hooked(x)
        loss.backward()
        self.assertEqual(loss.item(),4.)
        self.assertEqual(x.grad.item(),4.)

    def test_callback_adds_a_differentiable_prior(self):
        hooked=compile_hook(sparse_scene_optimizer)
        observed=[]
        def prior(k,cameras,depth,phase,iteration):
            observed.append((phase,iteration))
            return (cameras-1).square()
        x=torch.tensor(2.,requires_grad=True)
        loss=hooked(x,prior_callback=prior)
        loss.backward()
        self.assertEqual(loss.item(),5.)
        self.assertEqual(x.grad.item(),6.)
        self.assertEqual(observed,[('loss_func',0)])
