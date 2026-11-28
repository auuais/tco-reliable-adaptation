"""Small CPU-only controls for distillation geometry, optimization, and identity."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tco_two_track_distill import NamedAdam, factors_digest, normalizers, teacher_terms, training_order


def pack(centres, rotation=None, factor=1.):
    pose = torch.eye(4).repeat(5,1,1)
    if rotation is not None:
        pose[:, :3, :3] = rotation
    pose[:, :3, 3] = -(pose[:, :3, :3] @ centres.unsqueeze(-1)).squeeze(-1)
    return {"extrinsics":pose[None], "depth":torch.full((1,5,2,3,1), factor)}


class Objective(unittest.TestCase):
    def setUp(self):
        self.c = torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,2.,0.],[0.,1.,1.],[0.,0.,2.]])
        self.target = pack(self.c)
        self.rig = self.target["extrinsics"][0]

    def test_sim3_invariance(self):
        q = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        different_gauge = pack(3*self.c@q.T + torch.tensor([2.,-3.,8.]), q.T, 3.)
        depth, camera, pixels = teacher_terms(different_gauge, self.target, self.rig)
        self.assertLess(float(depth),1e-12)
        self.assertLess(float(camera),1e-12)
        self.assertEqual(pixels,30)

    def test_depth_gradient_sign_and_camera_gradient(self):
        student = pack(self.c + torch.tensor([[0,0,0],[0,0,.1],[0,0,0],[0,0,0],[0,0,0]]), factor=1.1)
        for value in student.values():
            value.requires_grad_()
        self.target["depth"].requires_grad_()
        d, c, _ = teacher_terms(student, self.target, self.rig)
        self.assertGreater(float(c), 0.)
        (d+c).backward()
        self.assertTrue((student["depth"].grad > 0).all())
        self.assertTrue(torch.isfinite(student["extrinsics"].grad).all())
        self.assertGreater(float(student["extrinsics"].grad.abs().sum()), 0.)
        self.assertIsNone(self.target["depth"].grad)

    def test_validity_and_training_normalization(self):
        student = pack(self.c)
        self.target["depth"][0,0,0,0,0] = float("nan")
        self.assertEqual(teacher_terms(student,self.target,self.rig)[2],29)
        student["depth"][0,1,0,0,0] = 0
        with self.assertRaisesRegex(ValueError,"Student"):
            teacher_terms(student,self.target,self.rig)
        with self.assertRaisesRegex(ValueError,"Unobservable"):
            teacher_terms(pack(self.c*0), self.target, self.rig)
        self.assertEqual(normalizers([(0.,float(i)) for i in range(6)]), (1e-8, 2.5))
        with self.assertRaises(ValueError):
            normalizers([(1.,1.)]*5)

    def test_persistent_adam_equals_uninterrupted_signed_updates(self):
        native = {"a":torch.tensor([.5,-.4]), "b":torch.tensor([.3])}

        def fresh(reversed_order):
            factors = {name:torch.nn.Parameter(native[name].clone())
                       for name in (("b","a") if reversed_order else ("a","b"))}
            return factors, torch.optim.Adam(factors.values(),lr=.0001)

        def step(params,opt,i):
            opt.zero_grad(set_to_none=True)
            for name, value in params.items():
                value.grad = torch.full_like(value,(i+1) * (-1 if name=="b" else .5))
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
            opt.step()

        uninterrupted, opt = fresh(False)
        bank = NamedAdam()
        for i in range(4):
            step(uninterrupted, opt, i)
            factors, optimizer = fresh(bool(i%2))
            bank.install(factors, optimizer)
            step(factors, optimizer, i)
            bank.capture(factors, optimizer)
        for name in native:
            torch.testing.assert_close(uninterrupted[name], bank.values[name], rtol=0, atol=0)
            for key, tensor in opt.state[uninterrupted[name]].items():
                torch.testing.assert_close(tensor, bank.moments[name][key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            artifact=Path(tmp)/"init.safetensors"
            save_file(bank.values,str(artifact))
            self.assertEqual(factors_digest(load_file(str(artifact))), factors_digest(uninterrupted))
        self.assertEqual(bank.updates,4)

    def test_register_six_unique_full40_training_visits(self):
        jobs = [dict(room=f"room{i}",slug=f"train_{i}", id=f"teacher_{i}", visit=80-i,split="Training",
                     arm="Full40",profile="clean",seed=42,steps=40,frames=list(range(5)),input_sha256=str(i)) for i in range(6)]
        self.assertEqual([j["visit"] for j in training_order(jobs)],sorted(j["visit"] for j in jobs))
        jobs[-1]["split"]="Validation"
        with self.assertRaises(ValueError):
            training_order(jobs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
