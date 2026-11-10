"""Check real Adam history, complete rollback, exception paths and RNG reuse."""
import copy
import random
import unittest
import numpy as np
import torch
from shared_gradient_proposals import OptimizerSnapshot,propose


def equal_tree(test,a,b):
    if torch.is_tensor(a):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(a.keys(),b.keys())
        for key in a:equal_tree(test,a[key],b[key])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal_tree(test,x,y)
    else:test.assertEqual(a,b)


class SharedProposalTests(unittest.TestCase):
    def setUp(self):
        self.p=torch.nn.Parameter(torch.tensor([1.,-2.,.4],dtype=torch.float64))
        self.opt=torch.optim.Adam([self.p],lr=.03)
        for gradient in [torch.tensor([.2,-.7,.9]),torch.tensor([-.3,.1,.5])]:
            self.p.grad=gradient.to(self.p);self.opt.step()
        self.groups={"visual":[torch.tensor([.3,-.9,.2],dtype=torch.float64)],
                     "prior":[torch.tensor([-.7,.4,1.1],dtype=torch.float64)]}

    def test_matches_direct_adam_and_restores_existing_history(self):
        before=self.p.detach().clone();state=copy.deepcopy(self.opt.state_dict());grad=self.p.grad.clone()
        expected={}
        for name,keys in {"full":["visual","prior"],"without":["visual"]}.items():
            p=torch.nn.Parameter(before.clone());optimizer=torch.optim.Adam([p],lr=.03);optimizer.load_state_dict(copy.deepcopy(state))
            p.grad=sum((self.groups[k][0] for k in keys),torch.zeros_like(p));torch.nn.utils.clip_grad_norm_([p],1.);optimizer.step()
            expected[name]=(p.detach().clone(),copy.deepcopy(optimizer.state_dict()))
        result=propose(self.opt,self.groups,{"full":["visual","prior"],"without":["visual"],"zero":[]},lambda name:self.p)
        self.assertTrue(torch.equal(self.p,before));self.assertTrue(torch.equal(self.p.grad,grad));equal_tree(self,self.opt.state_dict(),state)
        for name,(p,optimizer_state) in expected.items():
            self.assertTrue(torch.equal(result[name]["evaluation"],p));equal_tree(self,result[name]["state"].state,optimizer_state)
        self.assertTrue(torch.equal(result["zero"]["evaluation"],before))
        # The returned candidate remains usable after other proposals ran.
        result["without"]["state"].restore();self.assertTrue(torch.equal(self.p,expected["without"][0]))

    def test_rng_is_shared_and_restored_even_on_rejection(self):
        anchor=OptimizerSnapshot(self.opt)
        def sample(name):return (random.random(),float(np.random.random()),torch.rand(3))
        expected=sample("expected");anchor.restore()
        result=propose(self.opt,self.groups,{"a":["visual"],"b":["prior"]},sample)
        equal_tree(self,result["a"]["evaluation"],expected);equal_tree(self,result["b"]["evaluation"],expected)
        equal_tree(self,sample("after"),expected)

    def test_exception_restores_parameters_gradients_and_moments(self):
        anchor=OptimizerSnapshot(self.opt)
        def fail(name):
            torch.rand(4)
            raise RuntimeError("invalid validation support")
        with self.assertRaises(RuntimeError):propose(self.opt,self.groups,{"a":["visual"]},fail)
        self.assertTrue(torch.equal(self.p.cpu(),anchor.values[0]));self.assertTrue(torch.equal(self.p.grad.cpu(),anchor.gradients[0]))
        equal_tree(self,self.opt.state_dict(),anchor.state);self.assertTrue(torch.equal(torch.get_rng_state(),anchor.torch_rng))

    def test_invalid_gradients_and_missing_groups_fail_before_mutation(self):
        before=self.p.detach().clone()
        with self.assertRaises(ValueError):propose(self.opt,{"bad":[torch.full_like(self.p,float("nan"))]},{"a":["bad"]},lambda name:0)
        with self.assertRaises(ValueError):propose(self.opt,self.groups,{"a":["missing"]},lambda name:0)
        self.assertTrue(torch.equal(self.p,before))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable; exercised separately on local RTX 3080")
    def test_cuda_moments_and_rng_restore(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,2.],device="cuda"))
        optimizer=torch.optim.Adam([parameter],lr=.01)
        parameter.grad=torch.tensor([.2,-.3],device="cuda");optimizer.step()
        anchor=OptimizerSnapshot(optimizer)
        expected=torch.rand(4,device="cuda").cpu();anchor.restore()
        result=propose(optimizer,{"g":[torch.tensor([.1,.4],device="cuda")]},
            {"first":["g"],"second":["g"]},lambda name:torch.rand(4,device="cuda"))
        self.assertTrue(torch.equal(result["first"]["evaluation"],expected))
        self.assertTrue(torch.equal(result["second"]["evaluation"],expected))
        self.assertTrue(torch.equal(torch.rand(4,device="cuda").cpu(),expected))
        self.assertTrue(torch.equal(parameter.cpu(),anchor.values[0]))
        self.assertEqual(optimizer.state[parameter]["exp_avg"].device.type,"cuda")


if __name__=="__main__":unittest.main()
