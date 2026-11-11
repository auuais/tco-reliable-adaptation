import unittest

import torch

from tco_focal_policy import initial_focal_policy


class ToyTCO(torch.nn.Module):
    def forward_original_vggt(self, images):
        return images, dict(intrinsics=self.model_k.clone())

    def forward(self, images, gt_intrinsics):
        self.forward_original_vggt(images)
        return gt_intrinsics


class FocalIntegrationTest(unittest.TestCase):
    def test_regular_forward_uses_selected_focal_and_leaves_callers_tensor_intact(self):
        model=ToyTCO()
        k=torch.eye(3).repeat(1,5,1,1)
        k[:,:,[0,1],[0,1]]=500.
        model.model_k=k.clone()
        model.model_k[:,:,[0,1],[0,1]]*=1.20
        k[0,3,[0,1],[0,1]]=560.
        k[0,3,0,2]=36.
        before=k.clone()
        for arm,method,geometry,target in (("adaptive","relative",None,500.),("retain","relative",None,560.),
                ("consensus","relative",None,500.),("adaptive","relative_with_folds",False,560.),
                ("adaptive","relative_with_folds",True,500.)):
            records=[]
            with initial_focal_policy(model,arm,method,geometry,lambda heads,record:records.append(record)):
                actual=model(torch.zeros(1),gt_intrinsics=k)
            self.assertEqual(len(records),1)
            expected=before.clone()
            expected[0,3,[0,1],[0,1]]=target
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(k,before))
            self.assertTrue(torch.equal(model(torch.zeros(1),gt_intrinsics=k),before))


if __name__=="__main__":
    unittest.main()
