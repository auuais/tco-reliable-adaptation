import random
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from repeatability_capture import Capture


class ObservationOnly(unittest.TestCase):
    def test_capture_does_not_consume_randomness_or_change_tensor(self):
        with tempfile.TemporaryDirectory() as folder:
            x=torch.tensor([1.125,-3.5],dtype=torch.bfloat16)
            random.seed(17);np.random.seed(18);torch.manual_seed(19)
            expected=(random.random(),np.random.rand(),torch.rand(4))
            random.seed(17);np.random.seed(18);torch.manual_seed(19)
            capture=Capture(Path(folder))
            capture.rng("before")
            capture.tensors("prediction",dict(x=x))
            capture.rng("after")
            actual=(random.random(),np.random.rand(),torch.rand(4))
            self.assertEqual(actual[:2],expected[:2])
            self.assertTrue(torch.equal(actual[2],expected[2]))
            self.assertEqual(capture.records["before_rng"]["sha256"],capture.records["after_rng"]["sha256"])
            np.testing.assert_array_equal(np.load(Path(folder)/"prediction.npz")["x"],x.float().numpy())


if __name__=="__main__": unittest.main()
