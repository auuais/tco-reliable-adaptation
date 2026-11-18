"""Synthetic semantic-input comparison fixtures; no real input files loaded."""
from copy import deepcopy
import unittest

import numpy as np

import check_profiled_m3_reuse_inputs as audit


class ReuseFittingTests(unittest.TestCase):
    def fixture(self):
        case=dict(K=np.eye(3),w2c=np.tile(np.eye(4),(5,1,1)),c2w=np.tile(np.eye(4),(5,1,1)),
            samples=[dict(xy=np.array([[1,2],[3,4]],dtype=np.int64),depth=np.array([1.,2.])) for _ in range(5)],
            folds=[np.array([0,1],dtype=np.int64) for _ in range(5)])
        rows=[np.arange(14,dtype=np.float64).reshape(2,7) for _ in range(5)]
        return case,rows

    def test_all_camera_observation_fold_and_original_row_tensors(self):
        case,rows=self.fixture();tensors=audit.tensor_inventory(case,rows)
        result=audit.compare(tensors,deepcopy(tensors))
        self.assertEqual(len(result),23)
        self.assertEqual(result['view4/original_rows']['shape'],[2,7])

    def test_change_any_tensor_rejected(self):
        first=audit.tensor_inventory(*self.fixture())
        for key in first:
            second=deepcopy(first);second[key].flat[0]+=1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'mismatch'):
                audit.compare(first,second)

    def test_original_rows_order_matters(self):
        first=audit.tensor_inventory(*self.fixture());second=deepcopy(first)
        second['view0/original_rows']=second['view0/original_rows'][::-1].copy()
        with self.assertRaisesRegex(ValueError,'original_rows'):
            audit.compare(first,second)

    def test_dtype_and_missing_tensor_rejected(self):
        first=audit.tensor_inventory(*self.fixture());second=deepcopy(first)
        second['K']=second['K'].astype(np.float32)
        with self.assertRaises(ValueError): audit.compare(first,second)
        del second['K']
        with self.assertRaisesRegex(ValueError,'inventories'): audit.compare(first,second)

    def test_invalid_rows_and_nonfinite_input(self):
        case,rows=self.fixture();rows[0]=rows[0][:,:6]
        with self.assertRaisesRegex(ValueError,'seven-column'): audit.tensor_inventory(case,rows)
        case,rows=self.fixture();case['K'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'): audit.tensor_inventory(case,rows)

    def test_digest_binds_shape_and_dtype(self):
        value=np.arange(16,dtype=np.float64)
        self.assertNotEqual(audit.fingerprint(value)['sha256'],audit.fingerprint(value.reshape(4,4))['sha256'])
        self.assertNotEqual(audit.fingerprint(value)['sha256'],audit.fingerprint(value.astype(np.float32))['sha256'])


if __name__=='__main__':
    unittest.main()
