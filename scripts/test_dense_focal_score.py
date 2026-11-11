import unittest

from prepare_dense_focal_pilot import jobs
from score_dense_focal_pilot import METRICS, REFS, summarize


class DenseScoreTest(unittest.TestCase):
    def test_mismatched_seed_cannot_be_hidden_by_pooled_average(self):
        rows=[]
        for j in jobs():
            factor={"consensus":.9,"retain":1.0,"adaptive":.7 if j["seed"]==42 else .92}[j["arm"]]
            metrics={r:dict(initial=dict(zip(METRICS,(1.,2.,.9))),final=dict(zip(METRICS,(factor,2*factor,.9)))) for r in REFS}
            rows.append(dict(job=j,scores=metrics,edited=False,cpu_fp32_decision=False,peak_allocated_bytes=1))
        result=summarize(rows)
        self.assertEqual(result["comparator"],"consensus")
        self.assertTrue(result["checks"]["gain"])
        self.assertFalse(result["checks"]["seed_directions"])
        self.assertFalse(result["supports_transfer"])
        with self.assertRaises(ValueError):
            summarize(rows[:-1])
