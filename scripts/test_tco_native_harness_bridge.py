"""Cheap fixture tests for the post-core harness comparison and conditional expenditure."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tco_native_harness_bridge import fire_rule, prediction_delta, read, CORE, HEAD, FIRE


class Harness(unittest.TestCase):
    def test_strictly_bounded_fire_decision(self):
        lean={"scene_seconds":20.,"allocation_retries":0}
        current={"scene_seconds":50.,"allocation_retries":3}
        timeout=["timeout","timeout"]
        self.assertTrue(fire_rule(lean,current,True,timeout))
        self.assertFalse(fire_rule(lean,current,False,timeout))
        self.assertFalse(fire_rule(lean,current,True,["timeout","completed"]))
        self.assertFalse(fire_rule(lean,dict(current,scene_seconds=49.99),True,timeout))
        self.assertFalse(fire_rule(lean,dict(current,allocation_retries=0),True,timeout))
        self.assertFalse(fire_rule(None,current,True,timeout))

    def test_current_and_history_compare_identical_named_fields(self):
        with tempfile.TemporaryDirectory() as root:
            left,right=Path(root)/"lean.npz",Path(root)/"current.npz"
            zeros=np.zeros((3,4),np.float32)
            fields={k:zeros.copy() for k in ("canonical","depth","depth_conf","extrinsics","intrinsics")}
            np.savez(left,**fields)
            fields["depth"]+=.25
            np.savez(right,**fields)
            report=prediction_delta(left,right)
            self.assertTrue(report["canonical"]["bitwise_equal"])
            self.assertEqual(report["depth"]["rms"],.25)
            del fields["extrinsics"]
            np.savez(right,**fields)
            with self.assertRaises(ValueError):
                prediction_delta(left,right)

    def test_prespecified_cases_exist_once(self):
        rows=read(CORE)["jobs"]
        a,=[r for r in rows if r["id"]==HEAD]
        b,=[r for r in rows if r["id"]==FIRE]
        self.assertEqual((a["room"],a["seed"],a["steps"]),("heads/seq-01",42,10))
        self.assertEqual((b["room"],b["seed"],b["steps"]),("fire/seq-03",44,40))


if __name__=="__main__":
    unittest.main()
