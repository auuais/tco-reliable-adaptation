"""Negative controls: a complete-looking run with changed treatment must fail."""
import copy
import unittest
from aggregate_repeat_calibration import validate_job


class PlannedTreatment(unittest.TestCase):
    def setUp(self):
        self.job=dict(run_id="test",scene="fire/seq-03",calibration="released",seed=42,execution_repeat=0,
                      steps=40,size=518,frames=[0,200,400,600,800],learning_rate=.0003,pose_scale=1.,intrinsics_scale=1.)
        self.record=dict(status="complete",research_dirty=False,base_parameters_modified=[],manifest_sha256="m",runner_sha256="s",
            arguments={**{k:v for k,v in self.job.items() if k not in ["pose_scale","intrinsics_scale"]},
                       "pose_weight_scale":1.,"intrinsics_weight_scale":1.},
            configuration=dict(lambda_pose=1.,lambda_intrinsics=.01,tco_lr=.0003,lambda_mv_consistency=.2),
            image_shape=[5,3,392,518],metrics=dict(accuracy_mean_m=.1,completeness_mean_m=.1,normal_consistency_mean=.8))

    def test_changes_rejected(self):
        validate_job(self.job,self.record,"m","s")
        for field,value in [("calibration","axis_nominal"),("seed",123),("execution_repeat",1),("steps",39)]:
            changed=copy.deepcopy(self.record);changed["arguments"][field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError):validate_job(self.job,changed,"m","s")
        changed=copy.deepcopy(self.record);changed["metrics"]["accuracy_mean_m"]=float("nan")
        with self.assertRaises(AssertionError):validate_job(self.job,changed,"m","s")


if __name__=="__main__":unittest.main()
