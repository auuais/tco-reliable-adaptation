"""Synthetic CPU checks for D1's frozen sensor-only calibration bridge."""
import copy
import unittest
from unittest.mock import patch

import torch

from profiled_calibration import prepare_calibration, query_residual, solve_scales
import profiled_d1_bridge as bridge


torch.set_num_threads(1)


class SensorBridgeTests(unittest.TestCase):
    def setUp(self):
        y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        self.initial = torch.stack([1. + .04*i + .013*x + .021*y for i in range(5)]).float()
        self.confidence = torch.stack([1.1 + .01*i + .001*x + .002*y for i in range(5)]).float()
        xy = torch.cartesian_prod(torch.tensor([1, 3, 5, 7]), torch.tensor([1, 3, 5, 7]))
        samples, folds = [], []
        for i in range(5):
            measured = self.initial[i, xy[:, 1], xy[:, 0]].double()*(.7+.02*i)
            measured = measured*(1. + .015*torch.sin(torch.arange(16).double()))
            samples.append(dict(xy=xy.clone(),depth=measured))
            folds.append((torch.arange(16)%2).long())
        camera = torch.eye(4, dtype=torch.float64)[None].repeat(5,1,1)
        camera[:,0,3]=torch.arange(5,dtype=torch.float64)*.03
        self.case = dict(K=torch.tensor([[12.,0.,3.5],[0.,11.,3.5],[0.,0.,1.]],dtype=torch.float64),
                         c2w=camera,samples=samples,folds=folds)

    def prepared(self):
        return bridge.prepare_sensor_folds(self.initial,self.confidence,self.case)

    def test_two_fixed_folds_and_teacher_metadata(self):
        value=self.prepared()
        self.assertEqual(set(value),{"states","queries","metadata"})
        self.assertEqual([row["support_fold"] for row in value["metadata"]],[0,1])
        for state,query,row in zip(value["states"],value["queries"],value["metadata"]):
            self.assertEqual(state["edges"],[])
            self.assertEqual(state["graph_filtering"],[])
            self.assertEqual(row["support_counts"],[8]*5)
            self.assertEqual(row["query_counts"],[8]*5)
            self.assertTrue(row["ridge"]>0)
            for key in ("gains","sensor_normalizers","ridge"):
                self.assertFalse(state[key].requires_grad)
            self.assertTrue(all(not packet["depth"].requires_grad for packet in query))

    def test_query_matrix_equals_original_residual_and_solve_objective(self):
        prepared=self.prepared()
        candidate=self.initial*(1.+.01*torch.cos(torch.arange(8))[None,None,:])
        for state,query in zip(prepared["states"],prepared["queries"]):
            matrices=bridge.assemble_predictive_matrices(candidate,state,query)
            self.assertEqual(set(matrices),{"A","b","Q","d","ridge"})
            self.assertEqual(matrices["A"].shape,(40,5))
            self.assertEqual(matrices["Q"].shape,(40,5))
            self.assertTrue(all(value.dtype==torch.float64 and value.device.type=="cpu" for value in matrices.values()))
            for scales in (torch.tensor([.9,1.1,.8,1.2,1.],dtype=torch.float64),
                           solve_scales(matrices["A"],matrices["b"],matrices["ridge"])):
                actual=matrices["Q"]@scales-matrices["d"]
                expected=query_residual(candidate,state,query,scales)
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                torch.testing.assert_close(.5*actual.square().sum(),.5*expected.square().sum(),rtol=1e-12,atol=1e-12)

    def test_arbitrary_graph_is_irrelevant_at_graph_weight_zero(self):
        prepared=self.prepared()
        xy=self.case["samples"][0]["xy"]
        edges=[dict(i=i,j=j,xy_i=xy[:12].clone(),xy_j=xy.roll(3,0)[:12].clone())
               for i in range(5) for j in range(i+1,5)]
        for fold,(sensor,query) in enumerate(zip(prepared["states"],prepared["queries"])):
            support=[{key:value[self.case["folds"][i]==fold] for key,value in sample.items()}
                     for i,sample in enumerate(self.case["samples"])]
            graph=prepare_calibration(self.initial,self.case["K"],self.case["c2w"],support,edges,self.confidence)
            self.assertEqual(len(graph["edges"]),10)
            self.assertEqual(len(graph["graph_filtering"]),10)
            for key in ("gains","sensor_normalizers","ridge"):
                torch.testing.assert_close(sensor[key],graph[key],rtol=0,atol=0)
            for a,b in zip(sensor["support"],graph["support"]):
                for key in ("xy","depth","weight","factor"):
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            for candidate in (self.initial,self.initial+.01):
                a=bridge.assemble_predictive_matrices(candidate,sensor,query)
                b=bridge.assemble_predictive_matrices(candidate,graph,query)
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_assembly_does_not_reprepare_or_modify_fixed_state(self):
        prepared=self.prepared()
        before=copy.deepcopy(prepared)
        with patch.object(bridge,"prepare_calibration",side_effect=AssertionError("Forbidden re-preparation")):
            for state,query in zip(prepared["states"],prepared["queries"]):
                bridge.assemble_predictive_matrices(self.initial,state,query)
                bridge.assemble_predictive_matrices(self.initial*1.07,state,query)
        for old,state in zip(before["states"],prepared["states"]):
            for key in ("gains","sensor_normalizers","ridge","intrinsics","c2w"):
                torch.testing.assert_close(old[key],state[key],rtol=0,atol=0)
            for a,b in zip(old["support"],state["support"]):
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_matrix_linearity_and_fixed_targets(self):
        prepared=self.prepared()
        first=self.initial.double()
        second=first+.1*torch.sin(torch.arange(8,dtype=torch.float64))[None,:,None]
        fraction=.37
        for state,query in zip(prepared["states"],prepared["queries"]):
            m0=bridge.assemble_predictive_matrices(first,state,query)
            m1=bridge.assemble_predictive_matrices(second,state,query)
            mm=bridge.assemble_predictive_matrices((1-fraction)*first+fraction*second,state,query)
            for key in ("A","Q"):
                torch.testing.assert_close(mm[key],(1-fraction)*m0[key]+fraction*m1[key],rtol=1e-14,atol=1e-14)
            for key in ("b","d","ridge"):
                torch.testing.assert_close(m0[key],m1[key],rtol=0,atol=0)
                torch.testing.assert_close(m0[key],mm[key],rtol=0,atol=0)

    def test_source_packet_mutation_does_not_change_prepared_tensors(self):
        prepared=self.prepared()
        before=copy.deepcopy(prepared)
        self.case["samples"][0]["depth"].mul_(2)
        self.case["samples"][1]["xy"].zero_()
        self.case["folds"][2].zero_()
        self.case["c2w"].zero_()
        self.case["K"].zero_()
        self.confidence.zero_()
        for index in range(2):
            for a,b in zip(prepared["queries"][index],before["queries"][index]):
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            for key in ("gains","sensor_normalizers","ridge","intrinsics","c2w"):
                torch.testing.assert_close(prepared["states"][index][key],before["states"][index][key],rtol=0,atol=0)

    def test_teacher_acceptance_ignores_only_unused_graph_metadata(self):
        metadata=self.prepared()["metadata"]
        historical=copy.deepcopy(metadata)
        for row in historical: row["graph_filtering"]=[dict(unused="synthetic graph metadata")]
        result=bridge.verify_teacher_metadata(metadata,historical,rtol=0,atol=0)
        self.assertTrue(result["pass_all"])
        self.assertFalse(result["graph_metadata_compared"])
        historical[0]["gains"][0]+=1e-8
        with self.assertRaisesRegex(ValueError,"gains exceeds"):
            bridge.verify_teacher_metadata(metadata,historical,rtol=0,atol=1e-9)
        self.assertTrue(bridge.verify_teacher_metadata(metadata,historical,rtol=0,atol=1e-7)["pass_all"])

    def test_teacher_wrong_counts_folds_fields_and_values_rejected(self):
        metadata=self.prepared()["metadata"]
        for change in ("count","fold","missing","nan","negative"):
            historical=copy.deepcopy(metadata)
            if change=="count": historical[0]["support_counts"][0]+=1
            elif change=="fold": historical.reverse()
            elif change=="missing": del historical[0]["gains"]
            elif change=="nan": historical[0]["sensor_normalizers"][0]=float("nan")
            else: historical[0]["ridge"]=-1
            with self.subTest(change=change),self.assertRaises(ValueError):
                bridge.verify_teacher_metadata(metadata,historical,rtol=1e-12,atol=1e-12)

    def test_bad_teacher_tolerance_rejected(self):
        metadata=self.prepared()["metadata"]
        for rtol,atol in ((-1,0),(0,float("nan")),(True,0),(0,float("inf"))):
            with self.assertRaisesRegex(ValueError,"tolerances"):
                bridge.verify_teacher_metadata(metadata,metadata,rtol=rtol,atol=atol)

    def test_invalid_initial_raster_confidence_and_view_count_rejected(self):
        for change in ("views","nan","integer","confidence_zero","confidence_shape","confidence_inf"):
            initial=self.initial.clone()
            confidence=self.confidence.clone()
            if change=="views": initial=initial[:4]
            elif change=="nan": initial[0,0,0]=float("nan")
            elif change=="integer": initial=initial.long()
            elif change=="confidence_zero": confidence[0,0,0]=0
            elif change=="confidence_shape": confidence=confidence[:4]
            else: confidence[0,0,0]=float("inf")
            with self.subTest(change=change),self.assertRaises(ValueError):
                bridge.prepare_sensor_folds(initial,confidence,self.case)

    def test_invalid_calibration_rejected_without_orthogonalization(self):
        approximate=copy.deepcopy(self.case)
        approximate["c2w"][0,0,0]=1.0001
        prepared=bridge.prepare_sensor_folds(self.initial,self.confidence,approximate)
        self.assertEqual(float(prepared["states"][0]["c2w"][0,0,0]),1.0001)
        for change in ("singular","negative_focal","bad_bottom","nan"):
            case=copy.deepcopy(self.case)
            if change=="singular": case["c2w"][0,0,:3]=0
            elif change=="negative_focal": case["K"][0,0]=-1
            elif change=="bad_bottom": case["c2w"][0,3,3]=2
            else: case["K"][0,0]=float("nan")
            with self.subTest(change=change),self.assertRaises(ValueError):
                bridge.prepare_sensor_folds(self.initial,self.confidence,case)

    def test_bad_fold_labels_counts_and_observation_coordinates_rejected(self):
        for change in ("float_folds","unknown_fold","small_fold","duplicate_xy","bounds","fractional","nonpositive_depth"):
            case=copy.deepcopy(self.case)
            if change=="float_folds": case["folds"][0]=case["folds"][0].float()
            elif change=="unknown_fold": case["folds"][0][0]=2
            elif change=="small_fold": case["folds"][0][0]=1
            elif change=="duplicate_xy": case["samples"][0]["xy"][0]=case["samples"][0]["xy"][1]
            elif change=="bounds": case["samples"][0]["xy"][0,0]=8
            elif change=="fractional": case["samples"][0]["xy"]=case["samples"][0]["xy"].float()+.5
            else: case["samples"][0]["depth"][0]=0
            with self.subTest(change=change),self.assertRaises(ValueError):
                bridge.prepare_sensor_folds(self.initial,self.confidence,case)

    def test_nonpositive_observation_predictions_fail_without_masking(self):
        prepared=self.prepared()
        for coordinate in (self.case["samples"][0]["xy"][0],self.case["samples"][0]["xy"][1]):
            candidate=self.initial.clone()
            candidate[0,coordinate[1],coordinate[0]]=0
            with self.assertRaisesRegex(ValueError,"positive"):
                bridge.prepare_sensor_folds(candidate,self.confidence,self.case)
            with self.assertRaisesRegex(ValueError,"positive"):
                bridge.assemble_predictive_matrices(candidate,prepared["states"][0],prepared["queries"][0])

    def test_negative_unobserved_pixel_is_not_filtered_or_rejected(self):
        candidate=self.initial.clone()
        candidate[0,0,0]=-1
        prepared=bridge.prepare_sensor_folds(candidate,self.confidence,self.case)
        bridge.assemble_predictive_matrices(candidate,prepared["states"][0],prepared["queries"][0])
        self.assertEqual(float(candidate[0,0,0]),-1)

    def test_query_overlap_and_corrupt_fixed_state_rejected(self):
        prepared=self.prepared()
        for change in ("overlap","ridge","factor","grad","shape"):
            state=copy.deepcopy(prepared["states"][0])
            query=copy.deepcopy(prepared["queries"][0])
            if change=="overlap": query[0]["xy"][0]=state["support"][0]["xy"][0]
            elif change=="ridge": state["ridge"]=torch.tensor(0.,dtype=torch.float64)
            elif change=="factor": state["support"][0]["factor"][0]=float("nan")
            elif change=="grad": state["gains"].requires_grad_(True)
            else: state["shape"]=(4,8,8)
            with self.subTest(change=change),self.assertRaises(ValueError):
                bridge.assemble_predictive_matrices(self.initial,state,query)

    def test_explicit_matrix_guard_detects_query_formula_mismatch(self):
        prepared=self.prepared()
        with patch.object(bridge,"query_residual",return_value=torch.zeros(40,dtype=torch.float64)),self.assertRaisesRegex(ValueError,"disagrees"):
            bridge.assemble_predictive_matrices(self.initial,prepared["states"][0],prepared["queries"][0])


if __name__=="__main__":
    unittest.main()
