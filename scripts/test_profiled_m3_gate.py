"""Adversarial synthetic-only fixtures for the prospective pure M3 aggregator.

Run directly or through unittest discovery. No study files, arrays, GPUs or
reference metrics are read. These fixtures do not register a real M3 scorer.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
import unittest

import profiled_m3_gate as gate


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def synthetic(nominee="P3"):
    registration=dict(schema="profiled_m3_gate_registration_v1",nominee=nominee,policy=gate.policy(),
        provenance={key:h(key) for key in gate.PROVENANCE},scenes={},observations={},references={})
    for index,scene in enumerate(gate.SCENES):
        registration["scenes"][scene]=dict(rgb_sha256=h(scene+"/rgb"),clean_pose_sha256=h(scene+"/clean"),
            noisy_pose_sha256=h(scene+"/noisy"),noise_seed=20261112+index)
        for side,grid in itertools.product(gate.SIDES,gate.GRIDS):
            key="/".join((scene,side,grid))
            registration["observations"][key]={field:h(key+"/"+field)
                for field in ("packet_sha256","folds_sha256","native_rows_sha256")}
        for reference in gate.REFERENCES:
            key=scene+"/"+reference
            registration["references"][key]=dict(reference_sha256=h(key+"/archive"),mask_sha256=h(key+"/mask"),
                                                  mask_count=(1,100,10000)[index])
    rows=[]
    arms=gate.CONTROLS+(nominee,)
    relative={"P0":Fraction(1),"P1":Fraction(4,5),"P2":Fraction(17,20),"P5":Fraction(9,10),nominee:Fraction(7,10)}
    for scene,side,grid,pose,repeat,arm,reference in itertools.product(
            gate.SCENES,gate.SIDES,gate.GRIDS,gate.POSES,gate.REPEATS,arms,gate.REFERENCES):
        sid=gate.state_id(scene,side,grid,pose,repeat,arm)
        scene_spec=registration["scenes"][scene]
        ref_spec=registration["references"][scene+"/"+reference]
        identity=dict(rgb_sha256=scene_spec["rgb_sha256"],pose_sha256=scene_spec[pose+"_pose_sha256"],
            **registration["observations"]["/".join((scene,side,grid))],
            **{key:ref_spec[key] for key in ("reference_sha256","mask_sha256")},
            **{key:registration["provenance"][key] for key in
               ("runtime_sha256","original_checkpoint_sha256","initialization_contract_sha256")},
            optimizer_seed=42,process_receipt_sha256=h(sid+"/process"))
        # Different meters/counts expose wrong scene weighting or normalization joins.
        base=Fraction((1,10,100)[gate.SCENES.index(scene)]*(1,2,4)[repeat]
                      *(1 if side=="earlier" else 5)*(1 if grid=="centered" else 2)
                      *(1 if pose=="clean" else 3)*(1 if reference=="registered_rgb" else 7))
        metrics=dict(accuracy=float(base*relative[arm]),completeness=float(2*base*relative[arm]))
        rows.append(dict(scene=scene,side=side,grid=grid,pose=pose,repeat=repeat,arm=arm,reference=reference,
            state_id=sid,identity=identity,endpoint_sha256={kind:h(sid+"/"+kind) for kind in gate.ENDPOINTS},
            metrics_sha256=h(sid+"/"+reference+"/metrics"),final=metrics,native=dict(metrics),
            validity=dict(mask_count=ref_spec["mask_count"],evaluated_mask_count=ref_spec["mask_count"],
                main_nonpositive_depth=0,native_nonpositive_depth=0,main_nonfinite_fields=0,native_nonfinite_fields=0)))
    return registration,dict(schema="profiled_m3_scene_metrics_v1",registration_sha256=gate.canonical_sha256(registration),rows=rows)


def set_relative(packet,arm,accuracy,completeness=None,**selection):
    completeness=accuracy if completeness is None else completeness
    p0={(row["scene"],row["side"],row["grid"],row["pose"],row["repeat"],row["reference"]):row
        for row in packet["rows"] if row["arm"]=="P0"}
    for row in packet["rows"]:
        if row["arm"]==arm and all(row[key]==value for key,value in selection.items()):
            base=p0[tuple(row[key] for key in ("scene","side","grid","pose","repeat","reference"))]
            row["final"]={field:float(Fraction(str(base["final"][field]))*Fraction(str(factor)))
                          for field,factor in (("accuracy",accuracy),("completeness",completeness))}


def first_row(packet,**selection):
    return next(row for row in packet["rows"] if all(row[key]==value for key,value in selection.items()))


def one_cell(report):
    return next(row for row in report["cells"] if row["cell"]=="earlier_centered_clean")


class M3GateFixtures(unittest.TestCase):
    def test_complete_inventory_and_unequal_counts_use_equal_scene_means(self):
        registration,packet=synthetic()
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        self.assertEqual(len(result["raw_metric_rows"]),720)
        self.assertEqual(len(result["scene_table"]),1440)
        self.assertEqual(len(result["aggregate_table"]),480)
        self.assertEqual(result["observed_states"],360)
        set_relative(packet,"P3",.9,scene="heads")
        set_relative(packet,"P3",.6,scene="chess")
        set_relative(packet,"P3",.6,scene="fire")
        result=gate.evaluate(registration,packet)
        row=one_cell(result)["repetitions"][0]
        self.assertEqual(row["primary_nominee_S"]["exact"],"7/10")
        self.assertFalse(row["checks"]["no_scene_regression"])
        self.assertFalse(result["pass_all"])

    def test_exact_five_percent_r0_passes_but_less_fails(self):
        registration,packet=synthetic()
        set_relative(packet,"P3",.76)
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        self.assertEqual(one_cell(result)["repetitions"][0]["primary_gain"]["exact"],"1/20")
        set_relative(packet,"P3",.7600000001)
        self.assertFalse(gate.evaluate(registration,packet)["pass_all"])

    def test_repeats_only_need_positive_gain_and_zero_fails(self):
        registration,packet=synthetic()
        set_relative(packet,"P3",.799,repeat=1)
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        set_relative(packet,"P3",.8,repeat=2)
        result=gate.evaluate(registration,packet)
        self.assertFalse(result["pass_all"])
        self.assertFalse(one_cell(result)["repetitions"][2]["checks"]["primary_gain_positive"])

    def test_exact_control_ties_use_numeric_id_and_frozen_identity(self):
        registration,packet=synthetic()
        set_relative(packet,"P2",.8)
        set_relative(packet,"P5",.8)
        result=gate.evaluate(registration,packet)
        self.assertEqual(set(result["frozen_controls"].values()),{"P1"})
        packet["frozen_controls"]=dict(result["frozen_controls"])
        self.assertTrue(gate.evaluate(registration,packet)["pass_all"])
        packet["frozen_controls"]["earlier_centered_clean"]="P2"
        with self.assertRaisesRegex(ValueError,"frozen.*controls"):
            gate.evaluate(registration,packet)

    def test_control_is_not_reselected_by_repeat_or_reference(self):
        registration,packet=synthetic()
        set_relative(packet,"P5",.1,repeat=1)
        set_relative(packet,"P5",.05,reference="released")
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        self.assertEqual(set(result["frozen_controls"].values()),{"P1"})
        self.assertTrue(all(row["selected_control"]=="P1" for cell in result["cells"] for row in cell["repetitions"]))

    def test_every_control_is_eligible_and_zero_selected_denominator_rejected(self):
        for best in gate.CONTROLS:
            with self.subTest(best=best):
                registration,packet=synthetic()
                for arm in ("P1","P2","P5"):
                    set_relative(packet,arm,1.1 if best=="P0" else .8 if best==arm else .9)
                result=gate.evaluate(registration,packet)
                self.assertEqual(set(result["frozen_controls"].values()),{best})
        registration,packet=synthetic()
        set_relative(packet,"P1",0)
        with self.assertRaisesRegex(ValueError,"Zero selected-control"):
            gate.evaluate(registration,packet)

    def test_exact_same_condition_repeat_reference_P0_normalization(self):
        registration,packet=synthetic()
        result=gate.evaluate(registration,packet)
        candidate=[row for row in result["aggregate_table"] if row["arm"]=="P3" and row["endpoint"]=="final"]
        self.assertEqual({row["S"]["exact"] for row in candidate},{"7/10"})
        self.assertEqual({cell["mechanism"]["execution_floor"]["exact"] for cell in result["cells"]},{"0"})

    def test_component_uses_normalized_equal_scene_aggregate(self):
        registration,packet=synthetic()
        set_relative(packet,"P3",.816,.5)
        self.assertTrue(gate.evaluate(registration,packet)["pass_all"])
        set_relative(packet,"P3",.8160001,.5,repeat=2)
        result=gate.evaluate(registration,packet)
        self.assertFalse(result["pass_all"])
        self.assertFalse(one_cell(result)["repetitions"][2]["checks"]["accuracy_component_safe"])
        self.assertTrue(one_cell(result)["repetitions"][2]["checks"]["primary_gain_positive"])
        registration,packet=synthetic()
        set_relative(packet,"P3",.5,.8160001,repeat=1)
        self.assertFalse(one_cell(gate.evaluate(registration,packet))["repetitions"][1]["checks"]["completeness_component_safe"])

    def test_one_scene_failure_in_last_repeat_is_not_hidden_by_scene_means(self):
        registration,packet=synthetic()
        set_relative(packet,"P3",.801,scene="fire",side="earlier",grid="centered",pose="clean",repeat=2)
        result=gate.evaluate(registration,packet)
        row=one_cell(result)["repetitions"][2]
        self.assertTrue(row["checks"]["primary_gain_positive"])
        self.assertFalse(row["checks"]["no_scene_regression"])
        self.assertEqual(result["failed_cells"],["earlier_centered_clean"])

    def test_alternate_failure_in_one_cell_or_repeat_is_not_averaged_away(self):
        registration,packet=synthetic()
        set_relative(packet,"P3",.8001,side="later",grid="shifted",pose="noisy",repeat=2,reference="released")
        result=gate.evaluate(registration,packet)
        self.assertFalse(result["pass_all"])
        self.assertEqual(result["failed_cells"],["later_shifted_noisy"])
        set_relative(packet,"P3",.8,side="later",grid="shifted",pose="noisy",repeat=2,reference="released")
        self.assertTrue(gate.evaluate(registration,packet)["pass_all"])

    def test_matched_detached_arm_follows_nominee(self):
        registration,packet=synthetic("P4")
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        self.assertEqual(result["matched_detached_arm"],"P5")
        set_relative(packet,"P5",.69,repeat=2)
        result=gate.evaluate(registration,packet)
        self.assertFalse(result["pass_all"])
        self.assertFalse(one_cell(result)["mechanism"]["checks"]["full_beats_detached_all_repeats"])

    def test_mechanism_floor_is_sum_of_two_maximum_absolute_variations(self):
        registration,packet=synthetic()
        set_relative(packet,"P3",.75,repeat=1)
        set_relative(packet,"P3",.73,repeat=2)
        set_relative(packet,"P2",.9,repeat=2)
        mechanism=one_cell(gate.evaluate(registration,packet))["mechanism"]
        self.assertEqual(mechanism["nominee_execution_floor_term"]["exact"],"1/20")
        self.assertEqual(mechanism["detached_execution_floor_term"]["exact"],"1/20")
        self.assertEqual(mechanism["execution_floor"]["exact"],"1/10")

    def test_mechanism_strict_threshold_and_absolute_margin(self):
        # Call the same public complete-inventory gate. The accuracy gates may
        # also fail here; inspect the independent mechanism decision itself.
        registration,packet=synthetic()
        set_relative(packet,"P3",.7)
        set_relative(packet,"P2",.700001)
        mechanism=one_cell(gate.evaluate(registration,packet))["mechanism"]
        self.assertEqual(mechanism["execution_floor"]["exact"],"0")
        self.assertFalse(mechanism["pass_all"])
        set_relative(packet,"P2",.7000010001)
        self.assertTrue(one_cell(gate.evaluate(registration,packet))["mechanism"]["pass_all"])
        set_relative(packet,"P2",.85)
        set_relative(packet,"P3",.799999,repeat=1)
        set_relative(packet,"P2",.9,repeat=2)
        mechanism=one_cell(gate.evaluate(registration,packet))["mechanism"]
        self.assertEqual(mechanism["r0_contrast_minus_floor_and_margin"]["exact"],"0")
        self.assertFalse(mechanism["pass_all"])

    def test_candidate_nonpositive_main_disqualifies_without_mask_shrink(self):
        registration,packet=synthetic()
        row=first_row(packet,arm="P3",repeat=2,reference="released")
        row["validity"]["main_nonpositive_depth"]=1
        result=gate.evaluate(registration,packet)
        self.assertFalse(result["pass_all"])
        self.assertEqual(len(result["raw_metric_rows"]),720)
        self.assertEqual(sum(row["validity"]["main_nonpositive_depth"] for row in result["raw_metric_rows"]),1)
        row["validity"]["evaluated_mask_count"]-=1
        with self.assertRaisesRegex(ValueError,"mask.*filtered"):
            gate.evaluate(registration,packet)

    def test_native_invalid_counts_reported_without_substituting_gate_endpoint(self):
        registration,packet=synthetic()
        row=first_row(packet,arm="P3")
        row["validity"]["native_nonpositive_depth"]=1
        row["native"]=dict(accuracy=1000,completeness=1000)
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        set_relative(packet,"P3",.9)
        self.assertFalse(gate.evaluate(registration,packet)["pass_all"])

    def test_all_control_invalid_counts_retained_and_nonfinite_fields_fail(self):
        registration,packet=synthetic()
        row=first_row(packet,arm="P0")
        row["validity"]["main_nonpositive_depth"]=1
        result=gate.evaluate(registration,packet)
        self.assertTrue(result["pass_all"])
        pair=[item for item in packet["rows"] if item["state_id"]==row["state_id"]]
        for item in pair:
            item["validity"]["main_nonfinite_fields"]=1
        self.assertFalse(gate.evaluate(registration,packet)["pass_all"])

    def test_missing_duplicate_extra_or_unexpected_inventory_rejected(self):
        for mutation in (lambda p:p["rows"].pop(),lambda p:p["rows"].append(deepcopy(p["rows"][0])),
                         lambda p:p["rows"].__setitem__(-1,deepcopy(p["rows"][0])),
                         lambda p:p["rows"][0].update(repeat=3),lambda p:p["rows"][0].update(reference="other")):
            registration,packet=synthetic()
            mutation(packet)
            with self.assertRaises(ValueError):
                gate.evaluate(registration,packet)

    def test_nan_infinity_negative_and_zero_P0_denominators_rejected(self):
        for value in (float("nan"),float("inf"),-1):
            registration,packet=synthetic()
            first_row(packet,arm="P3")["final"]["accuracy"]=value
            with self.assertRaises(ValueError):
                gate.evaluate(registration,packet)
        for reference,repeat,kind in itertools.product(gate.REFERENCES,gate.REPEATS,gate.ENDPOINTS):
            registration,packet=synthetic()
            first_row(packet,arm="P0",reference=reference,repeat=repeat)[kind]["accuracy"]=0
            with self.assertRaisesRegex(ValueError,"Zero P0"):
                gate.evaluate(registration,packet)

    def test_extreme_finite_inputs_cannot_emit_infinite_report_numbers(self):
        registration,packet=synthetic()
        first_row(packet,arm="P0")["final"]["accuracy"]=1e-300
        first_row(packet,arm="P3")["final"]["accuracy"]=1e300
        with self.assertRaisesRegex(ValueError,"not finite"):
            gate.evaluate(registration,packet)

    def test_reference_repeat_noise_and_input_pairing_are_bound(self):
        for field in ("pose_sha256","packet_sha256","folds_sha256","native_rows_sha256",
                      "reference_sha256","mask_sha256","runtime_sha256","original_checkpoint_sha256"):
            registration,packet=synthetic()
            row=first_row(packet,arm="P3",repeat=2,pose="noisy")
            row["identity"][field]=h("changed "+field)
            with self.assertRaisesRegex(ValueError,"Unpaired"):
                gate.evaluate(registration,packet)
        registration,packet=synthetic()
        first_row(packet,repeat=1)["repeat"]=2
        with self.assertRaises(ValueError):
            gate.evaluate(registration,packet)
        registration,packet=synthetic()
        registration["scenes"]["heads"]["noise_seed"]+=1
        packet["registration_sha256"]=gate.canonical_sha256(registration)
        with self.assertRaisesRegex(ValueError,"noise seed"):
            gate.evaluate(registration,packet)

    def test_same_state_must_use_same_endpoints_and_fresh_processes(self):
        registration,packet=synthetic()
        packet["rows"][1]["endpoint_sha256"]["final"]=h("wrong reference endpoint")
        with self.assertRaisesRegex(ValueError,"different state/endpoint"):
            gate.evaluate(registration,packet)
        registration,packet=synthetic()
        receipt=first_row(packet,repeat=0)["identity"]["process_receipt_sha256"]
        target=first_row(packet,repeat=1)["state_id"]
        for row in packet["rows"]:
            if row["state_id"]==target:
                row["identity"]["process_receipt_sha256"]=receipt
        with self.assertRaisesRegex(ValueError,"receipt reused"):
            gate.evaluate(registration,packet)

    def test_prospective_policy_and_registration_identity_cannot_drift(self):
        registration,packet=synthetic()
        registration["policy"]["minimum_primary_r0_gain"]="0.01"
        packet["registration_sha256"]=gate.canonical_sha256(registration)
        with self.assertRaisesRegex(ValueError,"policy"):
            gate.evaluate(registration,packet)
        registration,packet=synthetic()
        packet["registration_sha256"]=h("another registry")
        with self.assertRaisesRegex(ValueError,"another registration"):
            gate.evaluate(registration,packet)

    def test_input_unchanged_and_no_arrays_or_release_authorization_claim(self):
        registration,packet=synthetic()
        before=deepcopy((registration,packet))
        result=gate.evaluate(registration,packet)
        self.assertEqual((registration,packet),before)
        self.assertFalse(result["arrays_loaded"])
        self.assertFalse(result["execution_registration_and_sealing_implemented"])
        self.assertFalse(result["release_authorized"])
        self.assertEqual(result["input_packet_sha256"],gate.canonical_sha256(packet))
        json.dumps(result,allow_nan=False)


def run_fixtures():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(M3GateFixtures)
    result=unittest.TestResult()
    suite.run(result)
    if not result.wasSuccessful():
        raise AssertionError("\n".join(str(value) for value in result.errors+result.failures))
    return dict(pass_all=True,synthetic_only=True,tests_passed=result.testsRun,
                real_arrays_loaded=False,real_study_scores_read=False,gpu_used=False)


if __name__=="__main__":
    print(json.dumps(run_fixtures(),indent=2))
