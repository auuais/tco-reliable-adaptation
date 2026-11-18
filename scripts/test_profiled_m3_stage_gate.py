"""Pure synthetic equivalence/adversarial tests for the M3 early-cell gate."""
from copy import deepcopy
import itertools
import json
import unittest

import profiled_m3_gate as full
import profiled_m3_stage_gate as early
from test_profiled_m3_gate import synthetic, set_relative, h


CELL="earlier_centered_clean"


def fixture(nominee="P3"):
    registration,packet=synthetic(nominee)
    # The real archived reference uses one common validity mask for both labels.
    for scene in full.SCENES:
        shared=h(scene+"/shared_mask")
        for reference in full.REFERENCES:
            registration["references"][scene+"/"+reference]["mask_sha256"]=shared
    for row in packet["rows"]:
        row["identity"]["mask_sha256"]=registration["references"][row["scene"]+"/"+row["reference"]]["mask_sha256"]
    packet["registration_sha256"]=full.canonical_sha256(registration)
    stage_registration=deepcopy(registration)
    stage_registration["schema"]="profiled_m3_early_registration_v1"
    del stage_registration["provenance"]["prediction_seal_sha256"]
    return registration,stage_registration,packet


def cell_rows(packet,repeat,cell=CELL):
    return deepcopy([row for row in packet["rows"] if full.cell_id(row["side"],row["grid"],row["pose"])==cell and row["repeat"]<=repeat])


def descriptors(registration,repeat,cell=CELL):
    arms=tuple(sorted(full.CONTROLS+(registration["nominee"],)))
    result={}
    for r in range(repeat+1):
        cells=(["earlier_centered_clean","later_centered_clean"] if r==0 and cell in
               ("earlier_centered_clean","later_centered_clean") else [cell])
        result[str(r)]=dict(schema="profiled_m3_stage_seal_descriptor_v1",repeat=r,cells=cells,
            registration_sha256=full.canonical_sha256(registration),prediction_seal_sha256=h(str(r)+"/"+"/".join(cells)),
            state_ids=sorted(full.state_id(scene,*early.CELLS[scope],r,arm)
                             for scene,scope,arm in itertools.product(full.SCENES,cells,arms)),endpoints=list(full.ENDPOINTS))
    return result


def run_cell(registration,packet,repeat,cell=CELL):
    seals=descriptors(registration,repeat,cell)
    prior=early.evaluate_cell(registration,cell_rows(packet,0,cell),cell=cell,repeat=0,stage_seals={"0":seals["0"]}) if repeat else None
    return early.evaluate_cell(registration,cell_rows(packet,repeat,cell),cell=cell,repeat=repeat,stage_seals=seals,prior_r0=prior)


class M3StageGateFixtures(unittest.TestCase):
    def assert_equivalent(self,stage,complete):
        reference=next(cell for cell in complete["cells"] if cell["cell"]==stage["cell"])
        for index,decision in enumerate(stage["repetitions"]):
            common={key:value for key,value in decision.items() if key not in
                    ("matched_detached_delta_S","necessary_matched_detached_positive")}
            self.assertEqual(common,reference["repetitions"][index])
            self.assertEqual(decision["matched_detached_delta_S"],reference["mechanism"]["contrasts"][index]["delta_S"])
        self.assertEqual(stage["selected_control"],reference["selected_control"])
        self.assertEqual(stage["primary_r0_control_pool"],reference["primary_r0_control_pool"])
        if stage["repeat"]==2:
            self.assertEqual(stage["mechanism"],reference["mechanism"])
            self.assertEqual(stage["pass_precheck"],reference["pass_all"])
        else:
            expected=all(row["pass_all"] for row in reference["repetitions"][:stage["repeat"]+1]) and all(
                row["delta_S"]["value"]>0 for row in reference["mechanism"]["contrasts"][:stage["repeat"]+1])
            self.assertEqual(stage["pass_precheck"],expected)
            self.assertIsNone(stage["mechanism"])
            self.assertIsNone(stage["mechanism_confirmed"])

    def test_complete_cell_decisions_equal_full_gate_for_both_nominees(self):
        for nominee in ("P3","P4"):
            registration,stage_registration,packet=fixture(nominee)
            complete=full.evaluate(registration,packet)
            for repeat,cell in itertools.product(full.REPEATS,(CELL,"later_shifted_noisy")):
                result=run_cell(stage_registration,packet,repeat,cell)
                self.assert_equivalent(result,complete)
                self.assertEqual(result["observed_rows"],30*(repeat+1))
                self.assertFalse(result["full_study_pass"])
                self.assertFalse(result["rental_authorized"])
                self.assertFalse(result["launch_authorized"])
                self.assertNotIn("evaluation_receipt_sha256",result["cell_decision"])

    def test_same_normalized_scene_and_component_tables_as_full_gate(self):
        registration,stage_registration,packet=fixture()
        complete=full.evaluate(registration,packet)
        stage=run_cell(stage_registration,packet,1)
        expected=[row for row in complete["scene_table"] if row["cell"]==CELL and row["repeat"]<=1]
        sort=lambda row:(row["repeat"],row["scene"],row["arm"],row["reference"],row["endpoint"])
        self.assertEqual(sorted(stage["scene_table"],key=sort),sorted(expected,key=sort))
        expected=[row for row in complete["aggregate_table"] if row["cell"]==CELL and row["repeat"]<=1]
        self.assertEqual(stage["aggregate_table"],expected)

    def test_exact5percent_tie_and_small_positive_later_repeat(self):
        registration,stage_registration,packet=fixture()
        set_relative(packet,"P2",.8)
        set_relative(packet,"P3",.76)
        set_relative(packet,"P3",.799,repeat=1)
        complete=full.evaluate(registration,packet)
        for repeat in full.REPEATS:
            result=run_cell(stage_registration,packet,repeat)
            self.assert_equivalent(result,complete)
            self.assertEqual(result["selected_control"],"P1")
            self.assertEqual(result["repetitions"][0]["primary_gain"]["exact"],"1/20")
            self.assertTrue(result["pass_precheck"])

    def test_control_cannot_be_reselected_by_repeat_or_alternate_reference(self):
        registration,stage_registration,packet=fixture()
        set_relative(packet,"P5",.1,repeat=1)
        set_relative(packet,"P5",.05,reference="released")
        result=run_cell(stage_registration,packet,2)
        self.assertEqual(result["selected_control"],"P1")
        self.assert_equivalent(result,full.evaluate(registration,packet))

    def test_failed_safety_in_any_seen_repeat_is_retained(self):
        mutations=(lambda p:set_relative(p,"P3",.801,scene="fire",repeat=1),
                   lambda p:set_relative(p,"P3",.8160001,.5,repeat=1),
                   lambda p:set_relative(p,"P3",.5,.8160001,repeat=1),
                   lambda p:set_relative(p,"P3",.8001,repeat=1,reference="released"),
                   lambda p:set_relative(p,"P3",.8,repeat=1),
                   lambda p:set_relative(p,"P2",.69,repeat=1))
        for mutation in mutations:
            registration,stage_registration,packet=fixture()
            mutation(packet)
            complete=full.evaluate(registration,packet)
            for repeat in (1,2):
                result=run_cell(stage_registration,packet,repeat)
                self.assertFalse(result["pass_precheck"])
                self.assert_equivalent(result,complete)

    def test_validity_failure_disqualifies_without_dropping_shared_mask_pixels(self):
        registration,stage_registration,packet=fixture()
        for row in packet["rows"]:
            if row["scene"]=="heads" and row["arm"]=="P3" and row["repeat"]==1:
                row["validity"]["main_nonpositive_depth"]=1
        result=run_cell(stage_registration,packet,2)
        self.assertFalse(result["pass_precheck"])
        self.assert_equivalent(result,full.evaluate(registration,packet))
        self.assertEqual(result["observed_rows"],90)
        rows=cell_rows(packet,0)
        rows[0]["validity"]["evaluated_mask_count"]-=1
        with self.assertRaisesRegex(ValueError,"invalid pixels were dropped"):
            early.evaluate_cell(stage_registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(stage_registration,0))

    def test_full_execution_floor_and_strict_margin_only_at_r2(self):
        registration,stage_registration,packet=fixture()
        set_relative(packet,"P3",.799999,repeat=1)
        set_relative(packet,"P2",.9,repeat=2)
        complete=full.evaluate(registration,packet)
        for repeat in (0,1):
            result=run_cell(stage_registration,packet,repeat)
            self.assertTrue(result["pass_precheck"])
            self.assertIsNone(result["mechanism_confirmed"])
            self.assert_equivalent(result,complete)
        result=run_cell(stage_registration,packet,2)
        self.assertFalse(result["pass_precheck"])
        self.assertFalse(result["mechanism_confirmed"])
        self.assertEqual(result["mechanism"]["r0_contrast_minus_floor_and_margin"]["exact"],"0")
        self.assert_equivalent(result,complete)

    def test_positive_early_contrast_can_fail_later_absolute_margin(self):
        registration,stage_registration,packet=fixture()
        for arm in ("P1","P2","P5"):
            set_relative(packet,arm,.000001)
        set_relative(packet,"P3",.0000009)
        self.assertTrue(run_cell(stage_registration,packet,0)["pass_precheck"])
        self.assertTrue(run_cell(stage_registration,packet,1)["pass_precheck"])
        result=run_cell(stage_registration,packet,2)
        self.assertFalse(result["pass_precheck"])
        self.assert_equivalent(result,full.evaluate(registration,packet))

    def test_later_repeat_requires_exact_unrewritten_prior_r0(self):
        _,registration,packet=fixture()
        seals=descriptors(registration,1)
        rows=cell_rows(packet,1)
        with self.assertRaisesRegex(ValueError,"explicit prior-r0"):
            early.evaluate_cell(registration,rows,cell=CELL,repeat=1,stage_seals=seals)
        prior=run_cell(registration,packet,0)
        tampered=deepcopy(prior)
        tampered["selected_control"]="P2"
        tampered["cell_decision"]["selected_control"]="P2"
        del tampered["report_semantic_sha256"]
        tampered["report_semantic_sha256"]=full.canonical_sha256(tampered)
        with self.assertRaisesRegex(ValueError,"Prior-r0"):
            early.evaluate_cell(registration,rows,cell=CELL,repeat=1,stage_seals=seals,prior_r0=tampered)
        rows[0]["metrics_sha256"]=h("rewritten r0 metric source")
        with self.assertRaisesRegex(ValueError,"Prior-r0"):
            early.evaluate_cell(registration,rows,cell=CELL,repeat=1,stage_seals=seals,prior_r0=prior)

    def test_original_r0_seal_and_registration_remain_frozen(self):
        _,registration,packet=fixture()
        prior=run_cell(registration,packet,0)
        seals=descriptors(registration,1)
        seals["0"]["prediction_seal_sha256"]=h("replaced old seal")
        with self.assertRaisesRegex(ValueError,"Prior-r0"):
            early.evaluate_cell(registration,cell_rows(packet,1),cell=CELL,repeat=1,stage_seals=seals,prior_r0=prior)
        registration["provenance"]["evaluation_registry_sha256"]=h("changed evaluator")
        with self.assertRaisesRegex(ValueError,"registry"):
            early.evaluate_cell(registration,cell_rows(packet,1),cell=CELL,repeat=1,stage_seals=seals,prior_r0=prior)

    def test_repeat_stages_cannot_reuse_one_prediction_seal(self):
        _,registration,packet=fixture()
        prior=run_cell(registration,packet,0)
        seals=descriptors(registration,1)
        seals["1"]["prediction_seal_sha256"]=seals["0"]["prediction_seal_sha256"]
        with self.assertRaisesRegex(ValueError,"repeat stages"):
            early.evaluate_cell(registration,cell_rows(packet,1),cell=CELL,repeat=1,stage_seals=seals,prior_r0=prior)

    def test_failed_r0_cannot_authorize_later_repeat(self):
        _,registration,packet=fixture()
        set_relative(packet,"P3",.9,repeat=0)
        prior=run_cell(registration,packet,0)
        self.assertFalse(prior["pass_precheck"])
        with self.assertRaisesRegex(ValueError,"failed r0"):
            early.evaluate_cell(registration,cell_rows(packet,1),cell=CELL,repeat=1,
                                stage_seals=descriptors(registration,1),prior_r0=prior)

    def test_missing_duplicate_wrong_cell_and_wrong_reference_rows_rejected(self):
        _,registration,packet=fixture()
        mutations=(lambda rows:rows.pop(),lambda rows:rows.__setitem__(-1,deepcopy(rows[0])),
                   lambda rows:rows[0].__setitem__("side","later"),lambda rows:rows[0].__setitem__("reference","other"),
                   lambda rows:rows[0].__setitem__("repeat",1))
        for mutation in mutations:
            rows=cell_rows(packet,0)
            mutation(rows)
            with self.assertRaises(ValueError):
                early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(registration,0))

    def test_missing_stage_seal_or_declared_state_cannot_be_filled_in(self):
        _,registration,packet=fixture()
        for mutation in (lambda seals:seals.clear(),lambda seals:seals["0"]["state_ids"].pop(),
                         lambda seals:seals["0"].__setitem__("endpoints",["final"]),
                         lambda seals:seals["0"].__setitem__("repeat",2)):
            seals=descriptors(registration,0)
            mutation(seals)
            with self.assertRaises(ValueError):
                early.evaluate_cell(registration,cell_rows(packet,0),cell=CELL,repeat=0,stage_seals=seals)

    def test_shared_mask_across_refs_is_mandatory_in_registration_and_counts(self):
        _,registration,packet=fixture()
        registration["references"]["heads/released"]["mask_sha256"]=h("different mask")
        with self.assertRaisesRegex(ValueError,"same archived scene mask"):
            run_cell(registration,packet,0)
        _,registration,packet=fixture()
        rows=cell_rows(packet,0)
        rows[1]["validity"]["native_nonpositive_depth"]=1
        with self.assertRaisesRegex(ValueError,"validity counts differ"):
            early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(registration,0))

    def test_pairing_process_endpoint_and_input_hash_guards(self):
        for field in ("pose_sha256","packet_sha256","folds_sha256","native_rows_sha256","reference_sha256","mask_sha256","runtime_sha256"):
            _,registration,packet=fixture()
            rows=cell_rows(packet,0)
            rows[0]["identity"][field]=h("wrong identity")
            with self.assertRaisesRegex(ValueError,"Unpaired"):
                early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(registration,0))
        _,registration,packet=fixture()
        rows=cell_rows(packet,0)
        rows[1]["endpoint_sha256"]["final"]=h("other reference output")
        with self.assertRaisesRegex(ValueError,"different state/endpoint"):
            early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(registration,0))
        rows=cell_rows(packet,0)
        for row in rows[2:4]:
            row["identity"]["process_receipt_sha256"]=rows[0]["identity"]["process_receipt_sha256"]
        with self.assertRaisesRegex(ValueError,"Process receipt reused"):
            early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(registration,0))

    def test_zero_denominator_nonfinite_and_native_endpoint_reporting(self):
        for invalid in (0,float("inf"),float("nan")):
            _,registration,packet=fixture()
            rows=cell_rows(packet,0)
            rows[0]["final"]["accuracy"]=invalid
            with self.assertRaises(ValueError):
                early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=descriptors(registration,0))
        registration,stage_registration,packet=fixture()
        for row in packet["rows"]:
            if row["arm"]=="P3":
                row["native"]=dict(accuracy=1000.,completeness=1000.)
                row["validity"]["native_nonpositive_depth"]=1
        stage=run_cell(stage_registration,packet,0)
        self.assertTrue(stage["pass_precheck"])
        self.assert_equivalent(stage,full.evaluate(registration,packet))

    def test_order_invariance_no_input_mutation_or_future_global_seal_claim(self):
        _,registration,packet=fixture()
        rows=cell_rows(packet,0)
        seals=descriptors(registration,0)
        before=deepcopy((registration,rows,seals))
        result=early.evaluate_cell(registration,rows,cell=CELL,repeat=0,stage_seals=seals)
        self.assertEqual((registration,rows,seals),before)
        reordered=early.evaluate_cell(registration,list(reversed(rows)),cell=CELL,repeat=0,stage_seals=seals)
        self.assertEqual(result,reordered)
        self.assertNotIn("prediction_seal_sha256",registration["provenance"])
        self.assertFalse(result["external_byte_verification_performed"])
        self.assertFalse(result["arrays_loaded"])
        json.dumps(result,allow_nan=False)


def run_fixtures():
    result=unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(M3StageGateFixtures).run(result)
    if not result.wasSuccessful():
        raise AssertionError("\n".join(str(value) for value in result.errors+result.failures))
    return dict(pass_all=True,synthetic_only=True,tests_passed=result.testsRun,
        real_arrays_loaded=False,real_scores_read=False,gpu_used=False,launch_authorized=False)


if __name__=="__main__":
    print(json.dumps(run_fixtures(),indent=2))
