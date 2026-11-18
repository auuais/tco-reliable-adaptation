"""Synthetic pure inventory/stopping tests; no M2/M3 scores or arrays are read."""
from copy import deepcopy
import hashlib
import itertools
import json
import unittest

import profiled_m3_inventory as inventory


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_plan(nominee="P3"):
    nomination=dict(nominee=nominee,**{field:h(field) for field in inventory.NOMINATION_FIELDS})
    inputs=dict(plan_file=inventory.INPUT_PLAN,attempt=inventory.INPUT_ATTEMPT,plan_sha256=h("input plan"),
        preparation_receipt_sha256={key:h("preparation/"+key) for key in ("earlier","later","poses")},
        case_input_sha256={inventory.case_id(*parts):h("case/"+inventory.case_id(*parts))
                          for parts in itertools.product(inventory.SCENES,inventory.SIDES,inventory.GRIDS,inventory.POSES)},
        pose_matrix_sha256={"/".join(parts):h("pose/"+"/".join(parts))
                            for parts in itertools.product(inventory.SCENES,inventory.POSES)},
        **{field:h(field) for field in inventory.IDENTITIES})
    return inventory.build_plan(nomination,inputs)


def audit(plan,job,approved=True):
    checks={field:True for field in inventory.REUSE_CHECKS}
    if not approved:
        checks["strict_runtime_and_precision"]=False
    return dict(schema="profiled_m3_reuse_audit_v1",plan_sha256=plan["plan_sha256"],job_id=job["job_id"],
        m2_run_id=job["reuse"]["m2_run_id"],checks=checks,decision="approved" if approved else "rejected",
        audit_receipt_sha256=h(job["job_id"]+"/reuse-audit"),m2_completion_sha256=h(job["job_id"]+"/m2-complete"),
        native_sha256=h(job["job_id"]+"/native"),final_sha256=h(job["job_id"]+"/final"))


def complete(plan,job,origin="fresh"):
    return dict(schema="profiled_m3_state_completion_v1",plan_sha256=plan["plan_sha256"],job_id=job["job_id"],origin=origin,
        **{field:job[field] for field in ("input_identity_sha256","pose_identity_sha256",*inventory.IDENTITIES)},
        completion_receipt_sha256=h(job["job_id"]+("/m2-complete" if origin=="m2_reuse" else "/complete")),
        process_receipt_sha256=h(job["job_id"]+"/original-process"),
        native_sha256=h(job["job_id"]+"/native"),final_sha256=h(job["job_id"]+"/final"))


def stage_review(plan,stage,completed,failed_cell=None):
    jobs={job["job_id"]:job for job in plan["jobs"]}
    metrics=[]
    for job_id,reference in itertools.product(stage["job_ids"],inventory.REFERENCES):
        metrics.append(dict(job_id=job_id,reference=reference,
            final_metrics_sha256=h(job_id+"/"+reference+"/final-metrics"),
            native_metrics_sha256=h(job_id+"/"+reference+"/native-metrics"),
            fixed_mask_sha256=h(jobs[job_id]["scene"]+"/"+reference+"/fixed-mask"),
            final_endpoint_sha256=completed[job_id]["final_sha256"],
            native_endpoint_sha256=completed[job_id]["native_sha256"]))
    return dict(schema="profiled_m3_stage_review_v1",plan_sha256=plan["plan_sha256"],stage_id=stage["stage_id"],metrics=metrics,
        cells={cell:dict(status="failed" if failed_cell==cell else "passed_precheck",selected_control="P1",
            evaluation_receipt_sha256=h(stage["stage_id"]+"/"+cell+"/decision"),
            mechanism_confirmed=(failed_cell!=cell) if stage["repeat"]==2 else None) for cell in stage["cells"]})


def prefix(plan,count):
    completed,audits,reviews={},{},{}
    jobs={job["job_id"]:job for job in plan["jobs"]}
    for stage in plan["stages"][:count]:
        for job_id in stage["job_ids"]:
            job=jobs[job_id]
            if job["reuse"]["eligible"]:
                audits[job_id]=audit(plan,job)
            completed[job_id]=complete(plan,job,"m2_reuse" if job["reuse"]["eligible"] else "fresh")
        reviews[stage["stage_id"]]=stage_review(plan,stage,completed)
    return dict(completed=completed,reuse_audits=audits,stage_reviews=reviews)


class M3InventoryFixtures(unittest.TestCase):
    def test_both_nominees_have_exact_deterministic_factorial_inventory(self):
        for nominee in ("P3","P4"):
            plan=synthetic_plan(nominee)
            self.assertEqual(plan,synthetic_plan(nominee))
            jobs=plan["jobs"]
            self.assertEqual(len(jobs),360)
            self.assertEqual(len({job["job_id"] for job in jobs}),360)
            self.assertEqual(len(plan["stages"]),23)
            self.assertEqual(plan["required_reference_rows"],720)
            self.assertEqual(plan["required_endpoint_metrics"],1440)
            self.assertEqual(set(plan["arms"]),set(inventory.CONTROLS)|{nominee})
            self.assertEqual({job["repeat"] for job in jobs},{0,1,2})
            self.assertEqual(sum(job["reuse"]["eligible"] for job in jobs),30)
            self.assertTrue(all(job["reuse"]["status"]=="unresolved" for job in jobs if job["reuse"]["eligible"]))
            self.assertEqual(plan["new_states_if_all_reuse_is_approved"],330)

    def test_stopping_order_first_both_directions_then_other_r0_then_repeats(self):
        plan=synthetic_plan()
        stages=plan["stages"]
        self.assertEqual(stages[0]["cells"],["earlier_centered_clean","later_centered_clean"])
        self.assertEqual(stages[0]["required_states"],30)
        self.assertEqual(stages[0]["required_reference_rows"],60)
        self.assertEqual([row["repeat"] for row in stages],[0]*7+[1]*8+[2]*8)
        for index,stage in enumerate(stages):
            self.assertEqual(stage["requires_previous_review"],None if index==0 else stages[index-1]["stage_id"])
        for job in plan["jobs"]:
            self.assertIn("scientific_registration",job["dependencies"])
            self.assertFalse(job["command_scope"]["executable"])
            self.assertEqual(job["outputs"]["references"],list(inventory.REFERENCES))
            self.assertEqual(job["outputs"]["endpoints"],list(inventory.ENDPOINTS))

    def test_local_single_gpu_seed_lora_optimizer_and_pair_contract(self):
        plan=synthetic_plan()
        self.assertEqual(plan["resource"],dict(local_gpu_count=1,max_concurrent_jobs=1,rental_usd=0))
        for job in plan["jobs"]:
            train=job["training"]
            self.assertEqual(train["seed"],42)
            self.assertTrue(train["fresh_pretrained_process"])
            self.assertTrue(train["original_weights_frozen"])
            if job["arm"]=="P0":
                self.assertEqual(train["macrosteps"],0)
                self.assertFalse(train["lora_enabled"])
            else:
                self.assertEqual((train["rank"],train["alpha"],train["macrosteps"],train["ordered_pairs_per_macrostep"]),(4,1,8,20))
                self.assertEqual((train["optimizer"],train["learning_rate"],train["clip_gradient_norm"]),("Adam",.0001,1.))

    def test_matching_inputs_reused_across_arms_and_repetitions(self):
        plan=synthetic_plan()
        for case in plan["inputs"]["case_input_sha256"]:
            rows=[job for job in plan["jobs"] if job["case_id"]==case]
            self.assertEqual(len(rows),15)
            self.assertEqual({job["input_identity_sha256"] for job in rows},{plan["inputs"]["case_input_sha256"][case]})
            self.assertEqual(len({job["runtime_sha256"] for job in rows}),1)
        for scene,pose in itertools.product(inventory.SCENES,inventory.POSES):
            rows=[job for job in plan["jobs"] if job["scene"]==scene and job["pose"]==pose]
            self.assertEqual(len(rows),60)
            self.assertEqual({job["pose_identity_sha256"] for job in rows},
                             {plan["inputs"]["pose_matrix_sha256"][scene+"/"+pose]})

    def test_initial_progress_requests_reuse_audit_and_never_qualifies(self):
        plan=synthetic_plan()
        result=inventory.inspect_progress(plan)
        self.assertEqual(result["next_action"]["operation"],"audit_conditional_M2_reuse")
        self.assertEqual(len(result["conditional_unexecuted_job_ids"]),360)
        self.assertFalse(result["final_gate_passed"])
        self.assertFalse(result["launch_authorized"])
        self.assertFalse(result["partial_inventory_qualified"])

    def test_reuse_requires_all_exact_checks_and_original_endpoint_hashes(self):
        plan=synthetic_plan()
        job=plan["jobs"][0]
        audits={job["job_id"]:audit(plan,job)}
        result=inventory.inspect_progress(plan,reuse_audits=audits)
        self.assertEqual(result["next_action"]["operation"],"bind_verified_original_M2_receipt")
        completed={job["job_id"]:complete(plan,job,"m2_reuse")}
        self.assertEqual(inventory.inspect_progress(plan,reuse_audits=audits,completed=completed)["completed_states"],1)
        audits[job["job_id"]]["checks"]["strict_runtime_and_precision"]=False
        with self.assertRaisesRegex(ValueError,"verdict"):
            inventory.inspect_progress(plan,reuse_audits=audits,completed=completed)
        audits[job["job_id"]]=audit(plan,job)
        completed[job["job_id"]]["final_sha256"]=h("different endpoint")
        with self.assertRaisesRegex(ValueError,"original M2 evidence"):
            inventory.inspect_progress(plan,reuse_audits=audits,completed=completed)

    def test_rejected_reuse_plans_fresh_without_dropping_state(self):
        plan=synthetic_plan()
        job=plan["jobs"][0]
        audits={job["job_id"]:audit(plan,job,False)}
        result=inventory.inspect_progress(plan,reuse_audits=audits)
        self.assertEqual(result["next_action"]["operation"],"plan_one_fresh_local_process")
        completed={job["job_id"]:complete(plan,job)}
        self.assertEqual(inventory.inspect_progress(plan,reuse_audits=audits,completed=completed)["completed_states"],1)
        completed[job["job_id"]]["origin"]="m2_reuse"
        with self.assertRaisesRegex(ValueError,"approved"):
            inventory.inspect_progress(plan,reuse_audits=audits,completed=completed)

    def test_repeat_or_shifted_state_is_never_eligible_M2_reuse(self):
        plan=synthetic_plan()
        for job in (next(job for job in plan["jobs"] if job["repeat"]==1),
                    next(job for job in plan["jobs"] if job["grid"]=="shifted")):
            with self.assertRaisesRegex(ValueError,"Ineligible"):
                inventory.inspect_progress(plan,reuse_audits={job["job_id"]:audit(plan,job)})

    def test_initial_directional_failure_preserves_all330_omitted_states(self):
        plan=synthetic_plan()
        state=prefix(plan,1)
        first=plan["stages"][0]
        state["stage_reviews"][first["stage_id"]]=stage_review(plan,first,state["completed"],"later_centered_clean")
        result=inventory.inspect_progress(plan,**state)
        self.assertEqual(result["status"],"stopped_mandatory_gate_failure")
        self.assertEqual(result["completed_states"],30)
        self.assertEqual(len(result["conditional_unexecuted_job_ids"]),330)
        self.assertIn("later_centered_clean",result["next_action"]["cell_decisions"])
        self.assertFalse(result["final_gate_passed"])
        extra=plan["jobs"][30]
        state["completed"][extra["job_id"]]=complete(plan,extra)
        with self.assertRaisesRegex(ValueError,"conditional stop barrier"):
            inventory.inspect_progress(plan,**state)

    def test_stage_cannot_score_missing_state_reference_or_endpoint(self):
        plan=synthetic_plan()
        state=prefix(plan,1)
        del state["completed"][plan["jobs"][0]["job_id"]]
        with self.assertRaisesRegex(ValueError,"every paired completed state"):
            inventory.inspect_progress(plan,**state)
        state=prefix(plan,1)
        review=next(iter(state["stage_reviews"].values()))
        review["metrics"].pop()
        with self.assertRaisesRegex(ValueError,"paired stage reference"):
            inventory.inspect_progress(plan,**state)
        state=prefix(plan,1)
        review=next(iter(state["stage_reviews"].values()))
        review["metrics"][0]["native_endpoint_sha256"]=h("wrong endpoint")
        with self.assertRaisesRegex(ValueError,"different completed endpoint"):
            inventory.inspect_progress(plan,**state)

    def test_no_arm_specific_mask_or_wrong_reference_metric_join(self):
        plan=synthetic_plan()
        for mutation in (lambda row:row.__setitem__("fixed_mask_sha256",h("changed mask")),
                         lambda row:row.__setitem__("reference","unregistered")):
            state=prefix(plan,1)
            review=next(iter(state["stage_reviews"].values()))
            mutation(review["metrics"][2])
            with self.assertRaises(ValueError):
                inventory.inspect_progress(plan,**state)

    def test_primary_r0_control_persists_through_repeats(self):
        plan=synthetic_plan()
        state=prefix(plan,8)
        result=inventory.inspect_progress(plan,**state)
        self.assertEqual(set(result["frozen_r0_controls"].values()),{"P1"})
        repeat_stage=plan["stages"][7]
        decision=next(iter(state["stage_reviews"][repeat_stage["stage_id"]]["cells"].values()))
        decision["selected_control"]="P2"
        with self.assertRaisesRegex(ValueError,"control identity changed"):
            inventory.inspect_progress(plan,**state)

    def test_same_input_and_unique_process_receipts_are_required(self):
        plan=synthetic_plan()
        for field in ("input_identity_sha256","pose_identity_sha256",*inventory.IDENTITIES):
            state=prefix(plan,1)
            first=plan["jobs"][0]["job_id"]
            state["completed"][first][field]=h("changed "+field)
            with self.assertRaisesRegex(ValueError,"paired inputs/runtime/endpoint"):
                inventory.inspect_progress(plan,**state)
        state=prefix(plan,1)
        one,two=(plan["jobs"][index]["job_id"] for index in (0,1))
        state["completed"][two]["process_receipt_sha256"]=state["completed"][one]["process_receipt_sha256"]
        with self.assertRaisesRegex(ValueError,"process receipt reused"):
            inventory.inspect_progress(plan,**state)

    def test_unknown_failure_requires_review_and_numerical_failure_stops(self):
        plan=synthetic_plan()
        job=plan["jobs"][0]
        for classification,status in (("unclassified","failure_review_required"),
                ("infrastructure","failure_review_required"),("numerical","stopped_numerical_failure")):
            receipt=dict(schema="profiled_m3_failure_receipt_v1",plan_sha256=plan["plan_sha256"],job_id=job["job_id"],
                classification=classification,failure_receipt_sha256=h("preserved failure"))
            result=inventory.inspect_progress(plan,failed={job["job_id"]:receipt})
            self.assertEqual(result["status"],status)
            self.assertEqual(result["failed_states"],1)
            self.assertEqual(len(result["conditional_unexecuted_job_ids"]),359)
            self.assertEqual(result["preserved_failures"][job["job_id"]],receipt)

    def test_no_failure_retry_can_silently_replace_earlier_attempt(self):
        plan=synthetic_plan()
        job=plan["jobs"][0]
        receipt=dict(schema="profiled_m3_failure_receipt_v1",plan_sha256=plan["plan_sha256"],job_id=job["job_id"],
            classification="unclassified",failure_receipt_sha256=h("failure"))
        with self.assertRaisesRegex(ValueError,"silently"):
            inventory.inspect_progress(plan,failed={job["job_id"]:receipt},completed={job["job_id"]:complete(plan,job)})

    def test_unregistered_nominee_changed_policy_and_wrong_input_plan_rejected(self):
        plan=synthetic_plan()
        nomination=deepcopy(plan["nomination"])
        nomination["nominee"]="P5"
        with self.assertRaises(ValueError):
            inventory.build_plan(nomination,plan["inputs"])
        inputs=deepcopy(plan["inputs"])
        inputs["attempt"]="inputs01"
        with self.assertRaisesRegex(ValueError,"inputs02"):
            inventory.build_plan(plan["nomination"],inputs)
        plan["jobs"][0]["training"]["seed"]=7
        with self.assertRaisesRegex(ValueError,"inventory or policy changed"):
            inventory.inspect_progress(plan)

    def test_single_gpu_dependency_order_rejects_future_fresh_state(self):
        plan=synthetic_plan()
        job=plan["jobs"][1]
        with self.assertRaisesRegex(ValueError,"serial dependency"):
            inventory.inspect_progress(plan,completed={job["job_id"]:complete(plan,job)},
                reuse_audits={job["job_id"]:audit(plan,job,False)})

    def test_complete_staged_inventory_still_requires_full_gate_and_registration(self):
        plan=synthetic_plan()
        state=prefix(plan,len(plan["stages"]))
        before=deepcopy((plan,state))
        result=inventory.inspect_progress(plan,**state)
        self.assertEqual((plan,state),before)
        self.assertEqual(result["status"],"ready_for_full_inventory_gate")
        self.assertTrue(result["all_model_states_completed"])
        self.assertEqual(result["completed_states"],360)
        self.assertEqual(result["conditional_unexecuted_job_ids"],[])
        self.assertFalse(result["final_gate_passed"])
        self.assertFalse(result["launch_authorized"])
        last=plan["stages"][-1]
        decision=next(iter(state["stage_reviews"][last["stage_id"]]["cells"].values()))
        decision["mechanism_confirmed"]=False
        with self.assertRaisesRegex(ValueError,"three-repeat mechanism"):
            inventory.inspect_progress(plan,**state)


def run_fixtures():
    result=unittest.TestResult()
    unittest.defaultTestLoader.loadTestsFromTestCase(M3InventoryFixtures).run(result)
    if not result.wasSuccessful():
        raise AssertionError("\n".join(str(value) for value in result.errors+result.failures))
    return dict(pass_all=True,tests_passed=result.testsRun,synthetic_only=True,
        real_scores_read=False,real_arrays_opened=False,gpu_used=False,launch_authorized=False)


if __name__=="__main__":
    print(json.dumps(run_fixtures(),indent=2))
