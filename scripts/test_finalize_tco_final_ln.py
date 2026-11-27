"""Synthetic final decision gates and public reporting checks for LayerNorm TCO."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finalize_tco_final_ln import (IMAGE, PROTOCOL, candidate_to_replicate, compact, comparison_table, interpretation,
                                   check_development_diagnostics, check_public_rescores,
                                   raw_rescore_record, same_seed_repeats, selected_branch, sha, validate_cost_and_coverage)
from replay_tco_final_ln import analyze
from test_tco_final_ln_replay import inputs


class TestFinalization(unittest.TestCase):
    def test_repeat_deltas_reversal_and_unavailable_pair(self):
        rows,jobs={},[]
        for arm,aa,bb in (("L10",10.,11.),("LN10",9.,12.),("L40",8.,7.)):
            seed=dict(room="chess/seq-03",seed=42,arm=arm,environment="chess",profile="clean",
                      input_sha256="input",reference_sha256="ref")
            steps=40 if arm=="L40" else 10
            jobs.append(dict(seed,id="duplicate_"+arm,steps=steps,role="duplicate"))
            for role,x in (("development",aa),("duplicate",bb)):
                key=role+"_"+arm
                rows[key]=dict(seed,id=key,role=role,status="completed",valid_points=17,adam_updates=steps,
                               predictions_sha256=key,accuracy_mm=x,completeness_mm=x+1)
        result=same_seed_repeats(rows,jobs)
        self.assertEqual(result["matched_pairs"],3)
        self.assertEqual(result["by_arm"]["LN10"]["metrics"]["accuracy_mm"]["delta_mm"],3.)
        self.assertTrue(result["ln10_vs_l10_percent"]["accuracy_mm"]["ordering_reversed"])
        rows["duplicate_L40"]["status"]="oom"
        limited=same_seed_repeats(rows,jobs)
        self.assertEqual(limited["matched_pairs"],2)
        self.assertNotIn("metrics",limited["by_arm"]["L40"])
        rows["duplicate_LN10"]["input_sha256"]="different"
        with self.assertRaisesRegex(ValueError, "identical input"):
            same_seed_repeats(rows,jobs)

    def test_independent_scores_authenticate_same_prediction_and_report_missing_selection(self):
        original_job=dict(id="a",role="primary",environment="apt2",room="apt2/one",seed=42,arm="LN10",
                          input_sha256="input",reference_sha256="ref",reference_file_sha256="ref_file",
                          input_file_sha256="inputs_file")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=root/"model"
            model.mkdir()
            saved=model/"prediction.npz"
            saved.write_bytes(b"test-prediction")
            (model/"complete.json").write_text(json.dumps({"predictions_sha256":sha(saved)}))
            first,second=(root/"scores"/kind/"a"/"score.json" for kind in ("first","independent"))
            first.parent.mkdir(parents=True); second.parent.mkdir(parents=True)
            score=dict(room="apt2/one",arm="LN10",seed=42,scoring_cpu=True,model_file_sha256="inputs_file",
                       input_sha256="input",reference_file_sha256="ref_file",reference_sha256="ref",
                       predictions_sha256=sha(saved),model_receipt_sha256=sha(model/"complete.json"),
                       valid_points=10,evaluator_sha256={"evaluator":"pinned"},protocol=PROTOCOL,accuracy_mm=1.,completeness_mm=2.)
            first.write_text(json.dumps(score))
            diff={"accuracy_mm":0.,"completeness_mm":0.}
            second.write_text(json.dumps(dict(score,first_score_sha256=sha(first),independent_rescore_mm_difference=diff)))
            actual=raw_rescore_record(model,root/"scores",original_job,{"evaluator":"pinned"})
            rows={"a":dict(score,status="completed",score_sha256=sha(first),rescore_sha256=sha(second),
                           rescore_mm_differences=diff),"b":{"status":"oom"}}
            jobs=[original_job,dict(original_job,id="b",arm="L40")]
            public=check_public_rescores(jobs,rows,{"a":actual})
            self.assertEqual((public["prespecified"],public["verified"],public["missing"]),(2,1,["b"]))
            bad=deepcopy(actual); bad["independent_mm"]["accuracy_mm"]+=.1
            with self.assertRaises(ValueError):
                check_public_rescores(jobs,rows,{"a":bad})
            saved.write_bytes(b"different-prediction")
            with self.assertRaises(ValueError):
                raw_rescore_record(model,root/"scores",original_job,{"evaluator":"pinned"})

    def test_diagnostics_cannot_pass_empty_and_can_record_failed_arms(self):
        job=dict(id="diag",room="heads",arm="LN10",seed=42,role="diagnostics",steps=2)
        both={f"aggregator.global_blocks.23.norm{i}.{k}":.1 for i in (1,2) for k in ("weight","bias")}
        scored={"diag":dict(job,status="completed",accuracy_mm=2.,completeness_mm=3.)}
        dev={arm:dict(room="heads",arm=arm,role="development",seed=42,status="completed",accuracy_mm=1.,
                      completeness_mm=2.) for arm in ("L10","LN10")}
        pair=dict(accuracy_delta_mm=1.,completeness_delta_mm=1.,positive_pixels=10,depth_scale_ratio=1.)
        d=dict(job,surface_accuracy_mm=2.,surface_completeness_mm=3.,loss_by_step=[.3,.2],
               preclip_l2={"ln":[.1,.3],"lora":[2.,.2]},times_clipped={"ln":0,"lora":1},
               comparisons={"L10":pair,"LN10":pair},diagnostic_only=True,ln_displacement_from_initial_by_step=[both,both])
        result=check_development_diagnostics(dict(rows=[d]),[job],scored,dev)
        self.assertAlmostEqual(result["original_lr_probes"]["heads"]["ln_joint_final_displacement"],.2)
        with self.assertRaises(ValueError):
            check_development_diagnostics({"rows":[]},[job],scored,dev)
        bad=dict(d,ln_displacement_from_initial_by_step=[])
        with self.assertRaises(ValueError):
            check_development_diagnostics({"rows":[bad]},[job],scored,dev)
        self.assertEqual(check_development_diagnostics({"rows":[dict(id="diag",status="oom")]},[job],
                         {"diag":{"status":"oom"}},dev)["missing"],["diag"])

    def test_original_candidate_survives_successful_primary(self):
        plan, rows = inputs()
        result = analyze(plan,rows,draws=90)
        self.assertEqual(result["joint_verdict"],"pass")
        self.assertEqual(selected_branch(result),"all_pass")
        components = {"analyses":{k:{} for k in ("LN10","LN10_scale","LN10_bias")}}
        self.assertEqual(candidate_to_replicate("all_pass",components),("LN10","L10"))
        self.assertIn("independent ARKitScenes visits",interpretation(result,result))
        self.assertIn("did not establish",interpretation(result,None))
        primary = compact(result)
        self.assertEqual(set(primary["mean_room_medians"]), {"L10","LN10","L40"})
        self.assertTrue(any("Accuracy (mm)" in x for x in comparison_table(primary)))

    def test_missing_accuracy_superiority_and_failed_redesign(self):
        plan, rows = inputs()
        for r in rows:
            if r["arm"]=="LN10":
                r["accuracy_mm"]=300.
        result=analyze(plan,rows,draws=90)
        self.assertEqual(result["joint_verdict"],"supported_violation")
        self.assertEqual(selected_branch(result),"other")
        gate=dict(candidate="LN10_low",coverage_complete=True,preregistered_development_rooms=True,eligible=False)
        follow=dict(independent_replication_gate=gate,point_gates=dict(better_accuracy_vs_short=False,accuracy_vs_full=False,
                                                                        completeness_vs_full=True,faster_vs_full=True))
        self.assertIsNone(candidate_to_replicate("other",follow))
        contradictory=deepcopy(follow)
        contradictory["independent_replication_gate"]["eligible"]=True
        with self.assertRaises(ValueError):
            candidate_to_replicate("other",contradictory)
        self.assertIn("violation",interpretation(result,None))

    def test_twenty_step_branch_and_changed_eligible_candidate(self):
        primary=dict(formal_coverage=True,joint_verdict="inconclusive",decisions_on_complete_rooms={
            "accuracy_superiority_vs_short":"pass","completeness_vs_short":"pass","accuracy_vs_full":"inconclusive"})
        branch=selected_branch(primary)
        self.assertEqual(branch,"equal_update_pass_only")
        condition=dict(candidate="LN20",coverage_complete=True,preregistered_development_rooms=True,eligible=True)
        follow=dict(independent_replication_gate=condition,point_gates=dict(better_accuracy_vs_short=True,accuracy_vs_full=True,
                                                                          completeness_vs_full=True,faster_vs_full=True))
        self.assertEqual(candidate_to_replicate(branch,follow),("LN20","L20"))
        self.assertIn("inconclusive",interpretation(primary,{"joint_verdict":"pass"}))

    def test_unrecovered_failure_retry_and_cost_accounting(self):
        def attempt(n,status):
            return dict(attempt=n,status=status,stopped=True,terminal_sha256=f"term{n}",start_sha256=f"start{n}",
                        exit_code=0 if status=="completed" else 1,wall_seconds=5.5,
                        complete_sha256="done" if status=="completed" else None,start_cost_usd=0,terminal_cost_usd=0)
        jobs=[dict(id="a",role="primary"),dict(id="b",role="primary")]
        a=[attempt(1,"timeout"),attempt(2,"completed")]
        b=[attempt(1,"oom")]
        rows={"a":dict(attempts=a,status="completed",model_receipt_sha256="done"),
              "b":dict(attempts=b,status="oom")}
        audit=dict(cases={"a":a,"b":b},image=IMAGE,registration_sha256="manifest",
                   scoring_batches=[dict(stopped=True,rental_compute_usd=0,exit_code=0)])
        with patch("finalize_tco_final_ln.sha", return_value="manifest"):
            r=validate_cost_and_coverage(jobs,rows,audit,{})
            self.assertEqual((r["slots"],r["completed"],r["attempts"]),(2,1,3))
            self.assertEqual([f["recovered"] for f in r["failures"]],[True,False])
            for bad in ("cost","roster","retry"):
                altered=deepcopy(audit)
                if bad=="cost":
                    altered["cases"]["a"][0]["start_cost_usd"]=.01
                elif bad=="roster":
                    del altered["cases"]["b"]
                else:
                    altered["cases"]["a"][0]["status"]="oom"
                with self.assertRaises(ValueError):
                    validate_cost_and_coverage(jobs,rows,altered,{})


if __name__=="__main__":
    unittest.main()
