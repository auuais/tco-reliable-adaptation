"""Synthetic final decision gates and public reporting checks for LayerNorm TCO."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from finalize_tco_final_ln import (IMAGE, candidate_to_replicate, compact, comparison_table, interpretation,
                                   selected_branch, validate_cost_and_coverage)
from replay_tco_final_ln import analyze
from test_tco_final_ln_replay import inputs


class TestFinalization(unittest.TestCase):
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
