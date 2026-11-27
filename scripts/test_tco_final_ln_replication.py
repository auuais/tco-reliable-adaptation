"""Offline checks for replication advancement, site identity, and the no-download path."""
import sys
import unittest
from unittest.mock import patch

import lock_tco_final_ln_replication as roster


class ReplicationTest(unittest.TestCase):
    def test_native_pass_keeps_ln10(self):
        primary = dict(dataset="stanford12", formal_coverage=True, joint_verdict="pass")
        follow = dict(dataset="development", branch="all_pass", analyses={k:{} for k in
                      ("LN10", "LN10_scale", "LN10_bias")})
        self.assertEqual(roster.gate_branch(primary, {"branch":"all_pass"}, follow), ("LN10","L10"))

    def test_conditional_gate(self):
        for equal, candidate, short, branch in [(True,"LN20","L20","equal_update_pass_only"),
                                                (False,"LN10_low","L10","other")]:
            primary = dict(dataset="stanford12", formal_coverage=True, joint_verdict="inconclusive",
                           decisions_on_complete_rooms=dict(accuracy_superiority_vs_short="pass" if equal else "inconclusive",
                                                            completeness_vs_short="pass"))
            advance = dict(candidate=candidate, eligible=True, coverage_complete=True, preregistered_development_rooms=True)
            points = dict(better_accuracy_vs_short=True, accuracy_vs_full=True, completeness_vs_full=True, faster_vs_full=True)
            follow = dict(dataset="development", independent_replication_gate=advance, point_gates=points)
            self.assertEqual(roster.gate_branch(primary, {"branch":branch}, follow), (candidate,short))
            points["accuracy_vs_full"] = False
            with self.assertRaisesRegex(ValueError, "Positive label"):
                roster.gate_branch(primary, {"branch":branch}, follow)
            advance["eligible"] = False
            self.assertIsNone(roster.gate_branch(primary, {"branch":branch}, follow))

    def test_identity_deduplicates_transitively(self):
        metadata = [dict(visit_id=i, video_id=i+900, venue_id=k) for i,k in [(100,"a"),(101,"a"),(102,"b"),(103,"NA")]]
        lasers = [dict(visit_id=i, laser_scanner_point_clouds_id="9001") for i in (101,102)]
        groups = roster.site_groups(metadata, lasers)
        self.assertEqual([groups.get(i,i) for i in (100,101,102,103)], [100,100,100,103])
        splits = [dict(fold="Validation",visit_id=v,video_id=s) for v,s in [(501,39),(99,30),("NA",21),(99,26),(501,39)]]
        self.assertEqual(roster.ranked_visits(splits), [(99,[26,30]),(501,[39])])
        with self.assertRaisesRegex(ValueError, "multiple visits"):
            roster.ranked_visits(splits + [dict(fold="Validation",visit_id=30,video_id=39)])

    def test_pending_and_negative_never_acquire(self):
        for state in ("pending","not_promoted"):
            with (patch.object(sys,"argv",["lock_tco_final_ln_replication.py","prepare"]),
                  patch.object(roster,"gate",return_value={"status":state}),
                  patch.object(roster,"prepare") as acquire):
                roster.main()
                acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
