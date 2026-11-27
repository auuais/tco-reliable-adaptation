"""Decision boundaries, paired units, retained-gain anchors, and receipt corruption tests."""

from copy import deepcopy
import unittest

from replay_tco_final_ln import analyze, grade
from score_tco_final_ln import check_optimizer, expected_names


def inputs():
    plan = dict(dataset="stanford12", seeds=[42,43,44], rooms=[
        dict(room=f"{'apt2' if i<2 else 'office1'}/r{i}",environment="apt2" if i<2 else "office1")
        for i in range(6)])
    rows = []
    for room in plan["rooms"]:
        for seed in plan["seeds"]:
            for arm, acc, comp, timer in (("L10",105,105,12),("L40",100,100,30),("LN10",101,103,10)):
                rows.append(dict(room=room["room"], seed=seed, arm=arm, accuracy_mm=acc,
                                 completeness_mm=comp, scene_seconds=timer, peak_allocated_bytes=100,
                                 peak_reserved_bytes=110, input_sha256="fixed", reference_sha256="ref"))
    return plan, rows


class TestReplay(unittest.TestCase):
    def test_exact_limits_and_superiority(self):
        plan, rows = inputs()
        result = analyze(plan, rows, draws=300)
        self.assertEqual(result["joint_verdict"], "pass")
        self.assertEqual(result["scored"],54)
        equal = dict(ratio=1.,lower_95=1.,upper_95=1.)
        self.assertEqual(grade(equal,1.,True),"supported_violation")
        self.assertEqual(grade(equal,1.),"pass")
        self.assertEqual(grade(dict(ratio=1.,lower_95=.99,upper_95=1.04),1.03),"inconclusive")
        self.assertEqual(grade(dict(ratio=1.04,lower_95=1.0,upper_95=1.07),1.03),"observed_violation_uncertain")

    def test_missing_arms_and_site_coverage(self):
        plan, rows = inputs()
        result = analyze(plan, rows[:-1], draws=90)
        self.assertEqual(result["joint_verdict"],"insufficient_coverage")
        self.assertEqual(len(result["missing"]),1)
        self.assertEqual(len(result["complete_rooms"]),5)
        plan["rooms"] = plan["rooms"][:-1]
        result = analyze(plan,rows[:-9],draws=90)
        self.assertFalse(result["formal_coverage"])
        self.assertEqual(result["missing"],[])
        bad = deepcopy(rows[:-9])
        bad[-1]["status"] = "timeout"
        result = analyze(plan, bad, draws=90)
        self.assertEqual(len(result["failures"]),1)

    def test_duplication_hash_and_cross_cohort(self):
        plan, rows = inputs()
        with self.assertRaises(ValueError):
            analyze(plan, rows+[rows[0]],draws=10)
        altered = deepcopy(rows)
        altered[-1]["input_sha256"] = "from a different window"
        with self.assertRaises(ValueError):
            analyze(plan,altered,draws=10)
        rows[0]["role"] = "development"
        with self.assertRaises(ValueError):
            analyze(plan,rows,draws=10)

    def test_anchor_and_duplicate_outside_primary(self):
        plan, rows = inputs()
        room = rows[0]["room"]
        rows.append(dict(room=room,role="anchor",arm="L0",accuracy_mm=110,completeness_mm=90))
        rows.append(dict(rows[0],role="duplicate",accuracy_mm=10000))
        result = analyze(plan,rows,draws=50)
        gain = result["retained_gain"][room]
        self.assertAlmostEqual(gain["accuracy_mm"]["fraction"],.9)
        self.assertIsNone(gain["completeness_mm"]["fraction"])
        self.assertEqual(result["excluded_roles"]["duplicate"],1)
        self.assertEqual(result["joint_verdict"],"pass")

    def test_within_environment_paired_seeds(self):
        plan, rows = inputs()
        for row in rows:
            # If an arm or a seed gets permuted independently, constant strata ratios will spread.
            src = row["seed"] - 41
            ratio = 0.25 if row["room"].startswith("apt2") else 2.5
            row["accuracy_mm"] = src * (ratio if row["arm"] == "LN10" else 1)
        out = analyze(plan,rows,draws=400)["comparisons"]["L40"]["accuracy_mm"]
        self.assertEqual(out["ratio"],1.75)
        self.assertEqual(out["upper_95"],1.75)
        self.assertEqual(out["lower_95"],1.75)
        self.assertEqual(out["environment_ratios"],{"apt2":.25,"office1":2.5})

    def test_followup_gate_uses_fresh_candidate_and_both_development_controls(self):
        plan, rows = inputs()
        room_map = {r["room"]: new for r,new in zip(plan["rooms"], ("heads/seq-01","chess/seq-03"))}
        plan = dict(dataset="development",seeds=[42,43,44],rooms=[
            dict(room=v, environment=v.split("/")[0]) for v in room_map.values()])
        follow=[]
        for row in rows:
            if row["room"] in room_map:
                arm={"LN10":"LN20","L10":"L20","L40":"L40"}[row["arm"]]
                follow.append(dict(row,role="development" if arm=="L40" else "followup",arm=arm,room=room_map[row["room"]]))
        result=analyze(plan,follow,"LN20","L20","L40",draws=100)
        self.assertEqual(result["joint_verdict"],"development")
        self.assertTrue(result["independent_replication_gate"]["eligible"])
        failed=analyze(plan,follow[:-1],"LN20","L20","L40",draws=100)
        self.assertFalse(failed["independent_replication_gate"]["eligible"])
        hurt=[dict(row, completeness_mm=106) if row["arm"]=="LN20" else row for row in follow]
        self.assertFalse(analyze(plan,hurt,"LN20","L20","L40",draws=100)["independent_replication_gate"]["eligible"])


def positive_receipt():
    lora, ln = expected_names("both",10)
    before = {name:"init" for name in ln}
    after = {name:"end" for name in ln}
    updates = [dict(update=i,counter_min=i,counter_max=i,lora_count=384,ln_count=4) for i in range(1,11)]
    opt = dict(lora_names=sorted(lora), ln_names=sorted(ln), named_counts={k:10 for k in lora|ln},
               frozen_mutations=[], steps=updates, ln_initial_sha256=before,ln_final_sha256=after)
    return dict(steps=10, completed_adam_updates=10, ln_policy="both", ln_lr=3e-5, optimizer=opt)


class TestOptimizerReceipt(unittest.TestCase):
    def test_registered_exact_names_and_counts(self):
        r = positive_receipt()
        check_optimizer(r)
        self.assertEqual(len(r["optimizer"]["lora_names"]),384)
        old = r["optimizer"]["ln_names"][0]
        r["optimizer"]["ln_names"][0] = old.replace("23","22")
        with self.assertRaises(ValueError):
            check_optimizer(r)

    def test_no_skip_update_or_amend_other_weight(self):
        for key in ("counter","step","frozen","stationary"):
            r = positive_receipt()
            opt = r["optimizer"]
            if key == "counter":
                opt["named_counts"][opt["ln_names"][-1]] = 9
            elif key == "step":
                opt["steps"].pop()
            elif key == "frozen":
                opt["frozen_mutations"].append("camera_head.weight")
            else:
                name = opt["ln_names"][-1]
                opt["ln_final_sha256"][name] = opt["ln_initial_sha256"][name]
            with self.assertRaises(ValueError):
                check_optimizer(r)

    def test_zero_lr_does_not_disable_lora(self):
        r = positive_receipt()
        r["ln_lr"] = 0
        r["optimizer"]["ln_final_sha256"] = r["optimizer"]["ln_initial_sha256"].copy()
        check_optimizer(r)
        r["optimizer"]["named_counts"][r["optimizer"]["lora_names"][0]] = 0
        with self.assertRaises(ValueError):
            check_optimizer(r)


if __name__ == "__main__":
    unittest.main()
