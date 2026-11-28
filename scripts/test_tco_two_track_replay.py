"""Small CPU regression tests for complete paired inference units and the registered gates."""
import copy
import unittest

from replay_tco_two_track import (ALL, GEOMETRY, SEEDS, a_analysis, b_fresh, b_pilot, choose_a,
                                  compare, next_hybrid, validated_index)
from tco_two_track_protocol import MANIFEST, NEW_SEEDS, digest, read, registration


def packets(n):
    return [dict(scene=f"room{i}",input_sha256=f"input{i}",reference_sha256=f"ref{i}") for i in range(n)]


def row(p,arm,seed,v=1.,**extra):
    return dict(room=p["scene"],input_sha256=p["input_sha256"],reference_sha256=p["reference_sha256"],
                arm=arm,seed=seed,valid_points=999,accuracy_mm=v,completeness_mm=v,scene_seconds=v,
                peak_allocated_bytes=100,peak_reserved_bytes=120,**extra)


def a_rows(p,seeds):
    out=[]
    for room in p:
        for seed in seeds:
            for arm,steps in (("Refresh1",10),("Full10",10),("Full40",40),("Cache10",10)):
                r=row(room,arm,seed,steps=steps)
                if arm=="Refresh1":
                    r.update(accuracy_mm=1.01,completeness_mm=1.03,scene_seconds=.9)
                elif arm=="Full40":
                    r["scene_seconds"]=4.
                out.append(r)
    return out


def b_rows(p,seeds):
    out=[]
    for room in p:
        for ref in ("native","reprojected"):
            for s in seeds:
                out.append(row(room,ref,s))
        out += [row(room,"fusion",None,.99,seeds=seeds),row(room,"triplicate_anchor",seeds[0])]
    return out


class StudyTest(unittest.TestCase):
    def test_registration_integrity(self):
        reg=read(MANIFEST)
        self.assertEqual(reg,registration())
        self.assertEqual(len(reg["a"]["dev_jobs"]),45)
        self.assertEqual(len(set(reg["a"]["clean_order"])),45)
        self.assertEqual(len(reg["a"]["engineering_jobs"]),5)
        self.assertEqual(reg["a"]["clean_order"][:4],reg["a"]["pilot_ids"])
        self.assertEqual(reg["runtime"]["runpod_usd"],0)
        self.assertEqual(reg["fresh"]["primary"]["explicit_excluded_visits"][-1],381875)

    def test_thresholds_and_missing_reference(self):
        p=packets(6)
        rows=a_rows(p,NEW_SEEDS)
        def analyze(table):
            return a_analysis(table,p,NEW_SEEDS,["Refresh1"],["Full10","Full40","Cache10"],True,draws=100)
        x=analyze(rows)
        self.assertEqual(x["candidates"]["Refresh1"]["equal_step"]["verdict"],"pass")
        self.assertEqual(next_hybrid(x),"Narrow")
        self.assertEqual(choose_a(x),"Refresh1")
        for r in rows:
            if r["arm"]=="Refresh1":
                r["scene_seconds"]=1.
        self.assertEqual(analyze(rows)["candidates"]["Refresh1"]["equal_step"]["verdict"],"supported_violation")
        self.assertEqual(analyze(rows[:-1])["candidates"]["Refresh1"]["equal_step"]["verdict"],"insufficient_coverage")
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            analyze(rows + rows[:1])
        broken=copy.deepcopy(rows)
        broken[0]["input_sha256"]="mismatch"
        with self.assertRaisesRegex(ValueError,"Different input"):
            analyze(broken)
        broken=copy.deepcopy(rows)
        broken[0]["steps"]=9
        with self.assertRaisesRegex(ValueError,"Unexpected Adam"):
            analyze(broken)
        broken=copy.deepcopy(rows)
        broken[0]["valid_points"]=3
        with self.assertRaisesRegex(ValueError,"support"):
            analyze(broken)

    def test_room_weighting_and_pairing(self):
        p=packets(2)
        rows=a_rows(p,SEEDS)
        for r in rows:
            baseline=1 if r["room"]=="room0" else 100
            r["accuracy_mm"]=(SEEDS.index(r["seed"])+1)*baseline
            if r["arm"]=="Refresh1" and r["room"]=="room0":
                r["accuracy_mm"]*=2
        result=a_analysis(rows,p,SEEDS,["Refresh1"],["Full10","Full40","Cache10"],draws=300)
        m=result["candidates"]["Refresh1"]["comparisons"]["Full10"]["metrics"]["accuracy_mm"]
        self.assertEqual(m["ratio"],1.5)
        self.assertEqual(m["leave_one_room_out"],{"room0":1.,"room1":2.})
        self.assertEqual((m["lower_95"],m["upper_95"]),(1.,2.))

    def test_b_development_sham_missing_and_strict_superiority(self):
        p=packets(3)
        rows=b_rows(p,SEEDS)
        r=b_pilot(rows,p,SEEDS,draws=90)
        self.assertTrue(r["eligible"])
        self.assertEqual(r["rooms_with_lower_accuracy"],3)
        self.assertEqual(r["coverage"]["expected"],24)
        self.assertFalse(b_pilot(rows[:-1],p,SEEDS,draws=90)["eligible"])
        broken=copy.deepcopy(rows)
        next(r for r in broken if r["arm"]=="triplicate_anchor")["accuracy_mm"]+=.001
        self.assertFalse(b_pilot(broken,p,SEEDS,draws=90)["eligible"])
        for r in rows:
            if r["arm"]=="fusion":
                r["accuracy_mm"]=1.
        self.assertFalse(b_pilot(rows,p,SEEDS,draws=90)["eligible"])

    def test_triplet_unit_and_mean_native_control(self):
        p=packets(6)
        rows=[]
        for room in p:
            rows.append(row(room,"fusion",None,1.,seeds=NEW_SEEDS))
            rows.append(row(room,"Full120",NEW_SEEDS[0],steps=120))
            for s,v in zip(NEW_SEEDS,(1.,1.,4.)):
                rows.append(row(room,"Full40",s,v=v,steps=40))
        result=b_fresh(rows,p,NEW_SEEDS,"fusion","Full120",True,draws=90)
        self.assertTrue(result["primary"]["passed"])
        m=result["comparisons"]["Full40"]["metrics"]["accuracy_mm"]
        self.assertEqual((m["ratio"],m["lower_95"],m["upper_95"]),(.5,.5,.5))
        self.assertEqual(result["coverage"]["expected"],30)
        broken=copy.deepcopy(rows)
        broken[0]["seeds"]=[7,7,77]
        with self.assertRaisesRegex(ValueError,"fusion triplet"):
            b_fresh(broken,p,NEW_SEEDS,"fusion","Full120",True,draws=40)
        broken=copy.deepcopy(rows)
        broken.append(dict(broken[0],seed=7))
        with self.assertRaisesRegex(ValueError,"Unregistered"):
            b_fresh(broken,p,NEW_SEEDS,"fusion","Full120",True,draws=40)


if __name__=="__main__":
    unittest.main()
