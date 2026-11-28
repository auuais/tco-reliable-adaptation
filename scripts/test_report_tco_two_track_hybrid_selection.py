"""Exact bootstrap and stage decision checks with small synthetic paired score rows."""
from collections import Counter
from itertools import product
from statistics import median
import unittest

from report_tco_two_track_hybrid_selection import compare, percentiles, room_distribution, seed_distribution


class SelectionReplayTest(unittest.TestCase):
    def test_resampling_preserves_pairs(self):
        a,b=[1,5,9],[12,4,2]
        weights=seed_distribution(a,b)
        # Seven of the 27 ordered resamples repeat each seed at least twice.
        # The remaining six use all three seeds and compare their medians.
        expected=Counter()
        for x,y in zip(a,b):
            expected[x/y]+=7
        expected[median(a)/median(b)]+=6
        self.assertEqual(weights,expected)
        self.assertEqual(seed_distribution([2,4,8],[1,2,4]),{2.:27})

    def test_exact_room_distribution_against_direct_enumeration(self):
        a=Counter({.5:7,1.:20})
        b=Counter({.75:21,1.5:6})
        one=list(a.elements())+list(b.elements())
        brute=sorted((i+j)/2 for i,j in product(one,repeat=2))
        actual=room_distribution([a,b])
        self.assertEqual(sum(actual.values()),54**2)
        self.assertEqual(actual,Counter(brute))
        self.assertEqual(percentiles(actual),[brute[int(.05*(len(brute)-1))],brute[int(.95*(len(brute)-1))]])

    def test_inclusive_error_margins_strict_time_and_failed_seed(self):
        rooms=["office/seq-02","courtyard"]
        seeds=[7,77,777]
        rows={}
        for room,seed,arm in product(rooms,seeds,("Refresh1","Full10")):
            record=dict(room=room,seed=seed,status="completed",valid_points=8,
                        input_sha256=room,reference_sha256=room,dataset="data",first_ten_view_permutations=[],
                        scene_seconds=10.,peak_allocated_bytes=100.,accuracy_mm=100.,completeness_mm=100.)
            if arm=="Refresh1":
                record.update(accuracy_mm=101.,completeness_mm=103.,scene_seconds=9.)
            rows[room,seed,arm]=record
        base=compare(rows,rooms,seeds,"Refresh1","Full10")
        self.assertTrue(base["gate"]["passed"])
        self.assertEqual(base["aggregate"]["accuracy_mm"]["ratio"],1.01)
        for r in rows.values():
            r["scene_seconds"]=10.
        self.assertFalse(compare(rows,rooms,seeds,"Refresh1","Full10")["gate"]["passed"])
        rows["courtyard",7,"Full10"]["status"]="timeout"
        result=compare(rows,rooms,seeds,"Refresh1","Full10")
        self.assertEqual((result["complete_rooms"],result["paired_slots"],result["expected_pairs"]),(1,5,6))
        self.assertIsNone(result["gate"])
        self.assertEqual(result["rooms"][1]["missing_or_failed"][0]["reference"],"timeout")


if __name__=="__main__":
    unittest.main()
