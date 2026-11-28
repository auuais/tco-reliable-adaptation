"""Small CPU-only fixtures for timing locks, singleton readouts, and archival accounting."""
import unittest

from plan_tco_two_track_comparators import jobs_for, select_count
from summarize_tco_two_track_controls import capacity, make_packets


class Controls(unittest.TestCase):
    def test_affordable_boundary_and_median(self):
        rooms=["a","b"]
        seeds=[42,43,44]
        data={}
        for room in rooms:
            for i,s in enumerate(seeds):
                for arm, normal in (("Refresh1",4.),("Full4",3.),("Full5",4.),("Full6",5.),("Full10",8.)):
                    data[arm,room,s]=normal*(9 if i==2 else 1)
        picked, values=select_count(data,rooms,seeds,"Refresh1",[4,5,6,10])
        self.assertEqual(picked,5)
        self.assertEqual(values[5]["ordinary_over_candidate"],1.)
        del data["Full5","a",42]
        with self.assertRaises(KeyError):
            select_count(data,rooms,seeds,"Refresh1",[4,5])

    def test_no_affordable_budget(self):
        data={(a,"r",s):float(k) for a,k in (("Narrow",1),("Full1",2),("Full2",3)) for s in (7,77,777)}
        self.assertIsNone(select_count(data,["r"],[7,77,777],"Narrow",[1,2])[0])

    def test_single_zero_forward_and_deterministic_block_order(self):
        rooms=[dict(room=f"visit/{i}",slug=f"v_{i}") for i in range(6)]
        arms=["Full10","Full40","Cache10","Refresh1","Full5","ZERO"]
        first=jobs_for("fresh",rooms,[7,77,777],arms,83)
        self.assertEqual(first,jobs_for("fresh",rooms[::-1],[7,77,777],arms,83))
        self.assertEqual(len(first),96)
        self.assertEqual(len({j["id"] for j in first}),96)
        self.assertEqual([j["seed"] for j in first if j["arm"]=="ZERO"], [7]*6)
        self.assertEqual(len([j for j in first if j["steps"]==5]),18)

    def test_common_cap_counts_failures_and_check_packet_digests(self):
        base=dict(room="r",arm="Refresh1",input_sha256="i",reference_sha256="r",peak_allocated_bytes=10,peak_reserved_bytes=18)
        rows=[dict(base,status="completed"),dict(base,status="oom")]
        self.assertEqual(capacity(rows)["Refresh1"]["failures"],{"oom":1})
        self.assertEqual(capacity(rows)["Refresh1"]["completed"],1)
        self.assertEqual(len(make_packets(rows)),1)
        rows[-1]["input_sha256"]="changed"
        with self.assertRaisesRegex(ValueError,"Unmatched"):
            make_packets(rows)


if __name__=="__main__":
    unittest.main()
