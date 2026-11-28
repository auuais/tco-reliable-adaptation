"""Release the registered ten-step pilot using resource counters, update histories and input identity."""
from tco_two_track_protocol import ROOT, read, digest, seal

RAW = ROOT / "reports/tco_two_track_v1"
PUBLIC = ROOT / "results/tco_two_track_v1"


def main():
    ledger = ROOT/"manifests/tco_two_track_v1_pilot.json"
    cases = read(ledger)["jobs"]
    expect = {"Full10":[384]*10, "Cache10":[192]*10,
              "Refresh1":[384]+[192]*9, "Refresh1Uncached":[384]+[192]*9}
    if {j["arm"] for j in cases}!=set(expect) or len(cases)!=4:
        raise ValueError("Wrong registered pilot")
    rows = []
    perms = []
    for j in cases:
        sources = [d for d in (RAW/"cases"/j["id"]).glob("attempt*") if (d/"terminal.json").is_file() and read(d/"terminal.json")["status"]=="completed"]
        if len(sources)!=1:
            raise ValueError("No unique successful pilot for "+j["id"])
        case, = sources
        report = case/"model/complete.json"
        row, terminal = read(report),read(case/"terminal.json")
        if (terminal["complete_sha256"]!=digest(report) or row["predictions_sha256"]!=digest(case/"model/prediction.npz")
                or any(row[k]!=v for k,v in j.items()) or row["profile"]!="clean" or row["room"]!="heads/seq-01"
                or row["seed"]!=42 or row["completed_adam_updates"]!=10 or row["active_counts"]!=expect[j["arm"]]
                or row["rental_compute_usd"] or row["oom_count"] or not 0<row["peak_allocated"]<=row["cap_bytes"]):
            raise ValueError("Unqualified pilot "+j["id"])
        perms.append(row["view_permutations"])
        if j["arm"] in ("Refresh1","Refresh1Uncached"):
            cache = row["cache"]
            if (row["adam_state_counts"]!={"prefix_1":192,"suffix_10":192} or
                  cache["generation"]!=1 or (j["arm"]=="Refresh1")!=(cache["hits"]>0)):
                raise ValueError("Hybrid cache/Adam pilot failed")
        rows.append({**{k:row[k] for k in ("id","arm","steps","scene_seconds","peak_allocated","peak_reserved","cap_bytes",
                     "adam_state_counts","active_counts","cache","allocation_retries","oom_count","predictions_sha256")},
                     "complete_sha256":digest(report)})
    if any(p!=perms[0] for p in perms):
        raise ValueError("View permutations differ")
    record=dict(passed=True, ledger_sha256=digest(ledger),cases=rows,
                common_view_permutations=perms[0],rental_compute_usd=0)
    seal(RAW/"pilot_pass.json",record)
    seal(PUBLIC/"a_resource_pilot.json",record)
    print("Four clean ten-step pilot cases passed",flush=True)


if __name__=="__main__":
    main()
