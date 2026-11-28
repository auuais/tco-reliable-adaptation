"""Verify the registered deterministic twins, stale-cache control and real-loss duplicates on CPU."""
import argparse


from pathlib import Path
import numpy as np
from qualify_tco_final_ln_bridge import compare
from tco_final_ln_protocol import ROOT, read, seal, sha


def verify(directory):
    meta = read(directory/"model/complete.json")
    if (read(directory/"terminal.json")["complete_sha256"] != sha(directory/"model/complete.json")
            or sha(directory/"model/prediction.npz") != meta["predictions_sha256"]):
        raise ValueError("Case digest mismatch")
    if meta.get("fixed_signed_gradients_sha256") and sha(directory/"model/fixed_signed_gradients.npz") != meta["fixed_signed_gradients_sha256"]:
        raise ValueError("Gradient digest mismatch")
    if meta["completed_adam_updates"]!=2 or meta["active_counts"]!=[384,192] or meta["adam_state_counts"]!={"prefix_1":192,"suffix_2":192}:
        raise ValueError("Unexpected Adam counts")
    return meta


def signed_twins(refdir,altdir,ref,alt):
    for k in ("lora_initial_sha256","seed","frames","input_sha256","view_permutations"):
        if ref[k]!=alt[k]:
            raise ValueError("Cached/uncached twins disagree at "+k)
    metrics = {}
    nonzero = set()
    with np.load(refdir/"model/fixed_signed_gradients.npz") as baseline, np.load(altdir/"model/fixed_signed_gradients.npz") as twin:
        names = ref["lora_initial_sha256"]
        expected = {"final/"+n for n in names} | {"grad/1/"+n for n in names}
        expected.update("grad/2/"+n for n in names if int(n.split(".")[2])>=12)
        if set(baseline.files)!=expected or set(twin.files)!=expected or len(expected)!=960:
            raise ValueError("Missing signed tensor")
        sign_negative_control = False
        for name in sorted(expected):
            values = baseline[name]
            metrics[name] = compare(values,twin[name])
            if name.startswith("grad/") and not sign_negative_control and np.any(values):
                sign_negative_control = not compare(values,-values)["passed"]
        # Since all B factors start at zero, B@A gives the signed effective delta
        # (up to the common alpha/rank). Identical final factors imply exact products.
        for name in sorted(n for n in names if n.endswith("lora_B")):
            left = "final/"+name
            right = left[:-1]+"A"
            b,a = baseline[left],baseline[right]
            active_rows = np.flatnonzero(np.any(b != 0,axis=1))
            changed = bool(len(active_rows) and np.any(b[active_rows[0]] @ a))
            if len(active_rows) and not changed:
                changed = bool(np.any(b @ a))
            passed = changed and metrics[left]["exact"] and metrics[right]["exact"]
            if changed:
                nonzero.add(name)
            if not passed:
                metrics["effective/"+name] = compare(b @ a,twin[left] @ twin[right])
            else:
                metrics["effective/"+name] = dict(exact=True,passed=True,relative_l2=0.,cosine=1.,sign_mismatches=0)
        if not sign_negative_control or not nonzero:
            raise ValueError("Ineffective update or wrong-sign control escaped")
    with np.load(refdir/"model/prediction.npz") as base, np.load(altdir/"model/prediction.npz") as twin:
        for key in ("depth","depth_conf","intrinsics","extrinsics","canonical"):
            b,t = (base[key],twin[key]) if key=="canonical" else (base[key][0],twin[key][0])
            if len(b)!=5 or len(t)!=5:
                raise ValueError("Missing view")
            for i in range(5):
                metrics[f"readout/{key}/{i}"] = compare(b[i],t[i])
    misses = {k:v for k,v in metrics.items() if not v["passed"]}
    a,b = ref["cache"],alt["cache"]
    # A two-step twin can have no hit when the final pass uses another dtype.
    # Its rebuilt BF16 and FP16 entries still must have matching physical tensors.
    cache_ok = (a["generation"]==b["generation"]==1 and a["builds"]>0 and b["builds"]==b["hits"]==0
                and b["prefix_stages"]>=a["prefix_stages"])
    return dict(passed=not misses and cache_ok,checked=len(metrics),bitwise=sum(v["exact"] for v in metrics.values()),
                failures=misses,nonzero_effective_modules=len(nonzero),sign_negative_control=sign_negative_control,
                cached=a,uncached=b,cache_check=cache_ok)


def run(ledger, raw):
    rows = read(ledger)["jobs"]
    items = {}
    for row in rows:
        completed = [p for p in (raw/"cases"/row["id"]).glob("attempt*")
                     if (p/"terminal.json").is_file() and read(p/"terminal.json")["status"]=="completed"]
        if len(completed)!=1:
            raise ValueError("Need one completed attempt: "+row["id"])
        d = completed[0]
        items[row["id"]] = (d,verify(d))
    def select(profile):
        return [items[j["id"]] for j in rows if j["profile"]==profile]
    (d1,r1),(d2,r2) = select("fixed")
    (stale_dir, stale), = select("stale")
    (d3,p1),(d4,p2) = select("real")
    fixed = signed_twins(d1,d2,r1,r2)
    control = stale["stale"]
    real = {}
    with np.load(d3/"model/prediction.npz") as a, np.load(d4/"model/prediction.npz") as b:
        for name in ("canonical","depth","depth_conf","extrinsics","intrinsics"):
            left,right = a[name].ravel(),b[name].ravel()
            diff = left.astype(np.float64)-right
            real[name] = dict(relative_l2=float(np.sqrt(np.sum(diff**2)/np.sum(left.astype(np.float64)**2))),
                              max_abs=float(np.abs(diff).max()))
    passed = (fixed["passed"] and control["stale_unchanged"] and control["rebuilt_matches_fresh"]
              and control["changed_elements"]>0 and p1["view_permutations"]==p2["view_permutations"])
    return dict(passed=bool(passed),ledger_sha256=sha(ledger),fixed=fixed,stale=control,
                real_loss_differences=real,real_view_orders_equal=p1["view_permutations"]==p2["view_permutations"],
                real_losses=[p1["losses"],p2["losses"]],
                source_cases=[dict(id=k,complete_sha256=sha(p/"model/complete.json")) for k,(p,r) in items.items()],
                rental_compute_usd=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs",type=Path,default=ROOT/"manifests/tco_two_track_v1_engineering.json")
    ap.add_argument("--raw",type=Path,default=ROOT/"reports/tco_two_track_v1")
    args = ap.parse_args()
    result = run(args.jobs,args.raw)
    target = args.raw/"qualification_pass.json"
    seal(target,result)
    print(result["passed"],result["fixed"]["checked"],result["fixed"]["bitwise"],flush=True)
    if not result["passed"]:
        raise SystemExit(1)

