"""Compare fixed-backward worker, upstream LoRA bridge and enabled zero-LR control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RELATIVE = 1.e-5
COSINE = .999999
ROBUST_FRACTION = 1.e-4


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def compare(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValueError("Array layouts differ")
    a, b = a.astype(np.float64, copy=False).ravel(), b.astype(np.float64, copy=False).ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite qualification component")
    exact = np.array_equal(a,b)
    if exact:
        return dict(relative_l2=0., cosine=1., sign_mismatches=0, exact=True, passed=True)
    norm_a = np.sqrt(np.sum(a*a))
    norm_b = np.sqrt(np.sum(b*b))
    if not norm_a or not norm_b:
        return dict(relative_l2=None, cosine=None, sign_mismatches=None, exact=False, passed=False)
    relative = np.sqrt(np.sum((a-b)**2)) / norm_a
    cosine = np.sum(a*b)/(norm_a*norm_b)
    robust = np.abs(a) > ROBUST_FRACTION*np.max(np.abs(a))
    wrong = np.count_nonzero(np.signbit(a[robust]) != np.signbit(b[robust])) + np.count_nonzero(b[robust]==0)
    ok = relative <= RELATIVE and cosine >= COSINE and wrong == 0
    return dict(relative_l2=float(relative), cosine=float(cosine), sign_mismatches=int(wrong), exact=False, passed=bool(ok))


def load(path):
    row = json.loads((path/"complete.json").read_text(encoding="utf-8"))
    for name,field in (("prediction.npz","predictions_sha256"), ("fixed_signed_gradients.npz","fixed_signed_gradients_sha256")):
        if digest(path/name) != row[field]:
            raise ValueError("Digest mismatch for " + str(path/name))
    return row


def bridge(reference, candidate):
    source, other = load(reference), load(candidate)
    keys = ("seed","frames","input_sha256","view_permutations","render_calls","completed_adam_updates","native_depth_batches")
    if (any(source[k] != other[k] for k in keys) or source["profile"] != "fixed_bridge"
            or other["profile"] != "fixed" or source["steps"] != 2 or other["ln_lr"] != 0
            or source["optimizer"]["lora_initial_sha256"] != other["optimizer"]["lora_initial_sha256"]
            or source["optimizer"]["ln_initial_sha256"] != other["optimizer"]["ln_final_sha256"]
            or source["optimizer"]["ln_initial_sha256"] != other["optimizer"]["ln_initial_sha256"]):
        raise ValueError("Bridge inputs, update schedule, native grouping or initial weights differ")
    results = {}
    with np.load(reference/"fixed_signed_gradients.npz", allow_pickle=False) as ref, np.load(candidate/"fixed_signed_gradients.npz", allow_pickle=False) as alt:
        if set(ref.files)!=set(alt.files) or len(ref.files) != 3*384:
            raise ValueError("Missing signed gradient or final parameter")
        negative_detected = False
        for name in ref.files:
            baseline=ref[name]
            results[name]=compare(baseline,alt[name])
            if not negative_detected and name.startswith(("1/","2/")) and np.any(baseline):
                negative_detected = not compare(baseline, -baseline)["passed"]
        if not negative_detected:
            raise ValueError("Wrong-sign control escaped")
    with np.load(reference/"prediction.npz", allow_pickle=False) as ref, np.load(candidate/"prediction.npz", allow_pickle=False) as alt:
        for key in ("canonical","depth","depth_conf","extrinsics","intrinsics"):
            r, c = ref[key], alt[key]
            if key != "canonical":
                r, c = r[0], c[0]
            if len(r)!=5 or len(c)!=5:
                raise ValueError("Wrong number of views")
            for i in range(5):
                results[f"readout/{key}/{i}"] = compare(r[i], c[i])
    misses = {name:metrics for name,metrics in results.items() if not metrics["passed"]}
    return dict(passed=not misses, reference=source["id"], candidate=other["id"],
                fraction_bitwise_equal=sum(v["exact"] for v in results.values())/len(results),
                count=len(results), failures=misses, sign_negative_control=True,
                weakest_cosine=min((v["cosine"] for v in results.values() if v["cosine"] is not None), default=None),
                largest_relative_l2=max((v["relative_l2"] for v in results.values() if v["relative_l2"] is not None), default=None))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("reference","standard","zero-lr","out"):
        parser.add_argument("--"+key, type=Path, required=True)
    args=parser.parse_args()
    if (load(args.standard)["ln_policy"]!="none" or load(args.zero_lr)["ln_policy"]!="both"):
        raise ValueError("Expected standard and all-LayerNorm zero-LR controls")
    rows=[bridge(args.reference,args.standard),bridge(args.reference,args.zero_lr)]
    result=dict(passed=all(row["passed"] for row in rows), relative_l2_limit=RELATIVE,
                minimum_cosine=COSINE, robust_relative_to_largest=ROBUST_FRACTION, comparisons=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(result["passed"], [r["count"] for r in rows])
    if not result["passed"]:
        raise SystemExit(1)
