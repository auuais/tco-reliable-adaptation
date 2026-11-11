"""Portable replay of the frozen camera-gradient evidence across Git checkout conventions."""
import hashlib
import json
import math
import subprocess

import numpy as np

import audit_masked_camera_gradients as audit


def compare(stored, replay):
    if type(stored) is not type(replay):
        raise ValueError("Result types differ")
    if isinstance(stored, dict):
        if stored.keys() != replay.keys():
            raise ValueError("Result fields differ")
        for k in stored:
            compare(stored[k], replay[k])
    elif isinstance(stored, float):
        if not math.isclose(stored,replay,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f"Numerical mismatch: {stored} vs {replay}")
    elif stored != replay:
        raise ValueError("Published value differs from replay")


def main():
    frozen = audit.read(audit.PLAN)
    rebuilt = audit.protocol()
    upstream = audit.ROOT / "third_party/TCO"
    revision = subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()
    if revision != "65387333877723cf99086f7f29089379dd50c59a":
        raise ValueError("Unexpected TCO revision")
    for name in ("constraints.py","tco_loss.py","utils/geometry.py"):
        key = "third_party/TCO/" + name
        git_bytes = subprocess.check_output(["git","-C",str(upstream),"show","HEAD:" + name])
        if (upstream / name).read_bytes().replace(b"\r\n",b"\n") != git_bytes.replace(b"\r\n",b"\n"):
            raise ValueError("Upstream source differs beyond checkout line endings: " + name)
        rebuilt["sources"][key] = hashlib.sha256(git_bytes.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")).hexdigest()
    if rebuilt != frozen:
        raise ValueError("Frozen protocol differs")
    result = audit.read(audit.PUBLIC / "summary.json")
    trace = audit.PUBLIC / "gradient_traces.zip"
    if result["manifest_sha256"] != audit.sha(audit.PLAN) or result["trace_sha256"] != audit.sha(trace):
        raise ValueError("Published protocol or gradient bytes changed")
    with np.load(trace,allow_pickle=False) as data:
        if data["variants"].shape != (183,6,5,9) or data["clean"].shape != (3,5,9):
            raise ValueError("Incomplete derivative arrays")
        references = dict(zip(data["scenes"].tolist(),data["clean"]))
        recomputed = []
        for row, variants in zip(result["rows"],data["variants"]):
            values = audit.errors(row,variants,references[row["scene"]])
            compare(row["errors"], values)
            recomputed.append(dict(row,errors=values))
    compare(result["results"],audit.aggregate(recomputed))
    print("Replayed 183 cases and 1,098 derivative arrays, including all advancement decisions.")


if __name__ == "__main__":
    main()
