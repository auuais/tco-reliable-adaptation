"""Verify prepared fitting inputs on CPU; never run a model or score geometry.

The original M2 loader body is executed with its existing CPU dependencies only,
avoiding a MASt3R import/model initialization. It reads permitted fitting packets
and poses, not prediction archives or evaluation depths.
"""
import argparse
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from calibration_audit import resized_k
from profiled_m3_network import load_case, native_prior_samples

ROOT = Path(__file__).resolve().parents[1]


def sha(path, text=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def function_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def original_cpu_loader():
    """Load exactly two trusted local function bodies, without upstream imports."""
    namespace = dict(ROOT=ROOT, np=np, torch=torch, json=json, sha=sha,
                     resized_k=resized_k, FRAMES=(25, 225, 425, 625, 825))
    sources = (("scripts/run_mast3r_neural_transfer.py", "intrinsic"),
               ("scripts/profiled_network.py", "load_case"))
    for file, name in sources:
        node = function_node(ROOT / file, name)
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(module, file, "exec"), namespace)
    return namespace["load_case"]


def check(plan, attempt):
    old = original_cpu_loader()
    existing = function_node(ROOT / "scripts/check_profiled_endpoints.py", "capture_endpoint")
    prospective = function_node(ROOT / "scripts/profiled_m3_network.py", "capture_endpoint")
    if ast.dump(existing, include_attributes=False) != ast.dump(prospective, include_attributes=False):
        raise ValueError("M3 endpoint computation differs from the immutable M2 body")
    rows = []
    for scene, side, grid, pose in itertools.product(("heads", "chess", "fire"), ("earlier", "later"),
                                                    ("centered", "shifted"), ("clean", "noisy")):
        case = load_case(plan, attempt, scene, side, grid, pose, device="cpu")
        original_rows = native_prior_samples(case)
        paired = grid == "centered" and pose == "clean"
        if paired:
            reference = old(scene, side, device="cpu")
            for key in ("K", "w2c", "c2w"):
                if not torch.equal(case[key], reference[key]):
                    raise ValueError(f"Original centered/clean {key} changed")
            for i in range(5):
                for key in ("xy", "depth"):
                    if not torch.equal(case["samples"][i][key], reference["samples"][i][key]):
                        raise ValueError("Original centered/clean observations changed")
                if not torch.equal(case["folds"][i], reference["folds"][i]):
                    raise ValueError("Original centered/clean spatial folds changed")
            for key in ("packet_sha256", "splits_sha256"):
                if case[key] != reference[key]:
                    raise ValueError("Original centered/clean input artifact identity changed")
        rows.append(dict(scene=scene, side=side, grid=grid, pose_condition=pose,
                         counts=[len(v) for v in original_rows], input_digest=case["prepared_input_digest"],
                         original_m2_loader_compared=paired, input_sha256=case["input_sha256"]))
    files = ("scripts/check_profiled_m3_network.py", "scripts/profiled_m3_network.py",
             "scripts/prepare_profiled_m3_inputs.py", "scripts/check_profiled_endpoints.py",
             "scripts/profiled_network.py", "scripts/run_mast3r_neural_transfer.py", "scripts/calibration_audit.py")
    return dict(schema="profiled_m3_cpu_case_bridge_v1", pass_all=True, cases=rows,
                cases_checked=len(rows), centered_clean_m2_cases_exact=sum(r["original_m2_loader_compared"] for r in rows),
                endpoint_body_ast_exact=True, source_sha256_lf={p: sha(ROOT / p, text=True) for p in files},
                device="cpu", gpu_used=False, prediction_arrays_opened=False, accuracy_scoring_performed=False,
                endpoint_gpu_execution_verified=False,
                limitation="Input/body equivalence does not replace a prospectively registered real M3 endpoint check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "manifests/lora_profiled_m3_inputs_v2.json")
    parser.add_argument("--attempt", default="inputs02")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.plan.resolve(), args.attempt)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: report[key] for key in ("pass_all", "cases_checked", "centered_clean_m2_cases_exact",
                                                  "endpoint_body_ast_exact", "gpu_used", "accuracy_scoring_performed")}))
