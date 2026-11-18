"""Compare the exact allowed fitting tensors of M2 and centered/clean M3.

This is one prerequisite for prospective M2 reuse, not a reuse authorization.
It loads fitting observations/poses only, never predictions or references.
Run in the pinned image without GPU access. The original M2 function bodies
are extracted without importing models; no bound implementation is changed.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

from check_profiled_m3_network import original_cpu_loader, function_node
from profiled_m3_network import load_case, native_prior_samples


ROOT=Path(__file__).resolve().parents[1]
IMAGE="sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50"
PLAN="manifests/lora_profiled_m3_inputs_v2.json"


def sha(path,text=False):
    value=Path(path).read_bytes()
    return hashlib.sha256(value.replace(b"\r\n",b"\n") if text else value).hexdigest()


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def fingerprint(value):
    value=array(value)
    header=json.dumps([value.dtype.str,list(value.shape)],separators=(",",":")).encode()
    return dict(shape=list(value.shape),dtype=value.dtype.str,
                sha256=hashlib.sha256(header+b"\0"+value.tobytes(order="C")).hexdigest())


def tensor_inventory(case,original_rows):
    if len(case["samples"])!=5 or len(case["folds"])!=5 or len(original_rows)!=5:
        raise ValueError("Exactly five fitting views are required")
    tensors={key:array(case[key]) for key in ("K","w2c","c2w")}
    for view in range(5):
        for key in ("xy","depth"):
            tensors[f"view{view}/{key}"]=array(case["samples"][view][key])
        tensors[f"view{view}/fold"]=array(case["folds"][view])
        rows=array(original_rows[view])
        if rows.ndim!=2 or rows.shape[1]!=7:
            raise ValueError("Native KDP needs the selected original seven-column rows")
        tensors[f"view{view}/original_rows"]=rows
    if any(not np.isfinite(value).all() for value in tensors.values()):
        raise ValueError("Nonfinite fitting tensor")
    return tensors


def compare(first,second):
    if set(first)!=set(second):
        raise ValueError("Fitting tensor inventories differ")
    report={}
    for name in sorted(first):
        a,b=first[name],second[name]
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):
            raise ValueError("M2/M3 fitting tensor mismatch: "+name)
        report[name]=fingerprint(a)
    return report


def original_prior_loader():
    namespace=dict(ROOT=ROOT,PUBLIC=ROOT/"results/lora_profiled_v1",np=np,
                   FRAMES=(25,225,425,625,825))
    source=ROOT/"scripts/check_profiled_endpoints.py"
    node=function_node(source,"native_prior_samples")
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),"exec"),namespace)
    return namespace["native_prior_samples"]


def check(image):
    if image!=IMAGE or platform.python_version()!="3.11.13" or torch.__version__!="2.7.1+cu128" or np.__version__!="2.1.2":
        raise ValueError("Use the pinned model image for this input-equivalence audit")
    if torch.cuda.is_available():
        raise ValueError("This prerequisite must run in a CPU-only container")
    old_case,old_rows=original_cpu_loader(),original_prior_loader()
    rows=[]
    for scene in ("heads","chess","fire"):
        for side in ("earlier","later"):
            first=old_case(scene,side,device="cpu")
            second=load_case(ROOT/PLAN,"inputs02",scene,side,"centered","clean",device="cpu")
            old_original,new_original=old_rows(first),native_prior_samples(second)
            tensors=compare(tensor_inventory(first,old_original),tensor_inventory(second,new_original))
            rows.append(dict(scene=scene,side=side,grid="centered",pose="clean",exact=True,
                tensor_sha256=tensors,compared_tensors=len(tensors),m3_case_input_digest=second["prepared_input_digest"],
                m3_input_sha256=second["input_sha256"],m2_packet_sha256=first["packet_sha256"],
                m2_splits_sha256=first["splits_sha256"],original_rows_compared_including_order=True))
    sources=("scripts/check_profiled_m3_reuse_inputs.py","scripts/check_profiled_m3_network.py",
             "scripts/profiled_m3_network.py","scripts/profiled_network.py","scripts/check_profiled_endpoints.py",
             "scripts/prepare_profiled_m3_inputs.py","scripts/prepare_profiled_inputs.py",
             "scripts/run_mast3r_neural_transfer.py","scripts/calibration_audit.py")
    m2=json.loads((ROOT/"manifests/lora_profiled_execution_v2.json").read_text(encoding="utf-8"))
    source_hashes={name:sha(ROOT/name,text=True) for name in sources}
    for name in set(sources)&set(m2["source_sha256_lf"]):
        if source_hashes[name]!=m2["source_sha256_lf"][name]:
            raise ValueError("Original M2 source identity changed: "+name)
    return dict(schema="profiled_m3_reuse_fitting_equivalence_v1",pass_all=True,
        source_sha256_lf=source_hashes,input_plan=PLAN,input_plan_sha256=sha(ROOT/PLAN),
        m2_registry_sha256=sha(ROOT/"manifests/lora_profiled_execution_v2.json"),
        runtime=dict(image=image,python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                     cuda_build=torch.version.cuda,device="cpu"),cases=rows,cases_exact=len(rows),
        hash_encoding="SHA256(compact_json([dtype.str,shape])+NUL+C_order_bytes)",
        gpu_used=False,prediction_arrays_opened=False,evaluation_reference_arrays_opened=False,
        optimizer_or_model_constructed=False,reuse_authorized=False,
        remaining_reuse_checks=["M2 selected sealed attempt membership and inventories",
            "successful isolated process receipt", "model/checkpoint/initialization/runtime/optimizer equality",
            "native/final endpoint acceptance", "source-bound registration and nominee identity"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous audit; choose a fresh output")
    report=check(args.image)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(pass_all=True,cases_exact=report["cases_exact"],gpu_used=False,reuse_authorized=False)))


if __name__=="__main__":
    main()
