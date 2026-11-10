"""Matched low-memory replication of the same camera interventions, in its own evidence tree."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import run_sensor_tco as parent
from evaluation_stability import dump,sha
from score_saved_geometry import read

ROOT=parent.ROOT
NAME="sensor_tco_readout_replication"
MANIFEST=ROOT/"manifests"/(NAME+".json")
RAW=ROOT/"reports"/NAME
PARENT=parent.MANIFEST


def plan():
    p=read(PARENT)
    jobs=[dict(j,run_id=j["run_id"].replace("sensor17_","sensor17r_",1)) for j in p["jobs"]]
    inputs=["scripts/selected_readouts.py","scripts/run_sensor_readout_replication.py",
            "patches/tco-selected-readouts.patch","results/selected_readouts/real_cpu_control.json",
            "scripts/run_sensor_tco.py","scripts/evaluate_sensor_tco.py"]
    return dict(**{k:v for k,v in p.items() if k not in ("study","sources","jobs","runtime")},
                study=NAME,jobs=jobs,parent_manifest_sha256=sha(PARENT),sources={f:sha(ROOT/f) for f in inputs},
                activation="Use after a technical interruption in the original local factorial run. Run original Heads/released and Chess/frame-only configurations first as direct engineering bridges. Preserve the full original evidence. Report all four conditions from this uniform replication separately from the original implementation.",
                runtime="Every run is fresh. All twelve use selected head readouts and otherwise the same upstream commands, full 40 updates and fixed frames. One attempt per new run, 30 minute job ceiling, archive incomplete attempts. Compare initial heads, adapter initialization, timing, peak memory and initial loss to available original executions. Do not infer GPU bitwise equivalence from the CPU control. An incompatible starting state halts the replication for review.")


def worker(job):
    import run_repeat_calibration as base
    from selected_readouts import enable

    original_factory=base.tco_module.TCO_VGGT_LoRA.from_pretrained
    def factory(*args,**kwargs):
        model=original_factory(*args,**kwargs)
        original_forward=model.forward_original_vggt
        def first_forward(*positional,**keyword):
            # run_memory_probe has removed the unused point head before this first VGGT forward.
            layers=enable(model)
            dump(ROOT/"runs"/(job["run_id"]+"_a1")/"readouts.json",
                 dict(retained_layers=layers,total_layers=model.aggregator.depth,patch_sha256=sha(ROOT/"patches/tco-selected-readouts.patch")))
            return original_forward(*positional,**keyword)
        model.forward_original_vggt=first_forward
        return model
    base.tco_module.TCO_VGGT_LoRA.from_pretrained=staticmethod(factory)
    parent.NAME,parent.MANIFEST=NAME,MANIFEST
    parent.run_one(job,1)


def check_bridge(job):
    original=ROOT/"runs"/(job["run_id"].replace("sensor17r_","sensor17_",1)+"_a1")
    if not (original/"capture.json").exists():return None
    folder=ROOT/"runs"/(job["run_id"]+"_a1")
    a,b=[read(f/"capture.json")["records"] for f in (original,folder)]
    tested={}
    for key,field in (("initial_predictions","tensor_sha256"),("initial_adapters","tensor_sha256"),
                      ("after_adapter_initialization_rng","sha256")):
        if key in a:
            tested[key]=a[key][field]==b[key][field]
    active=read(folder/"run.json")
    baseline=read(original/"run.json")
    tested["same_input_and_reference"]=baseline["evaluation_tensor_sha256"]==active["evaluation_tensor_sha256"]
    tested["same_supplied_prior"]=all(baseline["prior_metadata"][k]==active["prior_metadata"][k]
                                     for k in ("extrinsics_sha256","intrinsics_sha256"))
    old_loss=read(original/"losses.json") if (original/"losses.json").exists() else []
    new_loss=read(folder/"losses.json")
    bridge=dict(tested=tested,initial_total_loss_original=old_loss[0]["total"] if old_loss else None,
                initial_total_loss_selected=new_loss[0]["total"],original_completed=baseline["status"]=="complete")
    for key in ("peak_allocated_bytes","peak_reserved_bytes","inference_and_adaptation_seconds"):
        bridge[key]=dict(original=baseline.get(key),selected=active[key])
    return bridge


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("freeze","run","worker"))
    p.add_argument("--job",type=int)
    args=p.parse_args()
    spec=plan()
    if args.action=="freeze":
        if MANIFEST.exists():raise FileExistsError(MANIFEST)
        dump(MANIFEST,spec)
        return
    if read(MANIFEST)!=spec or any(sha(ROOT/f)!=d for f,d in read(PARENT)["sources"].items()):
        raise ValueError("Original or replication protocol changed")
    if args.action=="worker":
        worker(spec["jobs"][args.job])
        return
    RAW.mkdir(parents=True,exist_ok=True)
    if not any(read(f).get("status")=="interrupted" for f in (ROOT/"reports/sensor_tco_v17").glob("sensor17_*_a*.json")):
        raise ValueError("This replication is conditional on a recorded technical interruption")
    if sha(ROOT/"environment/development_dataset_cache_v7.npy")!=spec["dataset_cache_sha256"]:
        raise ValueError("Unexpected dataset cache")
    for i,job in enumerate(spec["jobs"]):
        if args.job is not None and args.job!=i:continue
        name=job["run_id"]+"_a1"
        path=RAW/(name+".json")
        if path.exists():
            old=read(path)
            if old["status"]!="complete" or (old.get("bridge") and not all(old["bridge"]["tested"].values())):
                raise ValueError("Previous attempt needs diagnosis: "+name)
            continue
        print("Starting",name,flush=True)
        try:
            with (RAW/(name+".log")).open("x") as log:
                subprocess.run([sys.executable,"-u",__file__,"worker","--job",str(i)],check=True,timeout=1800,
                               stdout=log,stderr=subprocess.STDOUT,
                               env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",
                                        NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42"))
            bridge=check_bridge(job)
            record=ROOT/"runs"/name/"run.json"
            dump(path,dict(**job,status="complete",attempt=1,record_sha256=sha(record),bridge=bridge))
        except (OSError,subprocess.SubprocessError) as exc:
            dump(path,dict(**job,status="interrupted",attempt=1,reason=str(exc)))
            raise
        if bridge and not all(bridge["tested"].values()):raise ValueError("Incompatible initial state: "+name)


if __name__=="__main__":main()
