"""Run one local eight-update engineering check with the original TCO losses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import run_sensor_readout_replication as readouts
from evaluation_stability import dump, sha
from score_saved_geometry import read


ROOT = readouts.ROOT
STUDY = "cached_operand_gpu_probe"
JOB = dict(scene="heads/seq-01", condition="released", run_id="operand_probe_heads")
FOLDER = ROOT / "runs" / (JOB["run_id"] + "_a1")
BASELINE = ROOT / "runs/sensor17_heads_seq-01_released_a1"
PLAN = ROOT / "manifests" / (STUDY + ".json")
CONTROL = ROOT / "results/selected_readouts/cached_operand_cpu.json"


def protocol():
    sources = [Path(__file__).resolve(), ROOT / "scripts/frozen_linear_storage.py",
               ROOT / "scripts/check_backbone_operands.py", readouts.MANIFEST]
    return dict(study=STUDY, job=JOB, steps=8, limit_seconds=480, cloud_usd=0,
                sources={p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
                parent=read(readouts.MANIFEST)["jobs"][0],
                activation="Use only after a technical interruption of the selected-readout replication and a passing two-update pretrained CPU equivalence control.",
                purpose="Measure feasibility of full-size adaptation on one local device. Check matching data, priors, initial model/adapter/RNG, initial loss, eight finite gradients, and memory. Later loss differences are descriptive. This short engineering comparison cannot estimate reconstruction accuracy or benchmark speed.")


def validate_inputs():
    if read(PLAN) != protocol():
        raise ValueError("Probe source or parent protocol changed")
    specs = (read(readouts.MANIFEST), read(readouts.PARENT))
    for spec in specs:
        for file, digest in spec["sources"].items():
            if sha(ROOT / file) != digest:
                raise ValueError(f"Source mismatch in {file}")
    cache = ROOT / "environment/development_dataset_cache_v7.npy"
    if sha(cache) != specs[0]["dataset_cache_sha256"]:
        raise ValueError("Dataset cache mismatch")
    actual = subprocess.check_output(["git", "-C", str(ROOT / "third_party/TCO"),
                                      "rev-parse", "HEAD"], text=True).strip()
    if actual != specs[0]["upstream_commit"]:
        raise ValueError("Unexpected upstream TCO revision")
    cpu = read(CONTROL)
    if not (cpu["bitwise_match"] and cpu["implementation_sha256"] == sha(ROOT / "scripts/frozen_linear_storage.py")
            and cpu["test_sha256"] == sha(ROOT / "scripts/check_backbone_operands.py")):
        raise ValueError("A matching successful CPU control is required")
    if not any(read(p).get("status") == "interrupted" for p in readouts.RAW.glob("*_a1.json")):
        raise ValueError("The selected-readout replication has no recorded interruption")
    if read(BASELINE / "run.json")["status"] != "complete":
        raise ValueError("Original completed reference is missing")


def worker():
    import torch
    import run_repeat_calibration as base
    from frozen_linear_storage import enable

    parent_plan = dict(read(readouts.PARENT), steps=protocol()["steps"])
    factory = base.tco_module.TCO_VGGT_LoRA.from_pretrained
    def construct(*args, **kwargs):
        model = factory(*args, **kwargs)
        original_enable = model.enable_lora_gradients
        def install():
            original_enable()
            dtype = torch.get_autocast_dtype("cuda")
            before = torch.cuda.memory_allocated()
            storage = enable(model, dtype)
            dump(FOLDER / "storage.json", dict(**storage, allocated_before_bytes=before,
                 allocated_after_bytes=torch.cuda.memory_allocated(), device=torch.cuda.get_device_name()))
        model.enable_lora_gradients = install
        return model
    base.tco_module.TCO_VGGT_LoRA.from_pretrained = staticmethod(construct)
    readouts.parent.plan = lambda: parent_plan
    readouts.NAME, readouts.MANIFEST = STUDY, PLAN
    readouts.worker(JOB)


def compare():
    a, b = (read(p / "run.json") for p in (BASELINE, FOLDER))
    x, y = (read(p / "capture.json")["records"] for p in (BASELINE, FOLDER))
    losses, gradients = read(FOLDER / "losses.json"), read(FOLDER / "gradients.json")
    target = read(BASELINE / "losses.json")
    checks = {k: x[k][field] == y[k][field] for k, field in
              (("initial_predictions", "tensor_sha256"), ("initial_adapters", "tensor_sha256"),
               ("after_adapter_initialization_rng", "sha256"))}
    checks.update(input_match=a["evaluation_tensor_sha256"] == b["evaluation_tensor_sha256"],
                  prior_match=a["prior_metadata"] == b["prior_metadata"],
                  initial_loss_match=losses[0]["total"] == target[0]["total"],
                  complete=b["status"] == "complete" and len(losses) == len(gradients) == protocol()["steps"],
                  finite_gradients=all(g["finite"] and g["tensors_with_gradient"] == 384 for g in gradients),
                  frozen_base=b["base_parameters_modified"] == [])
    return dict(checks=checks, concordant=all(checks.values()), storage=read(FOLDER / "storage.json"),
                final_record_sha256=sha(FOLDER / "run.json"), cpu_control_sha256=sha(CONTROL),
                peak_allocated_bytes=b["peak_allocated_bytes"], peak_reserved_bytes=b["peak_reserved_bytes"],
                inference_and_adaptation_seconds=b["inference_and_adaptation_seconds"],
                first_eight_total_losses=[dict(original=u["total"], cached=v["total"]) for u,v in zip(target, losses)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "worker", "run"))
    action = parser.parse_args().action
    if action == "freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN, protocol())
        return
    validate_inputs()
    if action == "worker":
        worker()
        return
    if FOLDER.exists():
        raise FileExistsError(FOLDER)
    report = ROOT / "reports" / STUDY
    report.mkdir(parents=True, exist_ok=True)
    log = report / "probe.log"
    with log.open("x") as stream:
        try:
            subprocess.run([sys.executable, "-u", __file__, "worker"], check=True,
                           timeout=protocol()["limit_seconds"], stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                    OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONHASHSEED="42"))
        except (OSError, subprocess.SubprocessError) as exc:
            dump(report / "status.json", dict(status="interrupted", error=str(exc)))
            raise
    result = compare()
    dump(report / "status.json", dict(status="complete", **result))
    print(json.dumps(result, indent=2), flush=True)
    if not result["concordant"]:
        raise ValueError("Engineering controls failed")


if __name__ == "__main__":
    main()
