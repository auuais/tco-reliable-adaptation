"""Repeat the eight-update engineering check with phase-wise initial GPU residency."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import probe_cached_operands_gpu as previous
from evaluation_stability import dump, sha
from score_saved_geometry import read


ROOT = previous.ROOT
NAME = "staged_initial_gpu_probe"
PLAN = ROOT / "manifests" / (NAME + ".json")
JOB = dict(previous.JOB, run_id="staged_probe_heads")
FOLDER = ROOT / "runs" / (JOB["run_id"] + "_a1")
REPORT = ROOT / "reports" / NAME


def protocol():
    paths = [Path(__file__), ROOT / "scripts/stage_initial_forward.py", previous.PLAN]
    return dict(study=NAME, job=JOB, steps=8, limit_seconds=480, cloud_usd=0,
                hashes={p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in paths},
                activation="The preceding cached-operand probe must have a recorded interruption before its storage installation.",
                method="During initial inference execute the upstream model with image encoder, transformer blocks and prediction heads on CUDA in sequence. Retain all four required feature tensors. Restore transformer blocks to CUDA before upstream adapter initialization. Use the same cached operands and eight full-size optimization steps as the preceding probe.",
                criteria="Match inputs, priors, initial prediction/adapter/RNG hashes, and initial scalar loss against the original completed Heads run. Require eight finite optimizer gradient records. Report all residency phase boundaries, timing, memory and any interruption, including the last completed phase.")


def validate():
    previous.validate_inputs()
    if read(PLAN) != protocol():
        raise ValueError("Staged probe changed")
    parent = previous.ROOT / "reports" / previous.STUDY / "status.json"
    if not parent.is_file() or read(parent)["status"] != "interrupted" or (previous.FOLDER / "storage.json").exists():
        raise ValueError("No preceding interruption before storage installation")


def worker():
    import run_repeat_calibration as base
    from stage_initial_forward import stage_initial_forward

    phases = {}
    def observe(name, memory):
        phases[name] = memory
        dump(FOLDER / "initial_phases.json", phases)

    factory = base.tco_module.TCO_VGGT_LoRA.from_pretrained
    def construct(*args, **kwargs):
        model = factory(*args, **kwargs)
        stage_initial_forward(model, observe)
        return model
    base.tco_module.TCO_VGGT_LoRA.from_pretrained = staticmethod(construct)
    # This fresh worker shares the previous eight-step execution, with distinct evidence paths.
    previous.FOLDER, previous.JOB, previous.STUDY, previous.PLAN = FOLDER, JOB, NAME, PLAN
    previous.worker()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "worker"))
    action = parser.parse_args().action
    if action == "freeze":
        if PLAN.exists():
            raise FileExistsError(PLAN)
        dump(PLAN, protocol())
        return
    validate()
    if action == "worker":
        worker()
        return
    if FOLDER.exists():
        raise FileExistsError(FOLDER)
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "probe.log").open("x") as log:
        try:
            subprocess.run([sys.executable, "-u", __file__, "worker"], check=True,
                           timeout=protocol()["limit_seconds"], stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                                    OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", PYTHONHASHSEED="42"))
        except (OSError, subprocess.SubprocessError) as exc:
            dump(REPORT / "status.json", dict(status="interrupted", reason=str(exc)))
            raise
    previous.FOLDER = FOLDER
    results = previous.compare()
    results["initial_phases"] = read(FOLDER / "initial_phases.json")
    dump(REPORT / "status.json", dict(status="complete", **results))
    if not results["concordant"]:
        raise ValueError("Starting-state or finite-gradient check failed")


if __name__ == "__main__":
    main()
