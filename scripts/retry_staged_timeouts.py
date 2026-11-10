"""After every first attempt, retry only documented runtime timeouts once, in study order."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import run_staged_sensor_factorial as study
from evaluation_stability import dump, sha
from score_saved_geometry import read


PLAN = study.ROOT / "manifests/staged_sensor_timeout_recovery.json"


def protocol():
    return dict(study_sha256=sha(study.PLAN), handler_sha256=sha(Path(__file__)),
                policy="Wait for terminal first-attempt records for all twelve predefined cells. Give each timeout exactly one fresh attempt in original study order with the same frozen model, seed, GPU settings, scientific configuration and 30-minute limit. No accuracy, loss, or number of completed steps is used for selection. Preserve every original byte and record both attempts. A second timeout leaves the full scientific comparison incomplete.",
                controls="Check the new attempt's initialized heads, adapters, RNG, input/reference and priors against the original attempt wherever available; verify forty finite gradients and the matched storage mode. The scoring code already accepts at most one completed attempt among the two. Report completion conditioning if any retry supplies a score.", cloud_usd=0)


def snapshot(folder):
    return {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob("*") if p.is_file()}


def worker(job, spec):
    staged = study.staged
    readouts = staged.previous.readouts
    first = study.ROOT / "runs" / (job["run_id"] + "_a1")
    second = first.with_name(job["run_id"] + "_a2")
    save = readouts.dump
    run_original = readouts.parent.run_one

    def save_readouts(path, values):
        if Path(path) != first / "readouts.json":
            raise ValueError("Unexpected readout output location")
        save(second / "readouts.json", values)

    def second_attempt(selected, attempt):
        if selected != job or attempt != 1:
            raise ValueError("Unexpected worker selection")
        run_original(selected, 2)

    # Only redirect the two first-attempt-specific output names in the frozen probe wrappers.
    readouts.dump = save_readouts
    readouts.parent.run_one = second_attempt
    staged.FOLDER, staged.JOB, staged.NAME, staged.PLAN = second, job, study.STUDY, study.PLAN
    staged.previous.protocol = lambda: {"steps":spec["steps"]}
    staged.worker()


def verify(job, spec, first, second):
    run=read(second/"run.json")
    previous=read(first/"run.json") if (first/"run.json").is_file() else {}
    a=read(first/"capture.json")["records"] if (first/"capture.json").is_file() else {}
    b=read(second/"capture.json")["records"]
    bridge = {name:a[name][field]==b[name][field] for name,field in
              (("initial_predictions","tensor_sha256"),("initial_adapters","tensor_sha256"),
               ("after_adapter_initialization_rng","sha256")) if name in a}
    for key in ("evaluation_tensor_sha256","prior_metadata"):
        if key in previous:
            bridge[key]=previous[key]==run[key]
    trace=read(second/"gradients.json")
    if (not all(bridge.values()) or run["attempt"]!=2 or run["status"]!="complete"
            or run["arguments"]["scene"]!=job["scene"] or run["sensor_condition"]!=job["condition"]
            or run["manifest_sha256"]!=sha(study.PLAN) or run["arguments"]["steps"]!=spec["steps"]
            or run["base_parameters_modified"] or len(trace)!=spec["steps"]
            or len(read(second/"losses.json"))!=spec["steps"]
            or not all(r["finite"] and r["tensors_with_gradient"]==384 for r in trace)
            or read(second/"storage.json")["projections"]!=192
            or "ready_for_adapters" not in read(second/"initial_phases.json")):
        raise ValueError("Retry trajectory or starting state failed validation")
    return dict(**job, attempt=2, status="complete", record_sha256=sha(second/"run.json"),
                first_attempt_bridge=bridge, recovery_protocol_sha256=sha(PLAN))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("freeze","run","worker"))
    parser.add_argument("--job",type=int)
    args=parser.parse_args()
    if args.action=="freeze":
        if PLAN.exists(): raise FileExistsError(PLAN)
        dump(PLAN,protocol())
        return
    spec=read(study.PLAN)
    study.check_inputs(spec)
    if read(PLAN)!=protocol(): raise ValueError("Timeout recovery protocol changed")
    jobs=spec["jobs"]
    statuses=[read(study.RAW/(j["run_id"]+"_a1.json")) for j in jobs]
    if args.action=="worker":
        if statuses[args.job]["status"]!="interrupted" or "timed out" not in statuses[args.job].get("reason",""):
            raise ValueError("Only a documented timeout can be retried")
        worker(jobs[args.job],spec)
        return
    for i,(job,old) in enumerate(zip(jobs,statuses)):
        if old["status"]=="complete":
            continue
        if old["status"]!="interrupted" or "timed out" not in old.get("reason",""):
            raise ValueError("Non-timeout interruption requires separate diagnosis")
        first=study.ROOT/"runs"/(job["run_id"]+"_a1")
        second=first.with_name(job["run_id"]+"_a2")
        path=study.RAW/(second.name+".json")
        if path.exists(): continue
        if second.exists(): raise FileExistsError(second)
        original_files=snapshot(first)
        with path.with_suffix(".log").open("x") as log:
            try:
                subprocess.run([sys.executable,"-u",__file__,"worker","--job",str(i)],check=True,timeout=spec["limit_seconds"],
                               stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",
                               OPENBLAS_NUM_THREADS="4",NUMEXPR_NUM_THREADS="4",PYTHONHASHSEED="42"))
                outcome=verify(job,spec,first,second)
            except subprocess.TimeoutExpired as exc:
                outcome=dict(**job,attempt=2,status="interrupted",reason=str(exc))
            except (OSError,subprocess.CalledProcessError,ValueError,KeyError) as exc:
                dump(path,dict(**job,attempt=2,status="interrupted",reason=str(exc)))
                raise
            if original_files!=snapshot(first): raise ValueError("First attempt was modified")
            dump(path,outcome)


if __name__=="__main__":
    main()
