"""Resume a registered clean GPU ledger after a retained OOM/watchdog exit, continuing independent controls."""
from pathlib import Path
import argparse
from run_profiled_d1_local import host_lock
from run_tco_two_track_a_local import MASTER, POLICY, ROOT, WORKER, RAW, committed, completed, launch
from tco_final_ln_protocol import read


def execute(ledger, pause_file=None):
    ledger=ledger.resolve()
    if ledger.parent != ROOT/"manifests" or not ledger.name.startswith("tco_two_track_v1_"):
        raise ValueError("Expected committed Track A ledger")
    committed(ledger,MASTER,WORKER,POLICY,Path(__file__))
    if not read(RAW/"qualification_pass.json")["passed"] or not read(RAW/"pilot_pass.json")["passed"]:
        raise ValueError("Clean resource and engineering qualification must pass")
    jobs=read(ledger)["jobs"]
    if any(j["profile"]!="clean" for j in jobs):
        raise ValueError("Only a clean ledger is accepted")
    with host_lock():
        for job in jobs:
            if pause_file is not None and pause_file.is_file():
                print('Paused before next case:', job['id'], flush=True)
                return 75
            status=launch(job,ledger)
            if status in ("provisioning","interrupted","timeout"):
                if pause_file is not None and pause_file.is_file():
                    print('Paused before retry:', job['id'], flush=True)
                    return 75
                status=launch(job,ledger)
            if status == "completed":
                continue
            if status in ("oom","timeout"):
                print(job["id"],"exhausted registered attempts:",status,flush=True)
                continue
            raise RuntimeError("Inspect "+job["id"]+": "+status)
    done=sum(completed(j) is not None for j in jobs)
    print("Terminal:",len(jobs),"completed:",done,"retained failures:",len(jobs)-done,flush=True)
    return 0


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs",type=Path,required=True)
    ap.add_argument("--pause-file",type=Path,help="Release the GPU lock at the next case boundary if this file exists (exit 75).")
    args=ap.parse_args()
    raise SystemExit(execute(args.jobs,args.pause_file))
