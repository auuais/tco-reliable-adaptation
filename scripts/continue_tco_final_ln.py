"""Resume the registered experiment after engineering. Each stage checks its immutable receipts."""
import argparse
from pathlib import Path
import subprocess
import sys

from tco_final_ln_protocol import LOCK, MANIFEST, PUBLIC, RAW, ROOT, STUDY, read
from run_tco_final_ln_local import attempts


def run(script, *args):
    command = [sys.executable,"-B","-u",str(ROOT/"scripts"/script),*[str(x) for x in args]]
    print("Starting",Path(script).stem,*args,flush=True)
    subprocess.run(command,check=True,cwd=ROOT)


def model(stage):
    run("run_tco_final_ln_local.py",stage)


def score(stage):
    run("release_tco_final_ln.py","score","--stage",stage)


def commit_lock(path):
    rel = path.relative_to(ROOT).as_posix()
    subprocess.run(["git","add","--",rel],cwd=ROOT,check=True)
    current = subprocess.run(["git","diff","--cached","--quiet","--",rel],cwd=ROOT)
    if current.returncode == 1:
        subprocess.run(["git","commit","--only","-m","Lock the outcome-selected TCO "+path.stem.split("_")[-1]+" cohort","--",rel],
                       cwd=ROOT,check=True)
    elif current.returncode != 0:
        raise RuntimeError("Unexpected Git status for "+rel)


def wait_owner(pid):
    # On Windows a process handle remains the correct identity until it exits, even if its PID is later reused.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE,wintypes.DWORD)
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x00100000,False,pid)
    if handle:
        try:
            while True:
                wait_status = kernel.WaitForSingleObject(handle,30000)
                if wait_status == 0:
                    break
                if wait_status != 258:
                    raise OSError("Could not wait on existing local launcher")
        finally:
            kernel.CloseHandle(handle)


def wait_development(pid):
    wait_owner(pid)
    plan = read(MANIFEST)
    for j in plan["development"]+plan["duplicates"]:
        paths = attempts(j["id"])
        if not paths or not (paths[-1]/"terminal.json").is_file():
            raise RuntimeError("Development process exited with outstanding slots: "+j["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-development-pid",type=int)
    parser.add_argument("--after-followup-pid",type=int, help="Continue an existing campaign through gated replication and archive")
    args = parser.parse_args()
    if not LOCK.is_file() or not read(RAW/"qualification_pass.json")["passed"]:
        raise RuntimeError("Require the engineering gate and exact fresh-room lock")
    if args.after_followup_pid:
        wait_owner(args.after_followup_pid)
        finish()
        return
    if args.active_development_pid:
        wait_development(args.active_development_pid)
    else:
        model("development")
        model("duplicates")
    model("primary")
    model("anchor")
    for stage in ("development","primary"):
        score(stage)
    run("lock_tco_final_ln.py","followup","--decision",PUBLIC/"primary_decision.json")
    commit_lock(ROOT/"manifests"/(STUDY+"_followup.json"))
    model("followup")
    model("diagnostics")
    score("followup")
    finish()


def finish():
    from lock_tco_final_ln_replication import InsufficientEligibleVisits, REPL_LOCK, gate, prepare
    from tco_final_ln_protocol import seal
    decision = gate()
    if decision["status"] == "pending":
        raise RuntimeError("Scientific branch is still pending: "+str(decision))
    if decision["status"] == "eligible":
        try:
            prepare(decision)
        except InsufficientEligibleVisits as exc:
            print("Independent replication remains unverified.", str(exc), flush=True)
        else:
            commit_lock(REPL_LOCK)
            model("replication")
            score("replication")
    elif decision["status"] != "not_promoted":
        raise ValueError("Unknown replication status")
    seal(PUBLIC/"replication_gate.json",decision)
    run("archive_tco_final_ln.py","final")
    print("Research artifacts archived; run final independent verification and publish.",flush=True)


if __name__ == "__main__":
    main()
