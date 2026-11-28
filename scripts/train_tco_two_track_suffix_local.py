"""Run the conditional suffix training pilot offline on the local RTX 3080."""
import argparse
import shutil
import subprocess
import time
from pathlib import Path

from run_profiled_d0_local import DOCKER, docker_json, ensure_stopped
from run_profiled_d1_local import host_lock
from run_tco_courtyard_camera_local import check_device_available
from run_tco_final_ln_local import classify, device, docker_base, mount, utc
from run_tco_two_track_a_local import MASTER, PREP, RAW, STUDY, committed, completed
from tco_final_ln_protocol import IMAGE, ROOT, read, seal, sha


LEDGER = ROOT / "manifests" / (STUDY + "_distillation_teachers.json")
POLICY = ROOT / "manifests" / (STUDY + "_distillation_policy.json")
TRIGGER = ROOT / "results" / STUDY / "a_distillation_trigger.json"
WORKER = ROOT / "scripts/train_tco_two_track_suffix.py"
MATH = ROOT / "scripts/tco_two_track_distill.py"


def plan_for(arm):
    from tco_two_track_distill import training_order
    ledger = read(LEDGER)
    if (ledger.get("registration_sha256") != sha(MASTER) or ledger.get("trigger_sha256") != sha(TRIGGER)
            or read(TRIGGER).get("proceed") is not True):
        raise ValueError("A committed hybrid decision must enable this training branch")
    jobs = training_order(ledger["jobs"])
    mounts, rows = [], []
    for n, job in enumerate(jobs):
        attempt = completed(job)
        if attempt is None:
            raise ValueError("Full40 teacher missing: " + job["id"])
        model = attempt / "model"
        result = read(model / "complete.json")
        keys = ("id", "slug", "room", "visit", "split", "seed", "profile", "arm", "steps",
                "frames", "input_sha256", "input_file_sha256")
        if (any(result.get(k) != job[k] for k in keys) or result["completed_adam_updates"] != 40
                or result["rental_compute_usd"] != 0):
            raise ValueError("Incorrect completed teacher receipt")
        packet = PREP / "model" / (job["slug"] + ".npz")
        teacher = model / "prediction.npz"
        if sha(packet) != job["input_file_sha256"] or sha(teacher) != result["predictions_sha256"]:
            raise ValueError("Invalid teacher/input binding")
        row = {k:job[k] for k in keys}
        row.update(mount=n, teacher_receipt_sha256=sha(model/"complete.json"), teacher_scene_seconds=result["scene_seconds"],
                   teacher_cold_seconds=result["cold_seconds"], predictions_sha256=result["predictions_sha256"])
        rows.append(row)
        mounts.extend(mount(packet, f"/inputs/{n}.npz"))
        if arm == "Distilled":
            mounts.extend(mount(teacher, f"/teachers/{n}.npz"))
    return dict(arm=arm, jobs=rows, ledger_sha256=sha(LEDGER), objective_policy_sha256=sha(POLICY)), mounts


def publish(case, arm):
    model = case / "model"
    terminal = read(case / "terminal.json")
    receipt = read(model / "complete.json")
    artifact = model / "initialization.safetensors"
    digest = sha(artifact)
    if (terminal["status"] != "completed" or terminal["complete_sha256"] != sha(model / "complete.json")
            or receipt["initialization_sha256"] != digest or receipt["optimizer_steps"] != 24
            or receipt["arm"] != arm or receipt["inputs_sha256"] != sha(case / "plan.json")):
        raise ValueError("Unverified training evidence")
    folder = PREP / "initializations"
    folder.mkdir(exist_ok=True, parents=True)
    dest = folder / (digest + ".safetensors")
    if not dest.exists():
        temp = folder / (digest + ".partial")
        with artifact.open("rb") as src, temp.open("wb") as out:
            shutil.copyfileobj(src, out)
        if sha(temp) != digest:
            raise ValueError("Initialization transfer failed")
        temp.rename(dest)
    if sha(dest) != digest:
        raise ValueError("Destination initialization checksum mismatch")
    public = ROOT / "results" / STUDY
    seal(public / ("a_training_" + arm + ".json"),
         dict(arm=arm, initialization_sha256=digest, registration_sha256=sha(MASTER),
              start=read(case / "start.json"), receipt=receipt, inputs=read(case / "plan.json")))
    arms = ("Distilled", "OfflineOriginal")
    files = {a:public / ("a_training_" + a + ".json") for a in arms}
    if all(f.is_file() for f in files.values()):
        both = {a:read(f) for a, f in files.items()}
        if (both[arms[0]]["inputs"]["jobs"] != both[arms[1]]["inputs"]["jobs"] or
                len({v["receipt"]["initialized_factors_sha256"] for v in both.values()}) != 1):
            raise ValueError("Two offline students used different samples or starting factors")
        def sample_order(row):
            return [(e["room"],e["visit"],e["input_sha256"],e["update"],e["sample_seed"])
                    for e in row["receipt"]["training_rows"]]
        if sample_order(both[arms[0]]) != sample_order(both[arms[1]]):
            raise ValueError("Offline training schedules differ")
        seal(public / "a_trained_initializations.json", dict(study=STUDY, registration_sha256=sha(MASTER),
             objective_policy_sha256=sha(POLICY), ledger_sha256=sha(LEDGER), trigger_sha256=sha(TRIGGER),
             initializations={a:both[a]["initialization_sha256"] for a in arms},
             training_receipts_sha256={a:sha(f) for a,f in files.items()}))
    print(arm, digest, flush=True)


def launch(arm):
    plan, bindings = plan_for(arm)
    root = RAW / "offline_training" / arm
    cases = sorted(root.glob("attempt[0-9][0-9]"))
    for case in cases:
        if read(case / "terminal.json")["status"] == "completed":
            if plan != read(case / "plan.json"):
                raise ValueError("Training ledger changed after completion")
            publish(case, arm)
            return
    if cases and (len(cases) >= 2 or read(cases[-1]/"terminal.json")["status"] not in ("interrupted","timeout","provisioning")):
        raise ValueError("Training already reached a recorded terminal outcome")
    check_device_available()
    gpu = device()
    n = len(cases) + 1
    case = root / f"attempt{n:02d}"
    case.mkdir(parents=True, exist_ok=False)
    seal(case/"plan.json", plan)
    name = f"tco-suffix-training-{arm.lower()}-{n}"
    cmd = docker_base(name, True)
    cmd[cmd.index("--label")+1] = "tco.study="+STUDY
    cmd += mount(Path("E:/ResearchData/TCO/checkpoints/VGGT-1B"), "/data/checkpoints/VGGT-1B")
    cmd += bindings + mount(case, "/case", True)
    cmd += [IMAGE, "python", "-B", "-u", "scripts/"+WORKER.name, "--arm", arm,
            "--plan", "/case/plan.json", "--out", "/case/model"]
    seal(case / "start.json", dict(started=utc(), gpu=gpu, image=IMAGE, plan_sha256=sha(case/"plan.json"),
         worker_sha256=sha(WORKER), math_sha256=sha(MATH), revision=subprocess.check_output(
             ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), rental_compute_usd=0))
    cid = None
    code, error, status = None, None, "provisioning"
    start = time.monotonic()
    try:
        cid = subprocess.check_output(cmd, text=True, timeout=45).strip()
        item, = docker_json(["inspect", cid])
        destinations = {m["Destination"] for m in item["Mounts"]}
        all_allowed = {"/workspace/scripts", "/workspace/configs", "/workspace/third_party/TCO",
                       "/root/.cache/torch_extensions", "/data/checkpoints/VGGT-1B", "/case"}
        all_allowed.update(f"/inputs/{j}.npz" for j in range(6))
        if arm == "Distilled":
            all_allowed.update(f"/teachers/{j}.npz" for j in range(6))
        if (item["Image"] != IMAGE or item["HostConfig"]["NetworkMode"] != "none"
                or item["Config"]["Labels"].get("tco.study") != STUDY or not item["HostConfig"]["DeviceRequests"]
                or destinations != all_allowed
                or any(m["RW"] for m in item["Mounts"] if m["Destination"] not in ("/case", "/root/.cache/torch_extensions"))):
            raise RuntimeError("Wrong training isolation boundary")
        seal(case/"container.json", dict(id=cid, destinations=sorted(destinations), image=IMAGE))
        with (case/"stdout.log").open("xb") as log:
            code = subprocess.run(DOCKER + ["start","--attach",cid], stdout=log, stderr=subprocess.STDOUT, timeout=5400).returncode
        status = classify(case/"stdout.log", code, case/"model")
    except subprocess.TimeoutExpired as exc:
        status, error = ("timeout" if cid else "provisioning"), str(exc)
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        error = repr(exc)
    finally:
        if cid:
            stopped, issues = ensure_stopped(cid)
            if not stopped or stopped["State"]["Running"]:
                raise RuntimeError("Could not stop suffix-training container "+str(issues))
        done = case/"model/complete.json"
        seal(case/"terminal.json", dict(status=status, error=error, code=code, stopped=True, ended=utc(),
             complete_sha256=sha(done) if done.exists() else None, wall_seconds=time.monotonic()-start, rental_compute_usd=0))
        if cid:
            subprocess.run(DOCKER+["rm",cid], check=True, timeout=25, capture_output=True)
    if status != "completed":
        raise RuntimeError("Inspect recorded training outcome: " + status)
    publish(case, arm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("Distilled","OfflineOriginal"), required=True)
    args = parser.parse_args()
    committed(MASTER, POLICY, LEDGER, TRIGGER, WORKER, MATH, Path(__file__))
    with host_lock():
        launch(args.arm)
