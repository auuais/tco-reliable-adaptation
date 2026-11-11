"""Pack and replay compact GPU observations without the checkpoint or raw point maps."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import yaml

from audit_dense_focal_pairs import audit
from evaluation_stability import sha
from prior_ablation_config import effective_config, validate_loss_components
from relative_focal_confirmation import confirmation
from run_dense_focal_pilot import PLAN, ROOT, RUNS
from score_dense_focal_pilot import PUBLIC, RAW, summarize
from score_saved_geometry import read


ARCHIVE = PUBLIC / "gpu_traces.zip"
FILES = ("run.json", "capture.json", "gradients.json", "losses.json", "terminal.json")


def pack(rows):
    if ARCHIVE.exists():
        raise FileExistsError(ARCHIVE)
    temp = ARCHIVE.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w") as archive:
        for row in rows:
            tag = row["tag"]
            for name in FILES:
                source = RAW / "terminal" / (tag + ".json") if name == "terminal.json" else RUNS / tag / name
                info = zipfile.ZipInfo(f"{tag}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    verify(rows, temp)
    temp.replace(ARCHIVE)


def verify_run(row, folder, plan):
    job = row["job"]
    run = read(folder / "run.json")
    terminal = read(folder / "terminal.json")
    digest = sha(folder / "run.json")
    if (digest != row["run_sha256"] or digest != terminal["run_sha256"] or terminal["job"] != job
            or terminal["attempt"] != row["attempt"] or terminal["status"] != "complete"
            or run["job"] != job or run["status"] != "complete" or run["study_sha256"] != sha(PLAN)
            or run["checkpoint_sha256"] != plan["model_sha256"] or run["optimizer_updates"] != plan["steps"]
            or run["base_parameters_changed"] or not run["model_fresh"]):
        raise ValueError(f"Trajectory identity differs in {row['tag']}")
    config_path = ROOT / 'configs/7scenes_baseline.yaml'
    if sha(config_path) != plan['sources']['configs/7scenes_baseline.yaml']:
        raise ValueError('Baseline configuration differs from the frozen design')
    expected = effective_config(yaml.safe_load(config_path.read_bytes()), 1., 1., plan['steps'], job['seed'], plan['lr'])
    if run['configuration'] != expected:
        raise ValueError(f"Camera, visual, adapter, or optimizer settings differ in {row['tag']}")
    for key in ("capture", "losses", "gradients"):
        if sha(folder / f"{key}.json") != run[f"{key}_sha256"]:
            raise ValueError(f"{key} trace does not match its trajectory")
    for name in ("initial", "final", "peak_allocated_bytes", "instrumented_inference_seconds", "cpu_fp32_decision"):
        if row[name] != run[name]:
            raise ValueError(f"{name} differs from the sealed observation")
    losses, grads = (read(folder / name) for name in ("losses.json", "gradients.json"))
    for values in (losses, grads):
        if [v["step"] for v in values] != list(range(plan["steps"])):
            raise ValueError("Incomplete optimizer trajectory")
    for loss, grad in zip(losses, grads):
        if not np.isfinite(list(loss.values())).all() or grad["tensors"] != 384 or not 0 < grad["post_clip_norm"] < np.inf:
            raise ValueError("Invalid optimizer observation")
        validate_loss_components(run["configuration"], loss)
    d = run["decision"]
    supplied = np.asarray(d["supplied_focal"], dtype=np.float32)
    view, choices = confirmation(supplied, np.asarray(d["model_focal"], dtype=np.float32), bool(d["geometric_support"]))
    edit = (job["arm"] == "consensus" and view is not None) or (job["arm"] == "adaptive" and choices[plan["method"]])
    if (d["arm"] != job["arm"] or d["method"] != plan["method"] or d["model_choices"] != choices
            or d["view"] != view or d["edited"] != edit or row["edited"] != edit):
        raise ValueError("Logged focal choice differs from camera-head replay")
    if edit:
        supplied[view] = np.median(np.delete(supplied, view, axis=0), axis=0)
    np.testing.assert_array_equal(supplied, d["used_focal"])


def verify(rows, path):
    plan = read(PLAN)
    seal_path = PUBLIC / "seal.json"
    if [r["job"] for r in rows] != plan["jobs"] or len(rows) != 72:
        raise ValueError("Missing executions or unexpected order")
    expected = {f"{r['tag']}/{name}" for r in rows for name in FILES}
    with zipfile.ZipFile(path) as archive, TemporaryDirectory() as project:
        if set(archive.namelist()) != expected or len(archive.infolist()) != len(expected):
            raise ValueError("Unexpected public trace files")
        root = Path(project)
        for row in rows:
            job = row["job"]
            if row["tag"] != job["run_id"] + "_a" + str(row["attempt"]) or row["attempt"] not in (1, 2):
                raise ValueError("Unexpected execution tag")
            folder = root / row["tag"]
            folder.mkdir()
            for name in FILES:
                (folder / name).write_bytes(archive.read(f"{row['tag']}/{name}"))
            verify_run(row, folder, plan)
        summary = read(PUBLIC / "summary.json")
        if summary["seal_sha256"] != sha(seal_path) or summary["results"] != summarize(summary["rows"]):
            raise ValueError("Published primary result differs from its scores")
        if len(summary["rows"]) != len(rows) or any(
            a != {k: b[k] for k in a} for a, b in zip(rows, summary["rows"])
        ):
            raise ValueError("Scored trajectories and sealed observations differ")
        if audit(root) != read(PUBLIC / "same_target_pairs.json"):
            raise ValueError("Matched controls do not replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "verify"))
    args = parser.parse_args()
    seal = read(PUBLIC / "seal.json")
    if seal["manifest_sha256"] != sha(PLAN):
        raise ValueError("Seal refers to another design")
    if args.action == "pack":
        pack(seal["rows"])
    else:
        verify(seal["rows"], ARCHIVE)
    print("Replayed 72 runs, 2,880 optimizer updates, 24 matched controls and both reference comparisons.")
