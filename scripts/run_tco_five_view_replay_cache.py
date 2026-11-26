"""Run a case from the local five-view replay and retention registration."""

import argparse
from contextlib import ExitStack
import faulthandler
import inspect
import os
from pathlib import Path
import re
from unittest.mock import patch

import torch

import run_tco_render_lifetime as inherited
from run_profiled_d0 import read, sha, write_new
from run_tco_one_percent import DATA_ROOT, load_case
from tco_one_percent_protocol import check_case_inputs
from tco_five_view_replay_cache_protocol import PRIVATE, REGISTRATIONS


def source_directory(identifier):
    """Validate the source completion and each of its immutable artifacts."""
    receipts = [read(path) for path in (PRIVATE / "launches" / identifier).glob("attempt*/terminal.json")]
    completed, = [row for row in receipts if row["job"] == identifier and row["verdict"] == "completed"]
    folder = PRIVATE / "cases" / identifier / completed["attempt"] / "case"
    complete = folder / "complete.json"
    if sha(complete) != completed["complete_sha256"]:
        raise ValueError(f"Unverified completion for {identifier}")
    for name, digest in read(complete)["artifact_sha256"].items():
        if Path(name).name != name or sha(folder / name) != digest:
            raise ValueError(f"Unverified source artifact for {identifier}/{name}")
    return folder


def run_model(job, out, supplied, reference):
    from tco_five_view_policy import install_policy

    policy_audit = None
    forward_source = inspect.getsource(inherited.base.tco_module.TCO_VGGT_LoRA.forward)

    def setup(model, _job, state, audit, scope, monitor):
        nonlocal policy_audit
        path, layouts, spans = setup_native(model, dict(job, policy="native"), state, audit, scope, monitor)
        policy_audit = install_policy(
            model, camera=job.get("camera", "native"), retention=job.get("retention", "original"),
            sites=job.get("sites", ()), operation=job.get("operation", "attention"),
            diagnostic=bool(job.get("profile")), update_index=lambda: state["updates"],
            forward_source=forward_source,
            camera_token_sink=(
                lambda update, tokens, grad: torch.save(
                    dict(update=update, camera_tokens=tokens, signed_gradient=grad),
                    out / f"camera_token_{update}.pt",
                )
            ) if job.get("gradient_mode") in {"capture", "replay", "negative"} else None,
        )
        return path, layouts, spans

    def store(path, payload):
        if path.name == "complete.json":
            payload.update(job, camera_checkpoint=job.get("camera") == "checkpoint",
                           policy_execution=policy_audit.report())
            payload["artifact_sha256"].update({
                file.name: sha(file) for file in out.iterdir() if file.is_file() and file.name != path.name
            })
        write_new(path, payload)

    setup_native = inherited.setup_execution
    with ExitStack() as stack:
        stack.enter_context(patch.multiple(inherited, setup_execution=setup, write_new=store))
        mode = job.get("gradient_mode")
        if mode:
            from tco_five_view_gradients import gradient_context

            source = source_directory(job["source_job"]) if job.get("source_job") else None
            q_source = source_directory(job["q_source_job"]) if job.get("q_source_job") else None
            stack.enter_context(gradient_context(inherited, job, out, source=source, q_source=q_source))
        # Keep the inherited native objective, optimizer path and timing boundary.
        run_job = dict(job, policy="native") if not mode else dict(job, policy="native", probe="real")
        inherited.run(run_job, out, supplied, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("attempt")
    parser.add_argument("--registration", choices=REGISTRATIONS, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.job) or not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        parser.error("Invalid case identifier")
    plan = read(REGISTRATIONS[args.registration])
    job, = (item for item in plan["jobs"] if item["id"] == args.job)
    out = PRIVATE / "cases" / args.job / args.attempt / "case"
    out.mkdir(parents=True, exist_ok=False)

    if job.get("backend") == "deterministic":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    probe = job.get("probe")
    if probe in {"camera_fixture", "camera_fixture_gpu"}:
        from tco_five_view_policy import camera_fixture, retention_fixture

        device = "cuda" if probe.endswith("_gpu") else "cpu"
        torch.set_num_threads(1)
        if device == "cuda":
            inherited.full40.warm_device()
        fixture = camera_fixture(device=device, amp=device == "cuda")
        if device == "cpu":
            fixture["retention"] = retention_fixture()
        write_new(out / "complete.json", dict(**job, fixture=fixture, rental_compute_usd=0))
        return
    if job.get("gradient_mode") == "regenerate":
        from tco_five_view_gradients import regenerate

        inherited.full40.warm_device()
        source = source_directory(job["source_job"])
        detail = regenerate(source, out, replica=job["replica"])
        artifacts = {file.name: sha(file) for file in out.iterdir() if file.is_file()}
        write_new(out / "complete.json", dict(**job, gradient=detail, artifact_sha256=artifacts,
                                               rental_compute_usd=0))
        return

    check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
    supplied, reference = load_case(job, out, job["width"])
    if probe == "rescore":
        with patch.object(inherited, "PRIVATE", PRIVATE):
            inherited.rescore(job, out, reference)
        return
    if probe == "reference_substitution":
        reference["pointclouds"] = reference["pointclouds"].copy()
        reference["pointclouds"] *= 0
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        run_model(job, out, supplied, reference)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
