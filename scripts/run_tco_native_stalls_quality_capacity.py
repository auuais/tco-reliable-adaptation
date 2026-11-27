"""Run registered local TCO cases with observed Adam and readout counts."""

import argparse
from contextlib import ExitStack, nullcontext
import faulthandler
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

import numpy as np
import torch

from run_profiled_d0 import read, sha, write_new
import run_tco_render_lifetime as lifetime
from run_tco_frozen_prefix import signature
from run_tco_one_percent import DATA_ROOT, load_case
from tco_native_stalls_quality_capacity_protocol import GSPLAT, PRIVATE, REGISTRATIONS, STUDY, verify
from tco_one_percent_protocol import ROOT, check_case_inputs
from tco_render_lifetime import EARLY_CLEAR, clear_gradients


PRIOR_STUDY = "tco_five_view_replay_cache_v1"
MODEL_POLICIES = {"native", "camera", "n", "g", "ng"}
SCORE_FIELDS = ("accuracy_mean_m", "completeness_mean_m")


def source_case(identifier, study=STUDY):
    if study not in {STUDY, PRIOR_STUDY} or not re.fullmatch(r"[a-z0-9_]+", identifier):
        raise ValueError("Unknown source identity")
    private = ROOT / "reports" / study
    attempts = private / "launches" / identifier
    found = []
    for path in sorted(attempts.glob("attempt*/terminal.json")):
        receipt = read(path)
        if (receipt["job"], receipt["attempt"]) != (identifier, path.parent.name):
            raise ValueError("Changed source attempt identity")
        if receipt["verdict"] == "completed":
            case = private / "cases" / identifier / receipt["attempt"] / "case"
            complete = case / "complete.json"
            if not receipt["stopped"] or sha(complete) != receipt["complete_sha256"]:
                raise ValueError("Unverified source completion")
            found.append((case, read(complete)))
    if len(found) != 1:
        raise ValueError(f"Expected one successful source for {identifier}")
    return found[0]


def checked_artifact(case, result, name):
    path = case / name
    if result["artifact_sha256"].get(name) != sha(path):
        raise ValueError(f"Changed source artifact {path.name}")
    return path


def decoded_input(supplied):
    return signature(dict(rgb=supplied["images"].numpy(), poses=supplied["extrs"],
                          intrinsics=supplied["intrs"]))


def rescore(job, out, supplied, reference):
    if torch.cuda.is_available():
        raise ValueError("Rescore cases require a CPU-only container")
    case, source = source_case(job["source_job"])
    for field in ("dataset", "scene", "frames", "width", "seed"):
        if job[field] != source[field]:
            raise ValueError(f"Rescore source mismatch in {field}")
    if decoded_input(supplied) != source["input_sha256"]:
        raise ValueError("Rescore decoded inputs changed")
    prediction = checked_artifact(case, source, "prediction.npz")

    def store(path, result):
        deltas_mm = {name: 1000 * (result["metrics"][name] - source["metrics"][name]) for name in SCORE_FIELDS}
        if not all(math.isfinite(value) for value in deltas_mm.values()):
            raise ValueError("Rescore returned a nonfinite error")
        result.update(
            deltas_mm=deltas_mm, rescore_passed=max(map(abs, deltas_mm.values())) <= 0.01,
            source_completion_sha256=sha(case / "complete.json"), source_prediction_sha256=sha(prediction),
            input_sha256=source["input_sha256"],
        )
        write_new(path, result)

    with patch.multiple(lifetime, PRIVATE=PRIVATE, write_new=store):
        lifetime.rescore(job, out, reference)


def renderer_probe(job, out):
    from check_tco_render_lifetime import renderer_fixture

    lifetime.full40.warm_device()
    case, source = source_case(job["source_job"], job.get("source_study", STUDY))
    names = ("renderer_inputs_step1.pt", "renderer_inputs_step2.pt")
    paths = [checked_artifact(case, source, name) for name in names]
    with TemporaryDirectory(prefix="tco-renderer-leaves-") as directory:
        first = Path(directory) / "renderer_inputs.pt"
        first.symlink_to(paths[0])
        first.with_name("renderer_inputs_1.pt").symlink_to(paths[1])
        result = renderer_fixture(first)
    write_new(out / "complete.json", dict(
        **job, renderer=result, source_completion_sha256=sha(case / "complete.json"), rental_compute_usd=0))


def observe_optimizer(model, job, receipt, check_lifetime):
    """Observe counters at the final native Adam call without retaining its buffers."""
    setup = model.backbone_peft_setup
    aggregate = model.aggregate_with_patch_tokens
    active = []
    if check_lifetime:
        receipt["pre_forward_gradients"] = []

        def observe_forward(*args, **kwargs):
            if kwargs.get("use_checkpoint"):
                retained = sum(param.grad is not None for param in active)
                receipt["pre_forward_gradients"].append(retained)
                if job["policy"] in EARLY_CLEAR and retained:
                    raise ValueError("Consumed gradients remained live at the next forward")
            return aggregate(*args, **kwargs)

        model.aggregate_with_patch_tokens = observe_forward

    def setup_observed(*args, **kwargs):
        optimizer, params = setup(*args, **kwargs)
        if not isinstance(optimizer, torch.optim.Adam) or len(params) != 384:
            raise ValueError("Unexpected optimizer configuration")
        if check_lifetime:
            active.extend(params)

        def on_step(adam, _args, _kwargs):
            receipt["step_calls"] += 1
            if receipt["step_calls"] != job["steps"]:
                return
            counters = [adam.state[param]["step"] for param in params]
            if any(step.device.type != "cpu" or step.ndim != 0 for step in counters):
                raise ValueError("Expected native scalar CPU Adam counters")
            counts = [float(step) for step in counters]
            receipt.update(factor_states=len(adam.state), counter_min=min(counts), counter_max=max(counts),
                           final_gradients=sum(param.grad is not None for param in params))

        optimizer.register_step_post_hook(on_step)
        return optimizer, params

    model.backbone_peft_setup = setup_observed
    if check_lifetime:
        def returned(_model, _args, _output):
            receipt["gradients_on_return"] = sum(param.grad is not None for param in active)

        model.register_forward_hook(returned)


def run_model(job, out, supplied, reference):
    from tco_five_view_gradients import GradientProbe, gradient_context

    mode = job.get("gradient_mode")
    probe = job.get("probe")
    effective_job = dict(job)
    if mode or probe in {"qualification", "reference_substitution", "real"}:
        effective_job["probe"] = "real"
    elif probe in {"identity", "leak"}:
        effective_job["probe"] = probe
    setup, unproject = lifetime.setup_execution, lifetime.unproject_depth_to_world_points_torch
    score, serialize = lifetime.base.evaluate, np.savez_compressed
    store = lifetime.write_new
    runtime_check = dict(torch="2.7.1+cu128", cuda="12.8", gsplat_sha256=GSPLAT)
    optimizer = dict(step_calls=0, factor_states=0, final_gradients=0, counter_min=None, counter_max=None)
    final = {}
    scores_and_writes = {}

    def install(model, inner_job, state, audit, scope, monitor):
        result = setup(model, inner_job, state, audit, scope, monitor)
        observe_optimizer(model, inner_job, optimizer, check_lifetime=audit is not None)
        return result

    def final_readout(depth, poses, intrinsics):
        views, _, height, width = supplied["images"].shape
        points = unproject(depth, poses, intrinsics)
        if (depth.shape not in {(1, views, height, width), (1, views, height, width, 1)}
                or poses.shape not in {(1, views, 3, 4), (1, views, 4, 4)}
                or intrinsics.shape != (1, views, 3, 3)
                or points.shape != (1, views, height, width, 3) or points.dtype != torch.float32):
            raise ValueError("Wrong view count, raster or dtype in the final reconstruction")
        final.update(views=views, height=height, width=width, geometry_dtype=str(points.dtype))
        return points

    def evaluate(*args, **kwargs):
        result = score(*args, **kwargs)
        if not all(math.isfinite(result[name]) and result[name] >= 0 for name in SCORE_FIELDS):
            raise ValueError("Scorer returned a nonfinite or negative geometry error")
        return result

    def save_archive(file, *args, **kwargs):
        started = time.perf_counter()
        result = serialize(file, *args, **kwargs)
        if isinstance(file, (str, Path)) and Path(file).parent == out:
            scores_and_writes[Path(file).name] = time.perf_counter() - started
        return result

    def save_receipt(path, payload):
        if path.name != "complete.json":
            return store(path, payload)
        steps, views = job["steps"], len(job["frames"])
        expected_layout = [(5,)] if views == 5 else [(8, 2)]
        if (payload["updates"] != steps or optimizer["step_calls"] != steps
                or payload["native_depth_batches"] != expected_layout or final.get("views") != views
                or payload["head_taps"] != [4, 11, 17, 23]
                or (steps and (optimizer["counter_min"] != steps or optimizer["counter_max"] != steps
                               or optimizer["factor_states"] != 384 or optimizer["final_gradients"] != 384))
                or any(payload["runtime"].get(key) != value for key, value in runtime_check.items())):
            raise ValueError("Registered optimizer, view layout, or runtime check failed")
        if optimizer.get("gradients_on_return", 384) != 384 and steps:
            raise ValueError("Final gradient availability changed")
        payload.update(job, optimizer=optimizer, final_readout=final,
                       serialization_seconds=scores_and_writes)
        if mode:
            payload["artifact_sha256"].update({item.name: sha(item) for item in out.iterdir()
                                               if item.name == "gradient_summary.json"
                                               or item.name.startswith(("anchor_step", "q_step", "renderer_inputs_step"))})
        return store(path, payload)

    with ExitStack() as stack:
        stack.enter_context(patch.multiple(
            lifetime, setup_execution=install, unproject_depth_to_world_points_torch=final_readout,
            write_new=save_receipt, PRIVATE=PRIVATE))
        stack.enter_context(patch.object(lifetime.base, "evaluate", evaluate))
        stack.enter_context(patch.object(np, "savez_compressed", save_archive))
        if mode:
            source = None
            if job.get("source_job"):
                source, capture = source_case(job["source_job"], job.get("source_study", STUDY))
                for name in ("gradient_summary.json", "anchor_step1.pt", "anchor_step2.pt", "q_step1.pt", "q_step2.pt"):
                    checked_artifact(source, capture, name)
            if mode in {"replay", "negative"} and job["policy"] in EARLY_CLEAR:
                restore = GradientProbe.preforward

                def restore_then_clear(gradients):
                    # Restore and verify the signed-q anchor before applying the lifetime change.
                    restore(gradients)
                    clear_gradients(gradients.optimizer)

                stack.enter_context(patch.object(GradientProbe, "preforward", restore_then_clear))
            stack.enter_context(gradient_context(lifetime, job, out, source=source))
        lifetime.run(effective_job, out, supplied, reference)


def fixture(job, out):
    if torch.cuda.is_available():
        raise ValueError("CPU fixtures require a CPU-only container")
    torch.set_num_threads(1)
    if job.get("probe") == "camera_fixture":
        from tco_five_view_policy import camera_fixture

        result = camera_fixture()
    else:
        from check_tco_render_lifetime import cpu_fixture

        result = cpu_fixture()
    write_new(out / "complete.json", dict(**job, fixture=result, rental_compute_usd=0))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("job")
    cli.add_argument("attempt")
    cli.add_argument("--registration", choices=REGISTRATIONS, required=True)
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.job) or not re.fullmatch(r"attempt[0-9]{2}", args.attempt):
        cli.error("Invalid case or attempt identity")
    plan = verify(args.registration)
    job, = (row for row in plan["jobs"] if row["id"] == args.job)
    if (job["policy"] not in MODEL_POLICIES or job["arm"] != "four"
            or len(job["frames"]) not in {5, 10} or len(set(job["frames"])) != len(job["frames"])):
        raise ValueError("Invalid native-batched experiment")
    out = PRIVATE / "cases" / args.job / args.attempt / "case"
    out.mkdir(parents=True, exist_ok=False)
    probe, profile = job.get("probe"), job.get("profile")
    if job.get("backend") == "deterministic" or probe in {"fixed", "leak"}:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    if profile == "fixture":
        fixture(job, out)
        return
    if probe == "renderer":
        renderer_probe(job, out)
        return
    context = nullcontext()
    if profile == "timeline":
        from tco_native_stall_trace import instrument_lifetime

        context = instrument_lifetime(lifetime, out, inventory=len(job["frames"]) == 10)
    faulthandler.dump_traceback_later(90, repeat=True)
    try:
        with context as trace:
            with trace.phase("load_case", cuda=False) if trace else nullcontext():
                check_case_inputs(job, plan["input_source_sha256"], DATA_ROOT)
                supplied, reference = load_case(job, out, job["width"])
            if probe == "rescore":
                rescore(job, out, supplied, reference)
            else:
                if probe == "reference_substitution" and job.get("dummy_reference", True):
                    if job.get("score"):
                        raise ValueError("A dummy reference must never supply a score")
                    reference["valid_mask"] = np.zeros_like(reference["valid_mask"])
                    reference["pointclouds"] = np.ones_like(reference["pointclouds"])
                run_model(job, out, supplied, reference)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
