"""Registered model engineering check with no optimizer or development scoring."""
import argparse
import random
from pathlib import Path
import signal
import time
import traceback

from profiled_compensation_protocol import (ROOT, PUBLIC, INVENTORY, THRESHOLDS, ENGINEERING,
    file_identity, read, require, verify_engineering)
from run_profiled_d0 import now, write_new
from run_profiled_lora import scientific_gpu_lock


def run(weights, attempt):
    import numpy as np
    import torch
    from check_profiled_lora_bridge import compare_gradients, gradient_summary, gradients
    from mast3r.model import AsymmetricMASt3R
    from mast3r_network_priors import digest, install
    from profiled_compensation_training import loss_for, output_adjoints, prepare
    from profiled_d1_artifacts import load_adapter, load_teacher
    from profiled_d1_bridge import verify_teacher_metadata
    from profiled_d1_capture import (ORIGINAL, INITIAL, FINAL, TorchCaptureOps,
                                     counted_method, tensor_sequence_digest)
    from profiled_network import (PAIRS, collect_points, fuse, load_case, pair_forward,
                                   replay_backward, training_features, views_for)
    from profiled_runtime import configure_runtime, verify_runtime
    from run_profiled_d1 import check_gpu, exact_raster

    started = time.monotonic()
    manifest, head = verify_engineering(weights)
    runtime = check_gpu()
    spec = read(ROOT / THRESHOLDS)
    inventory = read(ROOT / INVENTORY)
    limits = manifest["limits"]
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    model = AsymmetricMASt3R.from_pretrained(str(weights), local_files_only=True).eval().cuda().requires_grad_(False)
    original = list(model.state_dict().items())
    require(digest(original) == ORIGINAL, "Fresh model identity differs")
    ops = TorchCaptureOps(model, ROOT / "third_party/TCO/peft/lora.py")
    case = load_case(manifest["scene"], manifest["adaptation_source"], device="cuda")
    _, _, views = views_for(manifest["scene"]+"_base")
    features = training_features(model, views)
    feature_hash = tensor_sequence_digest(features)
    varying = tuple(tuple(p) for p in manifest["varying_pairs"])

    def guard():
        require(time.monotonic()-started < limits["wall_seconds"], "Bridge wall cap exceeded")
        require(torch.cuda.max_memory_allocated()/2**20 <= limits["cuda_allocated_mib"]
                and torch.cuda.max_memory_reserved()/2**20 <= limits["cuda_reserved_mib"],
                "Bridge reserved or allocated memory cap exceeded")

    def progress(count, name, payload):
        write_new(attempt / f"{name}.json", dict(**payload, decoder_calls_entered=count["calls"]))
        print(f"{name} complete at {count['calls']} decoder calls", flush=True)

    def retained(constants, prepared, arm, fold, precision):
        model.zero_grad(set_to_none=True)
        points = dict(constants)
        for pair in varying:
            points[pair], _ = pair_forward(model, features, pair, precision)
        loss, _, _ = loss_for(points, prepared, arm, fold)
        loss.backward()
        return float(loss.detach()), gradients(model)

    with counted_method(model, "_decoder", limits["total_decoder_calls"], guard) as count:
        pristine, confidence = collect_points(model, features)
        teacher = load_teacher(ROOT, inventory)
        depth, fused_confidence = fuse(pristine, confidence)
        rasters = {key: exact_raster(teacher[key], value.cpu().numpy())
                   for key, value in (("depth", depth), ("confidence", fused_confidence))}
        progress(count, "teacher", dict(rasters=rasters))
        require(all(row["equal"] for row in rasters.values()), "Archived pristine teacher differs")
        torch.manual_seed(42)
        modules = install(model, ROOT / "third_party/TCO/peft/lora.py")
        model.eval()
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        ops.validate_initial(modules, named)
        require(digest(named) == INITIAL, "Zero-adapter state differs from D1")
        zero, zero_conf = collect_points(model, features)
        require(all(torch.equal(pristine[p][k], zero[p][k]) for p in PAIRS for k in (0, 1))
                and all(torch.equal(confidence[p], zero_conf[p]) for p in PAIRS),
                "Exact zero-adapter equivalence failed")
        del zero, zero_conf
        prepared = prepare(case, pristine, confidence)
        teacher_meta = read(ROOT / inventory["teacher_metadata"]["file"])["states"]
        current = [dict(support_fold=fold, gains=s["gains"].cpu().tolist(),
                        sensor_normalizers=s["sensor_normalizers"].cpu().tolist(),
                        ridge=float(s["ridge"]), support_counts=[len(p["xy"]) for p in s["support"]],
                        query_counts=[len(p["xy"]) for p in prepared["queries"][fold]])
                   for fold, s in enumerate(prepared["states"])]
        metadata = verify_teacher_metadata(current, teacher_meta, rtol=1e-12, atol=1e-12)
        progress(count, "teacher_preparation", metadata)
        initial_objectives = []
        initial_gradient = None
        for arm in ("P3", "R0", "R1"):
            model.zero_grad(set_to_none=True)
            loss, scales, adjoints, terms = output_adjoints(pristine, prepared, arm, 0, measure=True)
            replay_backward(model, features, adjoints)
            g = gradients(model)
            g_hash = digest(g.items())
            require(terms["penalty"] == 0 and terms["compensation_l2"] == 0,
                    "Pristine redesign compensation or penalty does not vanish")
            if initial_gradient is None:
                initial_gradient = (float(loss), scales.cpu().tolist(), g_hash)
                gradient_summary(g)
            require((float(loss), scales.cpu().tolist(), g_hash) == initial_gradient,
                    "Initial P3/R0/R1 loss, scales or full 20-pair gradient differ")
            initial_objectives.append(dict(arm=arm, loss=float(loss), gradient_sha256=g_hash))
            del adjoints, g
        progress(count, "initial_full_gradients", dict(arms=initial_objectives, exact=True))
        model.zero_grad(set_to_none=True)
        ops.assign(named, load_adapter(ROOT, inventory))
        require(digest(named) == FINAL, "Engineering nonzero adapter differs from D1")
        nonzero, _ = collect_points(model, features)
        forward_rows, replay_rows = [], []
        for fold in (0, 1):
            with torch.no_grad():
                full, s_full, stats_full = loss_for(nonzero, prepared, "R1", fold, measure=True)
                detach, s_detach, stats_detach = loss_for(nonzero, prepared, "R2", fold, measure=True)
            require(torch.equal(full, detach) and torch.equal(s_full, s_detach) and
                    stats_full == stats_detach and stats_full["penalty"] > 0,
                    "R1/R2 nonzero-state forward match failed")
            forward_rows.append(dict(fold=fold, loss=float(full), scales=s_full.cpu().tolist(),
                                     terms=stats_full, exact=True))
            for arm in ("R1", "R2"):
                loss_retained, retained_g = retained(nonzero, prepared, arm, fold, "bfloat16")
                model.zero_grad(set_to_none=True)
                loss, _, adjoints, _ = output_adjoints(nonzero, prepared, arm, fold, varying)
                replay_backward(model, features, adjoints)
                g = gradients(model)
                gradient_summary(g)
                result = compare_gradients(retained_g, g)
                replay_rows.append(dict(fold=fold, arm=arm, retained_loss=loss_retained,
                                        replay_loss=float(loss), **result))
                progress(count, f"replay_fold{fold}_{arm}", replay_rows[-1])
                require(float(loss) == loss_retained
                        and result["relative_l2"] <= spec["replay_relative_gradient_l2_max"]
                        and result["cosine"] >= spec["replay_cosine_min"],
                        "Two-pair retained and replayed gradients disagree")
                del adjoints, retained_g, g
        progress(count, "forward_match", dict(rows=forward_rows))

        configure_runtime(allow_tf32=False)
        with torch.no_grad():
            fixed, _ = collect_points(model, features, precision="float32")
        differences = []
        for fold in (0, 1):
            _, g = retained(fixed, prepared, "R1", fold, "float32")
            gradient_summary(g)
            name = max(g, key=lambda n:float(g[n].abs().max()))
            index = int(g[name].abs().argmax())
            derivative = float(g[name].reshape(-1)[index])
            parameter = dict(named)[name].reshape(-1)[index]
            saved = parameter.detach().clone()
            del g
            model.zero_grad(set_to_none=True)
            rows = []
            with torch.no_grad():
                try:
                    for step in spec["finite_difference_steps"]:
                        loss_values = []
                        for sign in (-1, 1):
                            parameter.copy_(saved+sign*step)
                            points = dict(fixed)
                            new, _ = collect_points(model, features, precision="float32", pairs=varying)
                            points.update(new)
                            value, _, _ = loss_for(points, prepared, "R1", fold)
                            loss_values.append(float(value))
                        finite = (loss_values[1]-loss_values[0])/(2*step)
                        limit = spec["finite_difference_absolute_tolerance"] + (
                            spec["finite_difference_relative_tolerance"]*abs(derivative))
                        rows.append(dict(epsilon=step, minus=loss_values[0], plus=loss_values[1],
                                         finite_difference=finite, absolute_error=abs(finite-derivative),
                                         tolerance=limit, pass_check=abs(finite-derivative)<=limit))
                finally:
                    parameter.copy_(saved)
            passed = sum(r["pass_check"] for r in rows) >= spec["finite_difference_minimum_passes"]
            passed = passed and rows[-1]["pass_check"]
            differences.append(dict(fold=fold, parameter=name, flat_index=index,
                                    derivative=derivative, rows=rows, pass_check=passed))
            progress(count, f"finite_differences_fold{fold}", differences[-1])
            require(passed, "Real adapter finite differences failed original tolerances")

    configure_runtime()
    verify_runtime()
    guard()
    require(digest(original) == ORIGINAL and all(p.grad is None for p in model.parameters()),
            "Original state changed or residual gradients remain")
    require(digest(named) == FINAL and tensor_sequence_digest(features) == feature_hash,
            "Saved adapter or encoder features changed during the check")
    require(count["calls"] == 184, "Unexpected total decoder inventory")
    verify_engineering(weights)
    report = dict(schema="profiled_compensation_bridge_v1", pass_all=True,
                  execution_commit=head, engineering=file_identity(ENGINEERING), runtime=runtime,
                  original_state_unchanged=True, zero_adapter_exact=True, nonzero_state_restored=True,
                  zero_initial_full_gradient_match=initial_objectives, forward_matched=forward_rows,
                  retained_replay=replay_rows, parameter_finite_differences=differences,
                  decoder_calls=count["calls"], optimizer_updates=0, accuracy_scoring_performed=False,
                  rental_compute_usd=0, elapsed_seconds=time.monotonic()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    write_new(attempt / "report.json", report)
    print("Compensation engineering bridge passed", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempt", required=True, choices=("bridge01", "bridge02", "bridge03"))
    p.add_argument("--weights", type=Path, default=Path("/weights"))
    args = p.parse_args()
    out = ROOT / PUBLIC / "engineering" / args.attempt

    def timeout(_signal, _frame):
        raise TimeoutError("Compensation bridge exceeded its wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(1800)
    with scientific_gpu_lock(dict(run_id="compensation_bridge", attempt=args.attempt)):
        out.mkdir(parents=True, exist_ok=False)
        write_new(out / "started.json", dict(started_at_utc=now(), engineering=file_identity(ENGINEERING),
                                            optimizer_updates=0, rental_compute_usd=0))
        try:
            run(args.weights, out)
        except Exception as exc:
            write_new(out / "failure.json", dict(pass_all=False, error_type=type(exc).__name__,
                      message=str(exc), traceback=traceback.format_exc(), optimizer_updates=0))
            raise


if __name__ == "__main__":
    main()
