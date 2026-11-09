"""Run a fresh-model prior/budget diagnostic, leaving run_baseline.py immutable.

Invokes upstream inference and loss code. Instrumentation only records losses,
gradients, time and memory. A fresh model per invocation prevents adapter carryover.
Ground-truth depth is used exclusively after prediction for shape evaluation.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
import traceback

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["MPLBACKEND"] = "Agg"
ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "TCO"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(UPSTREAM / "base_models" / "vggt"))

import numpy as np
import open3d as o3d
import torch
from omegaconf import OmegaConf

from datasets.sevenscenes import SevenScenes
from evaluation.mv_recon.utils import accuracy, completion, umeyama
from utils.interfaces import infer_mv_pointclouds_w_priors_v2
import tco_vggt_lora as tco_module
from prior_corruptions import PROFILES, corrupt_priors


def dump(path, content):
    pathlib.Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def evaluate(pred, data, output):
    gt = data["pointclouds"]
    mask = data["valid_mask"]
    assert pred.shape == gt.shape
    assert np.isfinite(pred).all(), "Nonfinite prediction"
    scale, rotation, translation = umeyama(pred[mask].T, gt[mask].T)
    aligned = scale * np.einsum("nhwj,ij->nhwi", pred, rotation) + translation.T
    colors = data["images"].permute(0, 2, 3, 1).numpy()[mask].reshape(-1, 3)
    pc_pred, pc_gt = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    pc_pred.points = o3d.utility.Vector3dVector(aligned[mask].reshape(-1, 3))
    pc_gt.points = o3d.utility.Vector3dVector(gt[mask].reshape(-1, 3))
    pc_pred.colors = o3d.utility.Vector3dVector(colors)
    pc_gt.colors = o3d.utility.Vector3dVector(colors)
    # Match upstream: save the PLY before ICP, then use rigid ICP for metrics.
    # Compact diagnostic evidence retains raw predictions; omit regenerable PLYs.
    icp = o3d.pipelines.registration.registration_icp(
        pc_pred, pc_gt, 0.1, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pc_pred.transform(icp.transformation)
    pc_pred.estimate_normals()
    pc_gt.estimate_normals()
    normal_pred, normal_gt = np.asarray(pc_pred.normals), np.asarray(pc_gt.normals)
    acc, acc_med, nc1, nc1_med = accuracy(pc_gt.points, pc_pred.points, normal_gt, normal_pred)
    comp, comp_med, nc2, nc2_med = completion(pc_gt.points, pc_pred.points, normal_gt, normal_pred)
    result = {"accuracy_mean_m": float(acc), "accuracy_median_m": float(acc_med),
              "completeness_mean_m": float(comp), "completeness_median_m": float(comp_med),
              "normal_consistency_mean": float((nc1 + nc2) / 2),
              "normal_consistency_median": float((nc1_med + nc2_med) / 2),
              "nc1_mean": float(nc1), "nc2_mean": float(nc2),
              "valid_points": int(mask.sum()), "similarity_scale": float(scale),
              "similarity_rotation": rotation.tolist(), "similarity_translation": translation.tolist(),
              "icp_transformation": icp.transformation.tolist(),
              "protocol": "upstream_Umeyama_then_rigid_ICP_0.1m_then_nearest_neighbor"}
    dump(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/data/7scenes")
    parser.add_argument("--checkpoint", default="/data/checkpoints/VGGT-1B")
    parser.add_argument("--dataset-cache", default=None,
                        help="Optional cache generated from the full prepared dataset; allows transferring only selected frames.")
    parser.add_argument("--scene", default="heads/seq-01")
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 200])
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--prior-profile", choices=list(PROFILES), default="clean")
    parser.add_argument("--prior-seed", type=int, default=1729)
    parser.add_argument("--prior-weight", type=float, default=1.0)
    parser.add_argument("--study-id", default="diagnostics_v1")
    args = parser.parse_args()
    assert args.steps >= 0 and 0 <= args.prior_weight <= 1
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(ROOT / "configs" / "7scenes_baseline.yaml")
    cfg.tco_steps, cfg.seed = args.steps, args.seed
    cfg.lambda_pose *= args.prior_weight
    cfg.lambda_intrinsics *= args.prior_weight
    config = OmegaConf.to_container(cfg, resolve=True)
    upstream_commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    record = {"run_id": args.run_id, "status": "running", "arguments": vars(args),
              "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "upstream_commit": upstream_commit, "configuration": config,
              "scope": "debug" if args.size != 518 or args.frames != [0, 200, 400, 600, 800] else "exploratory_prior_budget_diagnostics",
              "prior_provenance": "reference_priors" if args.prior_profile == "clean" else "synthetically_perturbed_reference_priors",
              "study_id": args.study_id,
              "evaluation_only": "registered_depth_and_reference_pointcloud",
              "fresh_model_per_run": True,
              "versions": {name: importlib.metadata.version(name) for name in
                           ["torch", "torchvision", "gsplat", "numpy", "open3d", "huggingface-hub"]}}
    dump(output / "run.json", record)
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        o3d.utility.random.seed(args.seed)
        record["gpu"] = torch.cuda.get_device_name()
        record["cuda_version"] = torch.version.cuda
        from gsplat.cuda._backend import _C
        extension_path = pathlib.Path(_C.__file__)
        record["gsplat_backend"] = {"module": _C.__name__, "path": str(extension_path),
                                    "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest()}
        record["gpu_free_total_bytes_before"] = list(torch.cuda.mem_get_info())
        cache_file = output / "dataset_metadata.npy"
        if args.dataset_cache:
            source_cache = pathlib.Path(args.dataset_cache)
            shutil.copyfile(source_cache, cache_file)
            record["dataset_cache_sha256"] = hashlib.sha256(source_cache.read_bytes()).hexdigest()
        dataset = SevenScenes(SEVENSCENES_DIR=args.data_root, split="test", load_img_size=args.size,
                              cache_file=str(cache_file))
        data = dataset.get_data(sequence_name=args.scene, ids=args.frames)
        # Corrupt copies only. The original data remains the evaluator's reference.
        prior_e, prior_k, prior_metadata = corrupt_priors(data["extrs"], data["intrs"],
                                                         args.prior_profile, args.prior_seed, args.scene)
        np.savez_compressed(output / "supplied_priors.npz", extrinsics=prior_e, intrinsics=prior_k)
        dump(output / "prior_metadata.json", prior_metadata)
        record["prior_metadata"] = prior_metadata
        record["prior_generator_sha256"] = hashlib.sha256((ROOT / "scripts/prior_corruptions.py").read_bytes()).hexdigest()
        record["image_shape"] = list(data["images"].shape)
        record["input_sha256"] = {str(path): hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
                                  for path in data["image_paths"]}
        record["reference_sha256"] = {
            str(pathlib.Path(args.data_root) / args.scene / f"frame-{frame:06d}{suffix}"):
            hashlib.sha256((pathlib.Path(args.data_root) / args.scene / f"frame-{frame:06d}{suffix}").read_bytes()).hexdigest()
            for frame in args.frames for suffix in (".pose.txt", ".depth.proj.png")}
        record["runner_sha256"] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        record["upstream_method_sha256"] = {
            name: hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest()
            for name in ["tco_vggt_lora.py", "tco_loss.py", "objective.py", "constraints.py", "peft/lora.py", "utils/interfaces.py"]}
        record["compatibility_repairs"] = ["CPU bfloat16-to-float32 cast before NumPy export; no numerical values changed"]
        source_metadata = pathlib.Path(args.checkpoint) / "model.safetensors.download.json"
        record["checkpoint_download"] = json.loads(source_metadata.read_text())
        evaluation_arrays = dict(images=data["images"].numpy(), reference_points=data["pointclouds"],
                                 valid_mask=data["valid_mask"], intrinsics=data["intrs"], extrinsics=data["extrs"])
        input_digest = hashlib.sha256()
        for key, value in sorted(evaluation_arrays.items()):
            input_digest.update(key.encode()); input_digest.update(str(value.shape).encode())
            input_digest.update(str(value.dtype).encode()); input_digest.update(np.ascontiguousarray(value).tobytes())
        shared_inputs = ROOT / "runs/diagnostic_inputs" / input_digest.hexdigest()
        shared_inputs.mkdir(parents=True, exist_ok=True)
        inputs_file = shared_inputs / "evaluation_inputs.npz"
        if not inputs_file.exists():
            np.savez_compressed(inputs_file, **evaluation_arrays)
        record["evaluation_inputs_relative"] = str(inputs_file.relative_to(ROOT))
        record["evaluation_tensor_sha256"] = input_digest.hexdigest()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = tco_module.TCO_VGGT_LoRA.from_pretrained(args.checkpoint, enable_track=False).to("cuda").eval()
        torch.cuda.synchronize()
        record["model_load_seconds"] = time.perf_counter() - load_start
        record["model_load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["base_parameter_count"] = sum(p.numel() for p in model.parameters())
        record["stage"] = "model_loaded"
        dump(output / "run.json", record)
        base_versions = {id(p): (name, p._version) for name, p in model.named_parameters()}
        loss_records, gradient_records = [], []
        original_loss = tco_module.compute_tco_losses

        def logged_loss(*positional, **keyword):
            total, components = original_loss(*positional, **keyword)
            values = {key: float(value.detach()) for key, value in components.items() if torch.is_tensor(value) and value.numel() == 1}
            values["total"] = float(total.detach())
            values["step"] = len(loss_records)
            if not all(np.isfinite(value) for value in values.values()):
                raise FloatingPointError("Nonfinite adaptation loss")
            loss_records.append(values)
            dump(output / "losses.json", loss_records)
            return total, components

        tco_module.compute_tco_losses = logged_loss
        original_setup = model.backbone_peft_setup

        def logged_setup(*positional, **keyword):
            optimizer, params = original_setup(*positional, **keyword)
            record["adapter_parameter_count"] = sum(p.numel() for p in params)
            adapter_names = [name for name, p in model.named_parameters() if "lora_A" in name or "lora_B" in name]
            record["adapter_parameter_names"] = adapter_names
            record["stage"] = "adapters_initialized"
            dump(output / "run.json", record)

            def before_step(opt, positional_args, keyword_args):
                present = [p.grad for group in opt.param_groups for p in group["params"] if p.grad is not None]
                finite = bool(present) and all(bool(torch.isfinite(g).all()) for g in present)
                squared = sum(float(g.float().square().sum()) for g in present)
                entry = {"step": len(gradient_records), "finite": finite,
                         "gradient_norm_after_upstream_clipping": squared ** 0.5 if np.isfinite(squared) else None,
                         "tensors_with_gradient": len(present)}
                gradient_records.append(entry)
                dump(output / "gradients.json", gradient_records)
                if not finite or squared == 0:
                    raise FloatingPointError("Missing, zero, or nonfinite adapter gradients")
            optimizer.register_step_pre_hook(before_step)
            return optimizer, params

        model.backbone_peft_setup = logged_setup
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Pass only images and declared pose/intrinsics priors to inference.
        permitted = {"images": data["images"], "extrs": prior_e, "intrs": prior_k}
        predictions = infer_mv_pointclouds_w_priors_v2(permitted, model, cfg, sample_idx=1, scene_name=args.scene.replace('/', '-'))
        torch.cuda.synchronize()
        record["inference_and_adaptation_seconds"] = time.perf_counter() - started
        record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        changed = [name for p in model.parameters() if id(p) in base_versions
                   for name, version in [base_versions[id(p)]] if p._version != version]
        record["base_parameters_modified"] = changed
        assert not changed, f"Base parameters changed: {changed[:3]}"
        assert len(loss_records) == len(gradient_records) == args.steps
        assert np.isfinite(predictions).all()
        np.save(output / "predictions.npy", predictions)
        if not args.skip_evaluation:
            started = time.perf_counter()
            record["metrics"] = evaluate(predictions, data, output)
            record["evaluation_seconds"] = time.perf_counter() - started
        record["status"] = "complete"
        record["utc_end"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(output / "run.json", record)
        print(json.dumps({"run_id": record["run_id"], "status": record["status"],
                          "scope": record["scope"], "image_shape": record["image_shape"],
                          "steps": args.steps,
                          "seconds": record["inference_and_adaptation_seconds"],
                          "peak_allocated_GiB": record["peak_allocated_bytes"] / 2**30,
                          "metrics": {key: value for key, value in record.get("metrics", {}).items()
                                      if key in ["accuracy_mean_m", "completeness_mean_m", "normal_consistency_mean"]}}, indent=2))
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        dump(output / "run.json", record)
        raise


if __name__ == "__main__":
    main()
