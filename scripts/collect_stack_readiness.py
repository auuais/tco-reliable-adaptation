"""Summarize measured clean-room checks, keeping GPU limitations explicit."""
import datetime
import json
import math
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
rehearsal = pathlib.Path((root / "environment/latest_rehearsal_path.txt").read_text(encoding="utf-8-sig").strip()) / "tco-research"
load = lambda path: json.loads(path.read_text(encoding="utf-8-sig"))
transfer = load(root / "cloud/transfer_manifest.json")
assert load(rehearsal / "cloud/prepared-transfer.json") == transfer
host = load(root / "environment/offline-rehearsal-host-config.json")
assert host["NetworkMode"] == "none"
install = load(rehearsal / "environment/cloud_runtime_install.json")
probe = load(rehearsal / "environment/cloud_renderer_probe.json")
reference = load(root / "environment/renderer_probe.json")
run = load(rehearsal / "runs/rehearsal_precompiled224_s1_seed42/run.json")
assert install["status"] == probe["status"] == "passed" and run["status"] == "complete"
assert probe["required_precompiled_extension"] and run["gsplat_backend"]["module"] == "gsplat.csrc"
assert not run["base_parameters_modified"]
assert math.isclose(probe["loss"], reference["loss"], rel_tol=1e-5, abs_tol=1e-7)
relative_errors = {}
for name, value in probe["gradients"].items():
    expected = reference["gradients"][name]["norm"]
    assert value["finite"] and math.isclose(value["norm"], expected, rel_tol=1e-5, abs_tol=1e-7)
    relative_errors[name] = abs(value["norm"] - expected) / max(abs(expected), 1e-12)
architectures = (rehearsal / "environment/precompiled_cuda_architectures.txt").read_text()
assert "sm_86" in architectures and "sm_89" in architectures
gradients = load(rehearsal / "runs/rehearsal_precompiled224_s1_seed42/gradients.json")
assert len(gradients) == 1 and gradients[0]["finite"]
record = {"status": "ready_for_cloud_launch", "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "local_rehearsal_directory": str(rehearsal), "network_mode_during_rehearsal": "none",
          "base_image": "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel@sha256:3d614dfd422b7e43647491cbf07d6acc516c032fc49c594a94afdebd52552fb9",
          "offline_runtime": install, "compiled_architectures_verified_in_binary": ["sm_86", "sm_89"],
          "renderer_probe_seconds_without_compilation": probe["compile_and_probe_seconds"],
          "renderer_probe_gradient_relative_difference_from_original": relative_errors,
          "real_tco_rehearsal": {"run_id": run["run_id"], "image_shape": run["image_shape"],
                                 "steps": 1, "finite_gradient": True, "base_parameters_unchanged": True,
                                 "seconds": run["inference_and_adaptation_seconds"],
                                 "gpu": run["gpu"], "scope": "Reduced-input setup validation only"},
          "transfer": transfer, "cloud_launched": False, "cloud_spent_usd": 0,
          "remaining_remote_work": ["RunPod provisioning and direct SSH", "Transfer/extract prepared archives",
                                    "Install prebuilt packages", "Fetch pinned checkpoint", "48GB GPU preflight", "Ten full-resolution model jobs"],
          "limitations": ["Physical Ada execution remains untested on the local Ampere GPU",
                          "No full-resolution adaptation result exists yet",
                          "Local install timing is not a prediction of cloud provisioning/transfer time",
                          "Process timeouts do not automatically stop GPU billing"]}
(root / "environment/stack_readiness.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
