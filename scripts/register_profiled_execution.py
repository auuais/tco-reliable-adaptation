"""Register and guard the 36-state calibration-aware LoRA study, without launching it.

Engineering registration pins available bridge source before its GPU checks.
Scientific registration additionally requires a runner and passing gradient and
endpoint bridges bound to the current source, inputs, image and checkpoint.

Preserve earlier registrations after engineering fixes. Use --prepare --output
manifests/lora_profiled_engineering_v2.json --supersedes
manifests/lora_profiled_engineering_v1.json, then --verify --manifest with the new
path. Historical manifests retain their original source hashes and are expected
to fail current-source verification after those sources change.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from prepare_profiled_inputs import ROOT, PLAN, read, sha, build_splits
from profiled_runtime import EXPECTED_RUNTIME


ENGINEERING = ROOT / "manifests/lora_profiled_engineering_v1.json"
SCIENTIFIC = ROOT / "manifests/lora_profiled_execution_v1.json"
INPUT_PATHS = ("manifests/lora_profiled_design_v1.json", "results/lora_profiled_v1/inputs.json",
               "results/lora_profiled_v1/splits.npz", "manifests/lora_profiled_bridge_checks_v1.json")
PREPARATION = ("results/lora_profiled_v1/environment.json", "results/lora_profiled_v1/solver_checks.json",
               "results/lora_profiled_v1/split_checks.json", "results/lora_profiled_v1/baseline_graph.json")
SOURCES = ("scripts/register_profiled_execution.py", "scripts/prepare_profiled_inputs.py", "scripts/profiled_runtime.py",
           "scripts/profiled_calibration.py", "scripts/check_profiled_calibration.py",
           "scripts/profiled_baseline_graph.py", "scripts/profiled_network.py",
           "scripts/check_profiled_lora_bridge.py", "scripts/profiled_final_calibration.py",
           "scripts/check_profiled_endpoints.py", "scripts/mast3r_joint_prior_loss.py",
           "scripts/mast3r_network_priors.py", "scripts/mast3r_network_capture.py",
           "scripts/run_mast3r_neural_transfer.py", "scripts/mast3r_calibrated_alignment.py",
           "scripts/mast3r_alignment_hook.py", "third_party/TCO/peft/lora.py")
RUNNER = "scripts/run_profiled_lora.py"
EVALUATION = "manifests/lora_profiled_evaluation_v1.json"
BRIDGES = {
    "gradient": dict(path="results/lora_profiled_v1/gradient_bridge.json",
        checks=("pass_all", "original_state_unchanged", "zero_adapter_pass", "replay_gradient_pass",
                "finite_difference_pass", "full_macro_step_pass"),
        required_sources=("scripts/profiled_network.py", "scripts/profiled_calibration.py",
                          "scripts/check_profiled_lora_bridge.py", "scripts/mast3r_network_priors.py",
                          "scripts/mast3r_network_capture.py", "scripts/profiled_baseline_graph.py",
                          "third_party/TCO/peft/lora.py")),
    "endpoint": dict(path="results/lora_profiled_v1/endpoint_bridge.json",
        checks=("pass_all", "original_state_unchanged", "native_endpoint_pass", "final_endpoint_pass"),
        required_sources=("scripts/profiled_final_calibration.py", "scripts/profiled_calibration.py",
                          "scripts/mast3r_network_capture.py", "scripts/mast3r_calibrated_alignment.py",
                          "scripts/mast3r_alignment_hook.py", "scripts/mast3r_joint_prior_loss.py",
                          "scripts/check_profiled_endpoints.py")),
}


def relative_file(name):
    """Accept repository-relative file paths only, including paths in bridge maps."""
    if not isinstance(name,str) or Path(name).is_absolute() or "\\" in name:
        raise ValueError(f"Expected a repository-relative POSIX path: {name}")
    path = (ROOT/name).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"Missing or out-of-repository source: {name}")
    return path


def manifest_path(value, default):
    path = default if value is None else value
    path = path if path.is_absolute() else ROOT/path
    path = path.resolve()
    if path.parent != (ROOT/"manifests").resolve() or path.suffix != ".json":
        raise ValueError("Registry path must be a JSON file directly inside the repository manifests directory")
    return path


def jobs(plan, scientific_manifest=None):
    launch_manifest = (SCIENTIFIC if scientific_manifest is None else scientific_manifest).resolve().relative_to(ROOT.resolve()).as_posix()
    arms = plan["arms"]
    if [a["id"] for a in arms] != [f"P{i}" for i in range(6)]:
        raise ValueError("This execution registry requires the declared six-arm factorial")
    result = []
    for scene in plan["scenes"]:
        for side in plan["temporal_fit_sources"]:
            for arm in arms:
                name = f"{scene}_{side}_{arm['id']}"
                result.append(dict(run_id=name, scene=scene, adaptation_source=side,
                    development_evaluation_source="later" if side=="earlier" else "earlier",
                    arm=arm["id"], adapt_lora=arm["adapt_lora"],
                    optimizer_steps=plan["optimization"]["macro_steps"] if arm["adapt_lora"] else 0,
                    pairs_per_step=plan["optimization"]["all_directed_pairs_per_step"] if arm["adapt_lora"] else 0,
                    graph_weight=arm["graph_weight"], differentiate_solve=arm["differentiate_solve"],
                    input_packet=f"results/mast3r_partner_shape/{side}.npz",
                    split_prefixes=[f"{scene}_{side}_{frame}" for frame in plan["target_frames"]],
                    raw_attempt_parent=f"reports/lora_profiled_v1/cases/{name}",
                    public_attempt_parent=f"results/lora_profiled_v1/cases/{name}",
                    planned_argv=["python",RUNNER,"--run-id",name,"--manifest",
                        launch_manifest,"--attempt","FRESH_ATTEMPT_ID","--weights","/weights"],
                    argv_validation="planned; runner CLI must be tested before launch"))
    trained = [job for job in result if job["adapt_lora"]]
    if len(result)!=36 or len(trained)!=30 or any(job["optimizer_steps"]!=8 for job in trained):
        raise ValueError("Unexpected state count or optimization budget")
    if len({job["run_id"] for job in result}) != len(result):
        raise ValueError("Duplicate run IDs")
    return result


def inspect_bridge(kind, environment, input_hashes):
    spec = BRIDGES[kind]
    path = ROOT/spec["path"]
    if not path.is_file():
        return dict(ready=False, reasons=[f"Missing {spec['path']}"], artifact=None)
    reasons = []
    try:
        report = read(path)
        if report.get("runtime_policy") != EXPECTED_RUNTIME:
            reasons.append(f"{kind}: deterministic arithmetic policy differs or is missing")
        for field in spec["checks"]:
            if report.get(field) is not True:
                reasons.append(f"{kind}: {field} is not true")
        for field in ("docker_image", "checkpoint_sha256"):
            if report.get(field) != environment[field]:
                reasons.append(f"{kind}: {field} differs from verified environment")
        if not isinstance(report.get("execution_commit"),str) or len(report["execution_commit"]) != 40:
            reasons.append(f"{kind}: missing execution commit")
        recorded_inputs = report.get("input_sha256",{})
        for name, expected in input_hashes.items():
            if recorded_inputs.get(name) != expected:
                reasons.append(f"{kind}: input hash differs or is missing: {name}")
        source_map = report.get("source_sha256_lf",{})
        if not isinstance(source_map,dict):
            raise ValueError(f"{kind}: source_sha256_lf must be a path/hash mapping")
        for name in spec["required_sources"]:
            if name not in source_map:
                reasons.append(f"{kind}: missing required source hash: {name}")
        for name, expected in source_map.items():
            if sha(relative_file(name),text=True) != expected:
                reasons.append(f"{kind}: source changed: {name}")
    except (ValueError,KeyError,TypeError,OSError) as error:
        reasons.append(f"{kind}: invalid bridge report: {error}")
    return dict(ready=not reasons, reasons=reasons, artifact=spec["path"], sha256=sha(path))


def inspect(phase, scientific_manifest=None):
    plan = read(PLAN)
    inventory = jobs(plan, scientific_manifest)
    required_sources = list(SOURCES) + ([RUNNER] if phase=="scientific" else [])
    required_artifacts = list(INPUT_PATHS)+list(PREPARATION)+([EVALUATION] if phase=="scientific" else [])
    missing = [name for name in required_sources+required_artifacts if not (ROOT/name).is_file()]
    preparation_errors = []
    for name in PREPARATION:
        if (ROOT/name).is_file() and read(ROOT/name).get("pass_all") is not True:
            preparation_errors.append(f"Preparation is not passing: {name}")
    input_hashes = {name:sha(ROOT/name) for name in INPUT_PATHS if (ROOT/name).is_file()}
    environment = read(ROOT/PREPARATION[0]) if (ROOT/PREPARATION[0]).is_file() else {}
    bridges = {kind:inspect_bridge(kind,environment,input_hashes) for kind in BRIDGES}
    scientific_reasons = ([] if (ROOT/RUNNER).is_file() else [f"Missing {RUNNER}"])
    scientific_reasons += [reason for bridge in bridges.values() for reason in bridge["reasons"]]
    readiness = dict(phase=phase, registration_ready=not (missing or preparation_errors or
                         (phase=="scientific" and scientific_reasons)),
        missing_required_files=missing, preparation_errors=preparation_errors,
        scientific_ready=not (missing or preparation_errors or scientific_reasons),
        scientific_blockers=scientific_reasons, bridges=bridges,
        scientific_launch_performed=False, jobs=inventory)
    return readiness,plan,input_hashes,environment


def build_registry(phase,ready,plan,input_hashes,environment,destination=None,predecessor=None):
    # Recompute actual registered splits; existence or old pass_all flags alone
    # do not authorize a registration against changed immutable inputs.
    _, current, _ = build_splits()
    prepared = read(ROOT/"results/lora_profiled_v1/inputs.json")
    if current != prepared["registration"] or sha(ROOT/"results/lora_profiled_v1/splits.npz") != prepared["splits_binary_sha256"]:
        raise ValueError("Prepared inputs or split archive changed")
    sources = list(SOURCES)+([RUNNER] if phase=="scientific" else [])
    artifacts = list(INPUT_PATHS)+list(PREPARATION)
    if phase=="scientific":
        artifacts += [EVALUATION]+[spec["path"] for spec in BRIDGES.values()]
    history = None
    if predecessor is not None:
        previous = read(predecessor)
        if previous.get("phase") != phase:
            raise ValueError("Predecessor registry has a different phase")
        if destination is not None and predecessor.resolve()==destination.resolve():
            raise ValueError("A new registry must not overwrite its predecessor")
        history = dict(path=predecessor.relative_to(ROOT.resolve()).as_posix(),sha256=sha(predecessor),
            role="Historical source snapshot preserved; current-source validity is evaluated separately")
    return dict(schema=f"profiled_{phase}_v1", phase=phase,
        registry_id=None if destination is None else destination.stem,
        predecessor=history,
        purpose="Freeze engineering source and the prospective factorial" if phase=="engineering" else "Frozen scientific execution registry",
        evidence_scope="Registration only; this artifact is not a passed model experiment",
        execution_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        source_sha256_lf={name:sha(relative_file(name),text=True) for name in sources},
        artifact_sha256={name:sha(relative_file(name)) for name in artifacts},
        input_sha256=input_hashes, docker_image=environment["docker_image"],
        checkpoint_sha256=environment["checkpoint_sha256"], upstream_revisions=environment["upstream_revisions"],
        runtime_policy=EXPECTED_RUNTIME,
        study_settings=plan, states_total=36, trained_states=30, frozen_controls=6, jobs=ready["jobs"],
        endpoints=dict(primary="native KDP then common final five-scale refinement",secondary="native KDP only",
            native_coarse_iterations=300,native_fine_iterations=300,native_pose_weight=.01,
            final_graph_weight=.1,final_metric_gauge="one median ratio g0 over all permitted observations",
            final_camera_frame="native K/R, centers g0*(C_native-C0_native), depths g0*z_native",
            final_relative_scale_prior_center=1,final_matches="fresh native graph fixed during final solve"),
        execution_guards=dict(one_gpu_job_at_a_time=True,fresh_adapter_and_optimizer_per_state=True,
            overwrite_completed_or_failed_attempts=False,global_prediction_seal_before_development_scoring=True,
            original_model_weights_and_buffers_frozen=True,paid_compute_authorized=False,
            local_rental_compute_budget_usd=0,scientific_launch_requires_registry_phase="scientific"),
        bridge_status_at_registration=ready["bridges"],
        scientific_ready_at_registration=ready["scientific_ready"] if phase=="scientific" else False)


def verify_registry(path,phase):
    path = manifest_path(Path(path), ENGINEERING if phase=="engineering" else SCIENTIFIC)
    record = read(path)
    if record.get("phase") != phase or record.get("schema") != f"profiled_{phase}_v1":
        raise ValueError("Registry phase/schema differs")
    for name,expected in record["source_sha256_lf"].items():
        if sha(relative_file(name),text=True) != expected:
            raise ValueError(f"Registered source changed: {name}; preserve this historical registry and use a new version for revised source")
    for name,expected in record["artifact_sha256"].items():
        if sha(relative_file(name)) != expected:
            raise ValueError(f"Registered artifact changed: {name}")
    if phase=="scientific" and EVALUATION not in record["artifact_sha256"]:
        raise ValueError("Scientific registry omits the prospective evaluation rules")
    predecessor = record.get("predecessor")
    if predecessor is not None and sha(relative_file(predecessor["path"])) != predecessor["sha256"]:
        raise ValueError("Historical predecessor registry bytes changed")
    ready,plan,inputs,environment = inspect(phase,path if phase=="scientific" else None)
    if not ready["registration_ready"]:
        raise ValueError(json.dumps(ready,sort_keys=True))
    if record["jobs"] != ready["jobs"] or record["study_settings"] != plan:
        raise ValueError("Registered settings or inventory changed")
    expected_sources = set(SOURCES)|({RUNNER} if phase=="scientific" else set())
    if not expected_sources.issubset(record["source_sha256_lf"]):
        raise ValueError("Registry omits required source files")
    if record["input_sha256"] != inputs or record["docker_image"] != environment["docker_image"] or record["checkpoint_sha256"] != environment["checkpoint_sha256"]:
        raise ValueError("Registry input/runtime identity changed")
    if record.get("runtime_policy") != EXPECTED_RUNTIME:
        raise ValueError("Registry deterministic arithmetic policy differs")
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare",action="store_true")
    action.add_argument("--verify",action="store_true")
    action.add_argument("--dry-run",action="store_true")
    parser.add_argument("--phase",choices=("engineering","scientific"),default="engineering")
    parser.add_argument("--scientific",action="store_true",help="Alias for --phase scientific; enforce scientific readiness")
    parser.add_argument("--output",type=Path,help="New immutable registry path for --prepare; must not exist")
    parser.add_argument("--manifest",type=Path,help="Registry path to inspect with --verify")
    parser.add_argument("--supersedes",type=Path,help="Preserved same-phase predecessor to hash-link when preparing a new version")
    args = parser.parse_args()
    if args.output is not None and not args.prepare:
        parser.error("--output is only valid with --prepare")
    if args.manifest is not None and not args.verify:
        parser.error("--manifest is only valid with --verify")
    if args.supersedes is not None and not args.prepare:
        parser.error("--supersedes is only valid with --prepare")
    phase = "scientific" if args.scientific else args.phase
    default = SCIENTIFIC if phase=="scientific" else ENGINEERING
    destination = manifest_path(args.manifest if args.verify else args.output,default)
    predecessor = None if args.supersedes is None else manifest_path(args.supersedes,default)
    if args.verify:
        ready = verify_registry(destination,phase)
    else:
        ready,plan,inputs,environment = inspect(phase,destination if phase=="scientific" else None)
        if args.prepare:
            if destination.exists():
                parser.exit(2,"Preserve the existing registry. Use --verify or choose a new --output version.\n")
            if not ready["registration_ready"]:
                print(json.dumps(ready,indent=2))
                parser.exit(2,"Registration prerequisites are incomplete; no manifest written.\n")
            report = build_registry(phase,ready,plan,inputs,environment,destination,predecessor)
            with destination.open("x",encoding="utf-8",newline="\n") as stream:
                json.dump(report,stream,indent=2,allow_nan=False)
                stream.write("\n")
    print(json.dumps(ready,indent=2))
    if phase=="scientific" and not ready["scientific_ready"]:
        parser.exit(2,"Scientific execution is not ready; no job launched.\n")


if __name__ == "__main__":
    main()
