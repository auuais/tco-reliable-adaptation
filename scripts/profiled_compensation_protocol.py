"""Prospective release identities for the single MASt3R redesign."""
from pathlib import Path

from run_profiled_d0 import ROOT, committed_identity, read, sha
from run_profiled_d1 import INVENTORY, REGISTRY as D1_EXECUTION
from profiled_runtime import EXPECTED_RUNTIME

DESIGN = "manifests/lora_profiled_compensation_design_v1.json"
ENGINEERING = "manifests/lora_profiled_compensation_engineering_v1.json"
EXECUTION = "manifests/lora_profiled_compensation_execution_v1.json"
EVALUATION = "manifests/lora_profiled_compensation_evaluation_v1.json"
PARENT = "manifests/lora_profiled_execution_v2.json"
THRESHOLDS = "manifests/lora_profiled_bridge_checks_v1.json"
PUBLIC = "results/lora_profiled_compensation_v1"
RAW = "reports/lora_profiled_compensation_v1"
SCENES, SIDES = ("heads", "chess", "fire"), ("earlier", "later")
ARMS = ("P0", "P1", "P3", "R0", "R1", "R2")
ENGINEERING_SOURCES = ("scripts/profiled_compensation_protocol.py",
                       "scripts/profiled_compensation_training.py",
                       "scripts/check_profiled_compensation_bridge.py",
                       "scripts/run_profiled_compensation_local.py")
SCIENTIFIC_SOURCES = ("scripts/run_profiled_compensation.py",
                      "scripts/run_profiled_compensation_batch.py",
                      "scripts/score_profiled_compensation.py",
                      "scripts/test_profiled_compensation_protocol.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(name):
    return dict(file=name, sha256=sha(ROOT / name))


def verify_identity(desc):
    require(file_identity(desc["file"]) == desc, "Bound file changed: " + desc["file"])


def validate_design():
    design = read(ROOT / DESIGN)
    for item in (*design["motivation"], design["specification"], design["method_source"],
                 design["fixture_source"], design["source_schedule"]):
        verify_identity(item)
    require(tuple(row["arm"] for row in design["arms"]) == ARMS
            and tuple(design["scenes"]) == SCENES and tuple(design["adaptation_sources"]) == SIDES
            and design["fresh_states"] == 36 and design["fresh_trained_states"] == 30,
            "Unexpected redesign inventory")
    require(design["prospective_maximum_redesigns"] == 1 and design["rental_compute_usd"] == 0,
            "Redesign budget differs")
    return design


def source_maps(names):
    d1 = read(ROOT / D1_EXECUTION)
    parent = read(ROOT / PARENT)
    baseline = dict(d1["source_sha256_lf"])
    upstream = dict(d1["upstream_sha256_lf"])
    for name, digest in parent["source_sha256_lf"].items():
        target = upstream if name.startswith("third_party/") else baseline
        require(name not in target or target[name] == digest, "Conflicting baseline source hashes")
        target[name] = digest
    verify_hashes(baseline, text=True)
    verify_hashes(upstream, text=True)
    for name in names:
        baseline[name] = sha(ROOT / name, text=True)
    return dict(sorted(baseline.items())), dict(sorted(upstream.items()))


def verify_hashes(hashes, *, text=False):
    for name, digest in hashes.items():
        require(sha(ROOT / name, text=text) == digest, "Hash changed: " + name)


def engineering_payload(weights):
    validate_design()
    sources, upstream = source_maps(ENGINEERING_SOURCES)
    d1 = read(ROOT / D1_EXECUTION)
    inventory = read(ROOT / INVENTORY)
    for name in ("checkpoint", "config"):
        if weights is not None:
            model_file = weights / inventory[name]["file"]
            require(sha(model_file) == inventory[name]["sha256"], "Local weight/config bytes differ")
    return dict(schema="profiled_compensation_engineering_v1", design=file_identity(DESIGN),
                source_sha256_lf=sources, upstream_sha256_lf=upstream,
                d1_execution=file_identity(D1_EXECUTION), inventory=file_identity(INVENTORY),
                thresholds=file_identity(THRESHOLDS), runtime_policy=EXPECTED_RUNTIME,
                environment=d1["runtime_evidence"], docker_image=d1["image"],
                checkpoint_sha256=inventory["checkpoint"]["sha256"],
                config_sha256=inventory["config"]["sha256"], rental_compute_usd=0,
                scene="heads", adaptation_source="earlier", varying_pairs=[[0,1],[0,2]],
                saved_nonzero_engineering_state=inventory["roles"]["adapter"]["file"],
                archived_d1_teacher_required=True, optimizer_updates=0,
                limits=dict(total_decoder_calls=192, wall_seconds=1800, host_memory_gib=8,
                            cuda_allocated_mib=8192, cuda_reserved_mib=9216))


def verify_engineering(weights):
    manifest = read(ROOT / ENGINEERING)
    head = committed_identity(ROOT, ENGINEERING, manifest["source_sha256_lf"])
    require(engineering_payload(weights) == manifest, "Engineering executable registration differs")
    return manifest, head


def jobs():
    parent = {row["run_id"]: row for row in read(ROOT / PARENT)["jobs"]}
    result = []
    for scene in SCENES:
        for side in SIDES:
            for arm in ARMS:
                key = f"{scene}_{side}_{arm}"
                template = parent[f"{scene}_{side}_{arm if arm.startswith('P') else 'P3'}"]
                argv = ["python", "scripts/run_profiled_compensation.py", "--run-id", key,
                        "--manifest", EXECUTION, "--attempt", "FRESH_ATTEMPT_ID", "--weights", "/weights"]
                result.append(dict(template, run_id=key, arm=arm,
                    differentiate_solve=False if arm == "R2" else template["differentiate_solve"],
                    raw_attempt_parent=f"{RAW}/cases/{key}", public_attempt_parent=f"{PUBLIC}/cases/{key}",
                    planned_argv=argv))
    return result
