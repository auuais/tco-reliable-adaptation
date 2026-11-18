"""Static M3 execution-registry bytes and inventory checks; never release a job.

``build_execution`` constructs an in-memory descriptor, without writing it.
``verify_execution(root, registry_file, weights=None)`` reads JSON provenance and
hashes opaque files. It never decodes NPZ/checkpoints, imports a model, reads a
score payload or establishes a nominee. Nomination proof files are byte-bound
external prerequisites: a future release verifier must replay the existing M2
scorer/sealer and verify the actual engineering endpoint acceptance. Passing
this module is explicitly insufficient for execution, reuse or label access.

Top-level fields are schema, inventory, jobs, model_core, prerequisites and
launch_authorized=False. ``inventory`` is exactly build_plan(nomination,inputs).
model_core is the existing ModelContext envelope. Its 24 cases each contain
input_sha256 and case_input_digest. Inventory case identities mean those tensor
digests; runtime identity is canonical_sha256(model_core.runtime_identity), and
endpoint identity is canonical_sha256(the pinned M2 registry's endpoints).

Prerequisites contains a case_bridge {file,sha256}, an engineering_acceptance
descriptor or None, and nomination_proof mapping all four NOMINATION_FIELDS to
{file,sha256}. A present acceptance/proof file is not automatically accepted.
This avoids the registration/acceptance circularity and fabricated pass flags.
"""
from copy import deepcopy
import itertools
import json
from pathlib import Path

from profiled_artifacts import resolve_logical
from profiled_m3_inventory import (build_plan, case_id, SCENES, SIDES, GRIDS, POSES,
                                   INPUT_PLAN, INPUT_ATTEMPT, NOMINATION_FIELDS)
from profiled_m3_gate import canonical_sha256, digest, require
from profiled_runtime import EXPECTED_RUNTIME
from run_profiled_lora import sha


SELF = "scripts/profiled_m3_registry.py"
M2 = "manifests/lora_profiled_execution_v2.json"
ENVIRONMENT = "results/lora_profiled_v1/environment.json"
BRIDGE = "results/mast3r_neural_transfer/zero_adapter_bridge.json"
BASELINE = "results/lora_profiled_v1/baseline_graph.json"
RGB = "manifests/dense_focal_inputs.json"
PREPARATION = {key: f"results/lora_profiled_m3_v1/input_attempts/{INPUT_ATTEMPT}/{key}/complete.json"
               for key in ("earlier", "later", "poses")}
IMAGE = "sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50"
EXTRA_SOURCES = (SELF, "scripts/profiled_m3_model.py", "scripts/profiled_m3_network.py",
                 "scripts/profiled_m3_inventory.py", "scripts/profiled_m3_gate.py",
                 "scripts/prepare_profiled_m3_inputs.py", "scripts/check_profiled_m3_network.py",
                 "scripts/run_profiled_lora.py", "scripts/profiled_artifacts.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_at(root, name):
    return resolve_logical(root, name, must_exist=True)


def check_map(root, mapping, *, text=False):
    require(isinstance(mapping, dict) and mapping, "Missing registered file map")
    for name, expected in mapping.items():
        digest(expected, name)
        require(sha(file_at(root, name), text=text) == expected, "Registered bytes changed: " + name)


def descriptor(root, item):
    require(isinstance(item, dict) and set(item) == {"file", "sha256"}, "Expected file/sha256 descriptor")
    check_map(root, {item["file"]: item["sha256"]})
    return item["file"]


def build_execution(nomination, inputs, model_core, prerequisites):
    """Pure construction only; caller must provide real external provenance."""
    inventory = build_plan(nomination, inputs)
    return dict(schema="profiled_m3_execution_v1", inventory=inventory,
                jobs=deepcopy(inventory["jobs"]), model_core=deepcopy(model_core),
                prerequisites=deepcopy(prerequisites), launch_authorized=False)


def _cases(root, core, inventory, prerequisites, input_plan):
    case_name = descriptor(root, prerequisites["case_bridge"])
    require(core["artifact_sha256"].get(case_name) == prerequisites["case_bridge"]["sha256"],
            "CPU case bridge is not bound in model_core")
    bridge = read(file_at(root, case_name))
    require(bridge.get("schema") == "profiled_m3_cpu_case_bridge_v1" and bridge.get("pass_all") is True
            and bridge.get("cases_checked") == 24 and bridge.get("centered_clean_m2_cases_exact") == 6
            and bridge.get("endpoint_body_ast_exact") is True and bridge.get("device") == "cpu"
            and bridge.get("gpu_used") is False and bridge.get("prediction_arrays_opened") is False
            and bridge.get("accuracy_scoring_performed") is False,
            "CPU case evidence is incomplete or not the permitted input-only bridge")
    check_map(root, bridge["source_sha256_lf"], text=True)
    require(all(core["source_sha256_lf"].get(k) == v for k, v in bridge["source_sha256_lf"].items()),
            "CPU case bridge source is not in the execution source bundle")
    rows = bridge.get("cases")
    require(isinstance(rows, list) and len(rows) == 24, "Exactly24 CPU case receipts are required")
    expected = {case_id(*parts) for parts in itertools.product(SCENES, SIDES, GRIDS, POSES)}
    require(set(core["cases"]) == expected, "Model core must contain exactly24 cases")
    found = set()
    for row in rows:
        parts = tuple(row.get(key) for key in ("scene", "side", "grid", "pose_condition"))
        require(all(isinstance(value, str) for value in parts), "Malformed CPU case identity")
        key = case_id(*parts)
        require(key in expected and key not in found, "Duplicate or unexpected CPU case receipt")
        found.add(key)
        scene, side, grid, pose = parts
        require(row.get("original_m2_loader_compared") is (grid == "centered" and pose == "clean"),
                "Wrong centered/clean input-comparison scope")
        counts = row.get("counts")
        require(isinstance(counts, list) and len(counts) == 5 and all(type(n) is int and n >= 16 for n in counts),
                "Invalid five-view CPU fitting counts")
        digest(row.get("input_digest"), key + " tensor digest")
        case = core["cases"][key]
        require(case == dict(input_sha256=row["input_sha256"], case_input_digest=row["input_digest"]),
                "Model case differs from verified CPU fitting receipt")
        require(inventory["inputs"]["case_input_sha256"][key] == case["case_input_digest"],
                "Inventory case identity differs from actual fitting tensor digest")
        base = f"results/lora_profiled_m3_v1/input_attempts/{INPUT_ATTEMPT}"
        required = {INPUT_PLAN, base + "/poses/complete.json", base + "/poses/pose_packets.npz"}
        if grid == "centered":
            required.update((f"results/mast3r_partner_shape/{side}.npz", "results/lora_profiled_v1/splits.npz",
                             "results/lora_profiled_v1/inputs.json"))
        else:
            required.update((base + f"/{side}/complete.json", base + f"/{side}/shifted_packet.npz",
                             base + f"/{side}/shifted_splits.npz"))
        require(set(case["input_sha256"]) == required, "Wrong selected-case input file inventory")
        check_map(root, case["input_sha256"])
        require(case["input_sha256"][INPUT_PLAN] == inventory["inputs"]["plan_sha256"], "Case uses another input plan")
    require(found == expected, "Missing CPU case receipt")


def verify_execution(root, registry_file, *, weights=None):
    """Verify static bytes; always return launch_authorized=False.

    Score-report/nomination-verification/endpoint-acceptance payloads remain
    unopened. Their byte hashes alone cannot discharge release prerequisites.
    Optional weights verifies current safetensors AND config bytes, not history.
    """
    root = Path(root).resolve()
    registry_path = file_at(root, registry_file)
    registry_hash = sha(registry_path)
    record = read(registry_path)
    require(set(record) == {"schema", "inventory", "jobs", "model_core", "prerequisites", "launch_authorized"}
            and record["schema"] == "profiled_m3_execution_v1" and record["launch_authorized"] is False,
            "Expected a non-authorizing static execution descriptor")
    inventory = record["inventory"]
    require(inventory == build_plan(inventory["nomination"], inventory["inputs"]), "Inventory differs from fixed360 plan")
    require(record["jobs"] == inventory["jobs"], "Jobs differ from exact fixed360 inventory/order")
    core, prerequisites = record["model_core"], record["prerequisites"]
    require(set(core) == {"source_sha256_lf", "artifact_sha256", "input_plan", "input_attempt", "cases",
                         "docker_image", "checkpoint_sha256", "checkpoint_config_sha256", "original_state_sha256",
                         "runtime_identity"}, "Wrong ModelContext envelope")
    require(core["input_plan"] == INPUT_PLAN and core["input_attempt"] == INPUT_ATTEMPT,
            "Only preserved input plan v2/inputs02 is supported")
    require(set(prerequisites) == {"nomination_proof", "case_bridge", "engineering_acceptance"},
            "Missing separate scientific release prerequisites")
    from register_profiled_execution import SOURCES
    require((set(SOURCES) | set(EXTRA_SOURCES)).issubset(core["source_sha256_lf"]), "Required execution sources are unbound")
    check_map(root, core["source_sha256_lf"], text=True)
    require(core["source_sha256_lf"][SELF] == sha(Path(__file__), text=True), "Executing registry verifier differs")
    check_map(root, core["artifact_sha256"])
    required = {M2, ENVIRONMENT, BRIDGE, BASELINE, RGB, INPUT_PLAN, *PREPARATION.values()}
    require(required.issubset(core["artifact_sha256"]), "Required provenance artifacts are unbound")
    m2, environment, zero, rgb, plan = (read(file_at(root, name)) for name in (M2, ENVIRONMENT, BRIDGE, RGB, INPUT_PLAN))
    require(m2.get("schema") == "profiled_scientific_v1" and m2.get("phase") == "scientific"
            and m2.get("scientific_ready_at_registration") is True, "Parent M2 registration is not scientific")
    require(m2["artifact_sha256"].get(ENVIRONMENT) == core["artifact_sha256"][ENVIRONMENT], "M2 environment identity differs")
    require(m2["artifact_sha256"].get(BASELINE) == core["artifact_sha256"][BASELINE], "M2 baseline graph identity differs")
    require(all(core["source_sha256_lf"].get(name) == value for name, value in m2["source_sha256_lf"].items()),
            "A frozen M2 source is changed or unbound")
    require(core["docker_image"] == m2["docker_image"] == environment["docker_image"] == IMAGE,
            "Pinned local model image differs")
    require(core["checkpoint_sha256"] == m2["checkpoint_sha256"] == environment["checkpoint_sha256"]
            == zero["checkpoint_sha256"], "Pinned model checkpoint differs")
    for name in ("checkpoint_sha256", "checkpoint_config_sha256", "original_state_sha256"):
        digest(core[name], name)
    require(core["original_state_sha256"] == zero["original_state_hash"] == zero["final_state_hash"]
            and zero.get("pass_all") is True, "Original frozen-model bridge differs")
    runtime = dict(torch=environment["torch"], cuda=environment["cuda"], gpu=environment["gpu"], policy=EXPECTED_RUNTIME)
    require(core["runtime_identity"] == runtime and m2["runtime_policy"] == EXPECTED_RUNTIME,
            "Model arithmetic/runtime identity differs")
    inputs = inventory["inputs"]
    require(inputs["runtime_sha256"] == canonical_sha256(runtime)
            and inputs["checkpoint_sha256"] == core["checkpoint_sha256"]
            and inputs["endpoint_contract_sha256"] == canonical_sha256(m2["endpoints"]),
            "Inventory model/runtime/endpoint identity differs")
    require(inputs["plan_sha256"] == core["artifact_sha256"][INPUT_PLAN], "Wrong prepared plan identity")
    check_map(root, plan["metadata_sha256"])
    check_map(root, plan["source_sha256_lf"], text=True)
    require(all(core["source_sha256_lf"].get(k) == v for k, v in plan["source_sha256_lf"].items()),
            "Preparation source is outside the bound execution bundle")
    for scene in SCENES:
        image = rgb["inputs"][scene + "_base"]
        expected_scene = {"heads": "heads/seq-01", "chess": "chess/seq-03", "fire": "fire/seq-03"}[scene]
        require(image.get("scene") == expected_scene and image.get("batch") == "base", "Wrong RGB scene/window")
        require(isinstance(image.get("file"), str) and Path(image["file"]).name == image["file"], "Invalid RGB basename")
        name = "reports/dense_focal_pilot/inference/" + image["file"]
        require(core["artifact_sha256"].get(name) == image["sha256"], "Selected RGB bytes are unbound")
    for role, name in PREPARATION.items():
        receipt = read(file_at(root, name))
        require(core["artifact_sha256"][name] == inputs["preparation_receipt_sha256"][role], "Preparation receipt differs")
        require(receipt.get("pass_all") is True and receipt["plan_sha256"] == inputs["plan_sha256"]
                and receipt["source_sha256_lf"] == plan["source_sha256_lf"], "Preparation source/plan identity differs")
        if role != "poses":
            require(receipt.get("side") == role, "Wrong signed-source preparation")
        else:
            for scene, pose in itertools.product(SCENES, POSES):
                repeats = receipt["repeat_matrix_sha256"][scene][pose]
                require(set(repeats) == {"0", "1", "2"} and
                        set(repeats.values()) == {inputs["pose_matrix_sha256"][scene + "/" + pose]},
                        "Pose draw differs across repetitions or inventory")
    _cases(root, core, inventory, prerequisites, plan)
    proof = prerequisites["nomination_proof"]
    require(isinstance(proof, dict) and set(proof) == set(NOMINATION_FIELDS), "Incomplete external nomination linkage")
    for key, item in proof.items():
        descriptor(root, item)  # Do not parse score/proof payloads here.
        require(item["sha256"] == inventory["nomination"][key], "Nomination descriptor identity differs")
    acceptance = prerequisites["engineering_acceptance"]
    if acceptance is not None:
        descriptor(root, acceptance)  # Presence is not proof of real-model acceptance.
    if weights is not None:
        weights = Path(weights)
        require(sha(weights / "model.safetensors") == core["checkpoint_sha256"] and
                sha(weights / "config.json") == core["checkpoint_config_sha256"], "Live model weights/configuration differ")
    require(sha(registry_path) == registry_hash, "Registry changed during static verification")
    return dict(schema="profiled_m3_execution_byte_check_v1", registry=record, registry_sha256=registry_hash,
        pass_all=True, source_bytes_verified=True, artifact_bytes_verified=True, weights_verified=weights is not None,
        cases_checked=24, states_checked=360, arrays_decoded=False, score_payloads_read=False,
        nomination_verified=False, engineering_acceptance_verified=False, release_prerequisites_verified=False,
        launch_authorized=False, reuse_authorized=False, label_access_authorized=False,
        pending_release_checks=["Replay existing M2 seal/scorer and authenticate the exact nomination proof",
            "Verify real centered/clean endpoint acceptance against registered sources/inputs/config/runtime",
            "Verify current stage/attempt ownership and acquire the shared GPU lock",
            "Audit historical configuration provenance separately before any M2 state reuse"],
        scope="Static source/artifact/input byte consistency only; not scientific execution authorization")
