"""Freeze and verify the execution and scoring inputs before scientific GPU work."""
import argparse
from pathlib import Path
import subprocess

from native_lora_direction_inputs import ENVIRONMENTS, RAW, ROOT, SIDES, STATES
from prepare_profiled_inputs import read, sha
from run_profiled_d0 import IMAGE, write_new


MANIFEST = ROOT / "manifests/native_lora_direction_execution_v1.json"
RESULTS = ROOT / "results/native_lora_direction_v1"
REFERENCE = ROOT / "results/lora_profiled_v1/environment.json"
SOURCE_PATHS = (
    "scripts/download_stanford12_direction.py",
    "scripts/native_lora_direction_inputs.py",
    "scripts/native_lora_direction_registration.py",
    "scripts/run_native_lora_direction.py",
    "scripts/run_native_lora_direction_local.py",
    "scripts/score_native_lora_direction.py",
    "scripts/replay_native_lora_direction.py",
    "scripts/test_native_lora_direction.py",
    "docs/NATIVE_LORA_DIRECTIONAL_TEST_HANDOFF.md",
    "scripts/profiled_network.py", "scripts/profiled_runtime.py",
    "scripts/profiled_calibration.py", "scripts/mast3r_alignment_hook.py",
    "scripts/mast3r_calibrated_alignment.py", "scripts/mast3r_joint_prior_loss.py",
    "scripts/mast3r_network_capture.py", "scripts/mast3r_network_priors.py",
    "scripts/check_profiled_endpoints.py", "scripts/prepare_profiled_inputs.py",
    "scripts/run_profiled_d0.py", "scripts/run_profiled_d0_local.py",
    "scripts/run_profiled_d1_local.py", "scripts/run_profiled_lora.py",
    "third_party/TCO/peft/lora.py", "third_party/mast3r/mast3r/cloud_opt/sparse_ga.py",
    "third_party/mast3r/dust3r/dust3r/image_pairs.py",
)
BRIDGE = RESULTS / "engineering_bridge.json"


def git(args):
    return subprocess.check_output(["git", "-c", "safe.directory=*", "-C", str(ROOT), *args]).strip()


def inventory():
    return [(room, side) for room in ENVIRONMENTS for side in SIDES]


def prepare(intake, bridge, weights):
    ref = read(REFERENCE)
    if sha(weights / "model.safetensors") != ref["checkpoint_sha256"]:
        raise ValueError("Local model weight bytes differ from historical checkpoint")
    source = RAW / intake / "inputs.json"
    inputs = read(source)
    exact = read(RAW / bridge / "bridge.json")
    if not exact["exact"]:
        raise ValueError("Engineering bridge has not accepted the new endpoint")
    write_new(BRIDGE, exact)
    write_new(MANIFEST, dict(schema="native_lora_direction_execution_v1", intake=intake,
              intake_sha256=sha(source), inputs=inputs, inventory=inventory(), states=STATES,
              reserved_environment="office1", image=IMAGE, rental_usd_limit=0,
              checkpoint_sha256=ref["checkpoint_sha256"], checkpoint_config_sha256=sha(weights / "config.json"),
              runtime_sha256_lf=sha(REFERENCE, text=True), bridge_sha256_lf=sha(BRIDGE, text=True),
              source_sha256_lf={p: sha(ROOT / p, text=True) for p in SOURCE_PATHS},
              scoring=dict(primary="g0 times native axial depth, equal views then equal rooms, mm",
                           floor_mm=.05, repeat_multiplier=2., min_valid_per_view=8, agreeing_rooms=2,
                           g0="per-state pooled adaptation-only midpoint median ratio",
                           invalid_prediction="invalidate case; no sample exclusion",
                           split="exclude reused source-ray identities and adaptation pixels"),
              commands=dict(run="python scripts/run_native_lora_direction_local.py",
                            seal="python scripts/score_native_lora_direction.py seal",
                            score="python scripts/score_native_lora_direction.py score",
                            replay="python scripts/replay_native_lora_direction.py")))
    verify_registration()


def verify_registration(*, require_committed=False):
    record = read(MANIFEST)
    if (record["schema"] != "native_lora_direction_execution_v1" or
            record["states"] != STATES or record["inventory"] != [list(v) for v in inventory()] or
            set(record["source_sha256_lf"]) != set(SOURCE_PATHS) or record["image"] != IMAGE or
            record["rental_usd_limit"] != 0 or record["reserved_environment"] != "office1"):
        raise ValueError("Registration cohort, state inventory or runtime differs")
    if set(record["inputs"]["environments"]) != set(ENVIRONMENTS):
        raise ValueError("Unexpected room inventory")
    for environment, entry in record["inputs"]["environments"].items():
        if entry["room"] != sorted(entry["rooms"], key=str.casefold)[0] or (
                not entry["room"].startswith(environment + "/")):
            raise ValueError("Room differs from deterministic selection")
        frames = entry["target_frames"]
        if (len(frames) != 5 or len(set(frames)) != 5 or
                any(entry["packets"][side]["source_frames"] != [n + delta for n in frames]
                    for side, delta in SIDES.items())):
            raise ValueError("Temporal provenance differs from registration")
    if (sha(REFERENCE, text=True) != record["runtime_sha256_lf"] or
            sha(BRIDGE, text=True) != record["bridge_sha256_lf"]):
        raise ValueError("Original runtime or engineering receipt changed")
    source = RAW / record["intake"] / "inputs.json"
    if sha(source) != record["intake_sha256"] or read(source) != record["inputs"]:
        raise ValueError("Intake plan or hash changed")
    for rel, expected in record["source_sha256_lf"].items():
        if sha(ROOT / rel, text=True) != expected:
            raise ValueError(f"Source changed after registration: {rel}")
    if require_committed:
        committed = git(["show", "HEAD:" + MANIFEST.relative_to(ROOT).as_posix()])
        if committed.replace(b"\r\n", b"\n") != MANIFEST.read_bytes().strip().replace(b"\r\n", b"\n"):
            raise ValueError("Commit the exact registration before launch")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", required=True, choices=("intake01", "intake02"))
    parser.add_argument("--bridge", required=True, choices=("bridge01", "bridge02", "bridge03"))
    parser.add_argument("--weights", type=Path, default=Path("E:/ResearchData/TCO/checkpoints/MASt3R/hf"))
    opts = parser.parse_args()
    prepare(opts.intake, opts.bridge, opts.weights)
