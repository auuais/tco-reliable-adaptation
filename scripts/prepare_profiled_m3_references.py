"""Bind existing M3 reference bytes and reproduce reference-camera metadata.

This preparation NEVER decodes image/depth/prediction/reference archives. It
hashes their bytes and reads only existing JSON metadata and 15 permitted pose
text files. Camera matrices reproduce the historical reference builder's
transpose/inverse convention, not the network's supplied-pose convention.
The result is not an execution registration, prediction seal or score release.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "manifests/lora_profiled_m3_references_v1.json"
SCENES = {"heads": "heads/seq-01", "chess": "chess/seq-03", "fire": "fire/seq-03"}
FRAMES = [25, 225, 425, 625, 825]
INDEX = "manifests/dense_focal_inputs.json"
POSE_INDEX = "results/source_frame_spacing/inputs.json"
AUDIT = "results/mast3r_geometry_matching_v1/match_audit.json"
CALIBRATION = "manifests/sensor_frame_v14.json"
GEOMETRY_PLAN = "manifests/mast3r_geometry_matching_v1.json"
SOURCES = (
    "scripts/prepare_profiled_m3_references.py",
    "scripts/prepare_dense_focal_pilot.py",
    "scripts/zoom_dense_reference.py",
    "scripts/calibration_audit.py",
    "scripts/audit_mast3r_crossover_matches.py",
    "scripts/audit_sensor_frame.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path, text=False):
    if text:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(root, name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "Expected repository-relative POSIX path")
    candidate = Path(name)
    require(not candidate.is_absolute() and ".." not in candidate.parts,
            "Path escapes preparation root")
    result = (root / candidate).resolve()
    require(result.is_relative_to(root.resolve()) and result.is_file(), "Missing or escaped file: " + name)
    return result


def tensor_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(json.dumps([list(array.shape), array.dtype.str]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def matrix_packet(value, count, name):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (count, 4, 4) and np.isfinite(result).all(), "Invalid camera packet: " + name)
    require(np.array_equal(result[:, 3], np.tile([0., 0., 0., 1.], (count, 1))),
            "Invalid camera homogeneous row: " + name)
    require((np.linalg.det(result[:, :3, :3]) > 0).all(), "Camera must be invertible and right-handed: " + name)
    return result


def camera_functions(root):
    """Extract only the two audited functions, never import legacy experiment IO."""
    namespace = {"np": np}
    for filename, function in (("scripts/audit_sensor_frame.py", "effective_camera"),
                               ("scripts/audit_mast3r_crossover_matches.py", "reference_poses")):
        source = relative(root, filename)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
        require(len(nodes) == 1, "Expected one audited camera function: " + function)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["reference_poses"]


def build(root, assets):
    root, assets = Path(root).resolve(), Path(assets).resolve()
    require(assets.is_dir(), "Public pose asset root is missing")
    index, poses, audit, calibration, geometry = (
        read(relative(root, path)) for path in (INDEX, POSE_INDEX, AUDIT, CALIBRATION, GEOMETRY_PLAN))
    require(geometry["scenes"] == SCENES and geometry["frames"] == FRAMES, "Scene/frame contract changed")
    matrix = matrix_packet([calibration["calibration"]], 1, "depth-to-RGB calibration")[0]
    source_hashes = {name: sha(relative(root, name), text=True) for name in SOURCES}
    require(audit["source_sha256"] == source_hashes["scripts/audit_mast3r_crossover_matches.py"],
            "Historical camera-audit source differs from the current pinned function")
    reference_poses = camera_functions(root)
    scenes = {}
    pose_files = 0
    for scene, sequence in SCENES.items():
        key = scene + "_base"
        rgb, reference = index["inputs"][key], index["references"][key]
        require(rgb["scene"] == sequence and rgb["batch"] == "base", "Wrong base RGB identity")
        require(type(reference["valid_pixels"]) is int and 0 < reference["valid_pixels"] <= 5*392*518,
                "Invalid declared fixed-mask count")
        artifacts = {}
        for role, parent, record in (("rgb", "reports/dense_focal_pilot/inference", rgb),
                                      ("reference", "reports/dense_focal_pilot/evaluation", reference)):
            require(Path(record["file"]).name == record["file"], "Expected a simple archive filename")
            name = parent + "/" + record["file"]
            path = relative(root, name)
            actual = sha(path)
            require(actual == record["sha256"], "Historical " + role + " archive changed: " + scene)
            artifacts[role] = dict(file=name, sha256=actual, bytes=path.stat().st_size, decoded=False)
        prior_audit = audit["reference_cameras"][scene]
        expected = matrix_packet(prior_audit["world_to_camera"], 5, "historical reference cameras")
        hashes, original = {}, []
        for frame in FRAMES:
            name = sequence + f"/frame-{frame:06d}.pose.txt"
            path = (assets / name).resolve()
            require(path.is_relative_to(assets) and path.is_file(), "Missing or escaped original pose")
            digest = sha(path)
            require(digest == poses["frames"][sequence][str(frame)]["original"]["pose.txt"]
                    == prior_audit["original_pose_sha256"][str(frame)], "Original pose changed")
            original.append(np.loadtxt(path, dtype=np.float64))
            hashes[name] = digest
            pose_files += 1
        original = matrix_packet(original, 5, "original source poses")
        reproduced = matrix_packet(reference_poses(original, matrix), 5, "reproduced reference cameras")
        delta = float(np.max(np.abs(reproduced - expected)))
        require(np.allclose(reproduced, expected, rtol=1e-12, atol=1e-12),
                "Reference camera convention no longer reproduces the archived audit")
        require(audit["same_view_checks"][scene]["reference_sha256"] == artifacts["reference"]["sha256"],
                "Historical camera check refers to another dense reference")
        scenes[scene] = dict(sequence=sequence, frames=FRAMES, **artifacts,
            reference_fields=["registered_rgb", "released", "valid_mask"],
            fixed_mask=dict(shape=[5,392,518], dtype="bool", declared_count=reference["valid_pixels"],
                            shared_between_references=True, tensor_sha256=None,
                            semantic_validation="pending_authorized_decode_after_stage_seal"),
            primary_axial_camera=dict(reference="registered_rgb", world_to_camera=expected.tolist(),
                tensor_sha256=tensor_sha(expected), dtype=expected.dtype.str, shape=list(expected.shape),
                source_pose_sha256=hashes, historical_agreement_rtol=1e-12, historical_agreement_atol=1e-12,
                historical_camera_reproduction_passed=True, canonical_values="archived_camera_metadata",
                matrices_orthogonalized=False,
                convention="calibration @ effective_camera(loader_transpose_inverse(original_c2w))",
                current_reference_projection_check="not_run_no_reference_decode"),
            alternate_axial_camera=dict(reference="released", status="unsupported_not_registered"))
    return dict(schema="profiled_m3_reference_preparation_v1", scenes=scenes,
        metadata_sha256={name: sha(relative(root,name)) for name in (INDEX,POSE_INDEX,AUDIT,CALIBRATION,GEOMETRY_PLAN)},
        source_sha256_lf=source_hashes, original_pose_text_files_read=pose_files,
        image_depth_prediction_reference_archives_decoded=0,
        accuracy_scoring_performed=False, execution_registration=False, release_authorized=False,
        limitation="Byte identity and reference-camera algebra only. Mask semantics, current reprojection and dense metric evaluation remain gated. References are two conventions of shared data, not independent datasets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","verify"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    require(destination.is_relative_to(ROOT) and destination.suffix == ".json", "Output must remain in repository")
    if args.action == "prepare":
        require(not destination.exists(), "Preserve previous preparation; choose a new version")
    else:
        require(destination.is_file(), "Preparation to verify is missing")
    record = build(ROOT,args.assets)
    if args.action == "prepare":
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(record,stream,indent=2,allow_nan=False)
            stream.write("\n")
    else:
        require(read(destination) == record,"Reference preparation changed")
    print(json.dumps(dict(pass_all=True,action=args.action,scenes=len(record["scenes"]),
        pose_files=record["original_pose_text_files_read"],archives_decoded=0,release_authorized=False)))


if __name__ == "__main__":
    main()
