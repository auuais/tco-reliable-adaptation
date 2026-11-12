"""Validate portable crossover evidence, geometry, factorial contrasts and match summaries."""
import argparse
import itertools
import numpy as np

from audit_mast3r_crossover_matches import describe, aggregate, reference_poses
from plan_mast3r_crossover import ROOT, PARENT, MANIFEST, read, require, expected_plan, sha256, text_sha256, validate_public
from replay_mast3r_transfer import independently_resize
from neighbor_sparse_depth import tensor_digest
from score_mast3r_crossover import PUBLIC, summarize


def replay(local=False):
    plan = read(MANIFEST)
    require(plan == expected_plan(), "Frozen protocol changed")
    validate_public(plan)
    seal = read(PUBLIC / "predictions.json")
    require(seal["manifest_sha256"] == sha256(MANIFEST), "Manifest hash changed")
    require(seal["session_sha256"] == sha256(PUBLIC / "session.json"), "Session changed")
    require(seal["scoring_source_sha256"] == text_sha256(ROOT / "scripts/score_mast3r_crossover.py"), "Scoring code changed")
    for name, digest in seal["evaluator_sources"].items():
        require(text_sha256(ROOT / name) == digest, "Evaluator source changed")
    require([r["id"] for r in seal["rows"]] == [r["id"] for r in plan["runs"]], "Run list differs")
    session = read(PUBLIC / "session.json")
    require(session["matmul_allow_tf32"], "Parent TF32 setting missing")
    for name, digest in session["sources"].items():
        require(text_sha256(ROOT / name) == digest, "Runtime source changed")
    operands = {}
    with np.load(PUBLIC / "replay_points.npz") as samples:
        for row in seal["rows"]:
            tag = row["id"]
            folder = PUBLIC / "cases" / tag
            completion = read(folder / "complete.json")
            require(sha256(folder / "complete.json") == row["completion_sha256"], "Completion changed")
            require(completion["sources"] == session["sources"], "Source versions mixed")
            require(completion["job"] == next(r for r in plan["runs"] if r["id"] == tag), "Changed job")
            for name, digest in completion["files"].items():
                path = PUBLIC / "native_outputs" / (tag + ".npz") if name.startswith("reports/") else ROOT / name
                require(sha256(path) == digest, f"Changed completion file {name}")
            metadata = read(folder / "native.json")
            require(metadata["input_sha256"] == read(PARENT / "plan.json")["rgb_sha256"][row["scene"]], "RGB changed")
            aligned = read(folder / "alignment.json")
            pairs = aligned["consumed_pairs"]
            require(len(pairs) == len({tuple(p) for p in pairs}) == 20, "Missing consumed pair")
            names = {p for pair in pairs for p in pair}
            require(len(names) == 5 and {tuple(p) for p in pairs} == set(itertools.permutations(names, 2)), "Wrong directed view pairs")
            trace = read(folder / "trace.json")
            require([r["iteration"] for r in trace] == list(range(300)), "Incomplete trace")
            require(all(r["finite_gradient"] and r["received_gradient"] == 1. for r in trace), "Invalid callback gradients")
            for event in trace:
                weighted = .01 * (event["depth"] + event["rotation"] + event["translation"] + event["intrinsic"])
                np.testing.assert_allclose(weighted, event["total_weighted"], rtol=3e-6, atol=3e-7)
            path = PUBLIC / "native_outputs" / (tag + ".npz")
            require(sha256(path) == row["native_sha256"] == metadata["output_sha256"], "Native output changed")
            with np.load(path) as native:
                np.testing.assert_array_equal((native["depth"] <= 0).sum((1, 2)), metadata["nonpositive_depth_pixels"])
                if "endpoint" in row:
                    require(read(folder / "bridge.json")["pass"], "Archived bridge failed")
                    with np.load(PARENT / "native_outputs" / (row["endpoint"]["tag"] + ".npz")) as old:
                        for field in ("depth", "intrinsics"):
                            np.testing.assert_allclose(native[field], old[field], **plan["bridge_tolerance"])
                        a, b = native["c2w"].astype(float), old["c2w"].astype(float)
                        np.testing.assert_allclose(np.linalg.solve(a[0], a), np.linalg.solve(b[0], b), **plan["bridge_tolerance"])
                depth, intrinsic = independently_resize(native["depth"], native["intrinsics"])
                y, x = np.indices(depth.shape[1:], dtype=float)
                pixels = np.stack([x, y, np.ones_like(x)], axis=-1)[::9, ::9]
                c2w = native["c2w"].astype(float)
                points = np.stack([(pixels @ np.linalg.inv(intrinsic[i]).T * depth[i, ::9, ::9, None])
                                   @ c2w[i, :3, :3].T + c2w[i, :3, 3] for i in range(5)])
                np.testing.assert_allclose(points, samples[tag], rtol=1e-5, atol=2e-5)
                if local:
                    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
                    full = np.stack([(pixels @ np.linalg.inv(intrinsic[i]).T * depth[i, :, :, None])
                                     @ c2w[i, :3, :3].T + c2w[i, :3, 3] for i in range(5)])
                    with np.load(ROOT / "reports/mast3r_geometry_matching_v1/points" / (row["prediction"] + ".npz")) as saved:
                        require(tensor_digest(saved["points"]) == row["prediction"], "Full point map changed")
                        np.testing.assert_allclose(full, saved["points"], rtol=1e-5, atol=2e-5)
            operands[tag] = {r["forward"]: r for r in read(folder / "operands.json")["files"]}
    for row in seal["rows"]:
        base = f'{row["scene"]}_{row["grid"]}_{row["pose_packet"].removeprefix(row["scene"]+"_")}'
        xyz = operands[base + ("_G1B1" if row["geometry"] else "_G0B0")]
        confidence = operands[base + ("_G1B1" if row["bundle"] else "_G0B0")]
        for key, record in operands[row["id"]].items():
            require(record["xyz_sha256"] == xyz[key]["xyz_sha256"], "Geometry source changed")
            require(record["confidence_sha256"] == confidence[key]["confidence_sha256"], "Confidence source changed")
            require(record["matching_sha256"] == confidence[key]["matching_sha256"], "Match bundle changed")
    observed = read(PUBLIC / "observations.json")
    associations = read(PUBLIC / "associations.json")
    expected = {(r["id"], reference) for r in seal["rows"] for reference in plan["references"]}
    require(len(associations) == 96 and {(r["run_id"], r["reference"]) for r in associations} == expected, "Metric associations missing")
    registry = read(ROOT / "manifests/dense_focal_inputs.json")
    for association in associations:
        row = next(r for r in seal["rows"] if r["id"] == association["run_id"])
        record = observed[row["input_key"]][row["prediction"]][association["reference"]]
        require(record["metrics"] == association["metrics"] and record["identity"] == association["identity"], "Metric association changed")
        reference = registry["references"][row["input_key"]]
        require(record["identity"] == dict(prediction=row["prediction"], ref_archive=reference["sha256"],
                                           reference=association["reference"], valid_pixels=reference["valid_pixels"]), "Reference identity differs")
    scores = read(PUBLIC / "scores.json")
    expected_scores = summarize(seal, observed, read(PARENT / "predictions.json"), read(PARENT / "metrics.json"))
    require(scores == dict(seal_sha256=sha256(PUBLIC / "predictions.json"), **expected_scores), "Factorial result changed")
    audit = read(PUBLIC / "match_audit.json")
    require(audit["seal_sha256"] == sha256(PUBLIC / "predictions.json"), "Audit predates prediction seal")
    require(audit["source_sha256"] == text_sha256(ROOT / "scripts/audit_mast3r_crossover_matches.py"), "Audit code changed")
    require(len(audit["states"]) == 6 and set(audit["same_view_checks"]) == set(plan["scenes"]), "Audit scope incomplete")
    calibration = np.asarray(read(ROOT / "manifests/sensor_frame_v14.json")["calibration"])
    pose_index = read(ROOT / "results/source_frame_spacing/inputs.json")["frames"]
    with np.load(PARENT / "pose_packets.npz") as packets:
        for scene, sequence in plan["scenes"].items():
            original = np.linalg.inv(np.linalg.inv(calibration) @ packets[scene + "_clean"])
            expected_cameras = reference_poses(original, calibration)
            record = audit["reference_cameras"][scene]
            np.testing.assert_allclose(record["world_to_camera"], expected_cameras, rtol=1e-10, atol=1e-10)
            require(record["original_pose_sha256"] == {str(f): pose_index[sequence][str(f)]["original"]["pose.txt"]
                                                        for f in plan["frames"]}, "Reference pose provenance differs")
            require(max(audit["same_view_checks"][scene]["max_native_pixel_error"]) < .01, "Reference projection check failed")
    for state, data in audit["states"].items():
        path = PUBLIC / "match_audit" / (state + ".npz")
        require(sha256(path) == data["archive_sha256"], "Match evidence changed")
        with np.load(path) as arrays:
            for record in data["records"]:
                for group in ("shared", "gained", "lost"):
                    prefix = f'{record["first"]}_{record["second"]}_{group}'
                    require(record[group] == describe(arrays[prefix + "_matches"], arrays[prefix + "_errors"], arrays[prefix + "_reason"]), "Match summary differs")
        require(aggregate(data["records"]) == data["unordered_pair_then_scene"], "Pair aggregation differs")
    attempt = PUBLIC / "attempts/tf32-disabled01/cases/heads_centered_clean_G0B0"
    require(not read(attempt / "bridge.json")["pass"], "Initial failed bridge missing")
    with np.load(attempt / "native.npz") as rebuilt, np.load(PUBLIC / "diagnosis/original_cache01/native.npz") as direct:
        for name in rebuilt:
            np.testing.assert_array_equal(rebuilt[name], direct[name])
    require(all(field["exact"] for row in read(PUBLIC / "bridge_diagnosis.json")["canonical_views"]
                for field in row["fields"].values()), "Canonical diagnosis changed")
    print("Verified 48 reconstructions, 24 archived bridges, 96 reference associations and six match audits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="Also independently reconstruct all pixels and verify full raw point maps.")
    replay(parser.parse_args().local)
