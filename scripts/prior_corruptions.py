"""Synthetic prior diagnostics. Evaluation targets and global RNG are never modified."""
import hashlib
import json
import pathlib

import numpy as np

PROFILES_PATH = pathlib.Path(__file__).resolve().parents[1] / "configs/diagnostic_profiles.json"
PROFILES = json.loads(PROFILES_PATH.read_text())


def tensor_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def axis_rotation(axis, radians):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(radians) * skew + (1 - np.cos(radians)) * (skew @ skew)


def corrupt_priors(extrinsics, intrinsics, profile, seed, scene):
    """Extrinsics are world-to-camera. Perturb rotations and world camera centers.

    The first pose is fixed to remove reference-frame corruption. Every other
    camera gets an independent random direction with the declared fixed error
    magnitude. Translation is relative to the clean trajectory diameter, not
    distance from an arbitrary world origin. Focal bias is shared across views.
    No projection of the dataset's slightly nonorthogonal rotations is performed.
    These synthetic benchmark severities are NOT calibrated sensor uncertainty.
    """
    settings = PROFILES[profile]
    original_e, original_k = np.asarray(extrinsics), np.asarray(intrinsics)
    if original_e.ndim != 3 or original_e.shape[1:] not in [(3, 4), (4, 4)]:
        raise ValueError("Expected N x 3 x 4 or N x 4 x 4 extrinsics")
    if original_k.shape != (len(original_e), 3, 3):
        raise ValueError("Expected one 3 x 3 intrinsics matrix per view")
    assert np.isfinite(original_e).all() and np.isfinite(original_k).all()
    e, k = original_e.copy(), original_k.copy()
    centers = -np.linalg.solve(original_e[:, :3, :3].astype(np.float64),
                               original_e[:, :3, 3:4].astype(np.float64))[..., 0]
    diameter = float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())
    translation = settings["translation_diameter_fraction"] * diameter
    if settings["translation_diameter_fraction"] and diameter <= 1e-8:
        raise ValueError("Translation corruption undefined for a zero-baseline trajectory")
    deltas = []
    for index in range(1, len(e)):
        # Independent per-component streams make isolated and combined conditions paired.
        vectors = {}
        for component in ["rotation", "translation"]:
            digest = hashlib.sha256(f"{seed}:{scene}:{index}:{component}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            vector = rng.normal(size=3)
            vectors[component] = vector / np.linalg.norm(vector)
        r = original_e[index, :3, :3].astype(np.float64)
        if settings["rotation_deg"]:
            r = axis_rotation(vectors["rotation"], np.deg2rad(settings["rotation_deg"])) @ r
        center = centers[index] + translation * vectors["translation"]
        # Preserve clean inputs bit-for-bit, including stored nonorthogonality.
        if settings["rotation_deg"] or translation:
            e[index, :3, :3] = r
            e[index, :3, 3] = -r @ center
        deltas.append({"view": index, "rotation_axis": vectors["rotation"].tolist(),
                       "center_displacement": (translation * vectors["translation"]).tolist()})
    if settings["focal_scale"] != 1:
        k[:, 0, 0] *= settings["focal_scale"]
        k[:, 1, 1] *= settings["focal_scale"]
    metadata = {"profile": profile, "seed": seed, "scene": scene, "settings": settings,
                "provenance": "synthetic_corruption_of_reference_priors_for_diagnostics",
                "first_pose_unchanged": True, "translation_scale": "maximum_pairwise_clean_camera_center_distance",
                "trajectory_diameter": diameter, "center_displacement_magnitude": translation,
                "profile_file_sha256": hashlib.sha256(PROFILES_PATH.read_bytes()).hexdigest(),
                "extrinsics_sha256": tensor_hash(e), "intrinsics_sha256": tensor_hash(k),
                "deltas": deltas}
    assert np.array_equal(e[0], original_e[0])
    assert np.isfinite(e).all() and np.isfinite(k).all()
    return e, k, metadata
