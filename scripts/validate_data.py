"""Check camera conventions, masks and projection round-trips before inference."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "TCO"))
import numpy as np
from datasets.sevenscenes import SevenScenes
from utils.geometry import closed_form_inverse_se3

dataset = SevenScenes(SEVENSCENES_DIR="/data/data/7scenes", split="test", load_img_size=518,
                      cache_file=str(ROOT / "environment" / "validation_dataset_cache.npy"))
data = dataset.get_data(sequence_name="heads/seq-01", ids=[0, 200, 400, 600, 800])
images = data["images"].numpy()
assert images.dtype == np.float32 and 0 <= images.min() <= images.max() <= 1
assert np.isfinite(images).all()
records = []
for index, frame in enumerate([0, 200, 400, 600, 800]):
    world = data["pointclouds"][index]
    valid = data["valid_mask"][index]
    extr = data["extrs"][index]
    intr = data["intrs"][index]
    y, x = np.nonzero(valid)
    select = np.linspace(0, len(y) - 1, min(1000, len(y)), dtype=int)
    y, x = y[select], x[select]
    camera = world[y, x] @ extr[:3, :3].T + extr[:3, 3]
    pixels = camera @ intr.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    error = np.linalg.norm(pixels - np.stack([x, y], axis=1), axis=1)
    assert camera[:, 2].min() > 0
    # KinectFusion's stored rotations are not perfectly orthonormal. The release
    # uses transpose-as-inverse, which has a small, measurable real-data residual.
    # Test the mathematical projection with the exact inverse of the transform
    # actually used to create the reference points; retain nominal residuals.
    used_c2w = closed_form_inverse_se3(extr[None])[0]
    exact_w2c = np.linalg.inv(used_c2w)
    exact_camera = world[y, x] @ exact_w2c[:3, :3].T + exact_w2c[:3, 3]
    exact_pixels = exact_camera @ intr.T
    exact_pixels = exact_pixels[:, :2] / exact_pixels[:, 2:]
    exact_error = np.linalg.norm(exact_pixels - np.stack([x, y], axis=1), axis=1)
    assert exact_error.max() < 0.001, exact_error.max()
    raw_pose = np.loadtxt(pathlib.Path("/data/data/7scenes/heads/seq-01") / f"frame-{frame:06d}.pose.txt")
    w2c = np.eye(4)
    w2c[:3] = extr
    inverse_error = float(np.abs(w2c @ raw_pose - np.eye(4)).max())
    reference_w2c = closed_form_inverse_se3(raw_pose[None])[0]
    assert np.allclose(w2c, reference_w2c, atol=1e-10, rtol=0)
    records.append({"frame": frame, "valid_fraction": float(valid.mean()),
                    "nominal_SE3_roundtrip_pixel_error": float(error.max()),
                    "exact_inverse_roundtrip_pixel_error": float(exact_error.max()),
                    "world_to_camera_inverse_error": inverse_error,
                    "rotation_orthonormality_error": float(np.abs(raw_pose[:3, :3].T @ raw_pose[:3, :3] - np.eye(3)).max()),
                    "sample_depth_range_m": [float(camera[:, 2].min()), float(camera[:, 2].max())]})
report = {"status": "passed", "image_shape": list(images.shape), "frames": records,
          "note": "Tests numerical convention consistency, not physical calibration accuracy."}
(ROOT / "environment" / "data_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
