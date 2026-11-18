"""CPU checks of the native-to-metric final calibration convention only."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from profiled_final_calibration import refine_native


OUTPUT = Path(__file__).resolve().parents[1] / "results/lora_profiled_v1/final_calibration_checks.json"


def fixture():
    depth = torch.full((5, 8, 10), 2., dtype=torch.float64)
    intrinsic = torch.tensor([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]], dtype=torch.float64)
    cameras = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    cameras[:, 0, 3] = torch.arange(5, dtype=torch.float64) * .2
    xy = torch.tensor([[x, y] for y in (1, 3, 5) for x in (1, 3, 5, 7)])
    samples = [dict(xy=xy.clone(), depth=torch.full((len(xy),), 2., dtype=torch.float64)) for _ in range(5)]
    matches = []
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4)):
        first = torch.tensor([[x, y] for y in (1, 3, 5) for x in range(j - i, 10)])
        second = first.clone()
        second[:, 0] -= j - i
        matches.append(dict(i=i, j=j, xy_i=first, xy_j=second))
    return depth, intrinsic, cameras, samples, matches, torch.ones_like(depth)


def world_points(depth, intrinsic, cameras):
    """Independent NumPy column-vector reconstruction, not the row-assembly code."""
    z = np.asarray(depth)
    k = np.asarray(intrinsic)
    transforms = np.asarray(cameras)
    if k.ndim == 2:
        k = np.repeat(k[None], len(z), axis=0)
    y, x = np.mgrid[:z.shape[1], :z.shape[2]]
    pixels = np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    points = []
    for view in range(len(z)):
        camera_points = np.linalg.solve(k[view], pixels) * z[view].ravel()
        homogeneous = np.concatenate([camera_points, np.ones((1, x.size))])
        world = transforms[view] @ homogeneous
        points.append(world[:3].T.reshape(z.shape[1], z.shape[2], 3))
    return np.stack(points)


def expected_failure(operation):
    try:
        operation()
    except ValueError as error:
        return str(error)
    raise AssertionError("Expected explicit invalid-input rejection")


def checks():
    torch.set_num_threads(2)
    native, k, cameras, samples, matches, confidence = fixture()
    original = tuple(t.clone() for t in (native, k, cameras, confidence))
    original_samples, original_matches = copy.deepcopy(samples), copy.deepcopy(matches)
    result, metric_cameras, scales, identity = refine_native(native, k, cameras, samples, matches, confidence)
    identity_error = float((result - native).abs().max())
    assert identity_error < 1e-12 and identity["g0"] == 1.
    torch.testing.assert_close(metric_cameras, cameras, rtol=0., atol=0.)
    torch.testing.assert_close(scales, torch.ones(5, dtype=torch.float64), rtol=0., atol=1e-12)
    # Identical native outputs must receive identical intervention, including the
    # frozen control. This is an endpoint equality fixture, not a model LoRA test.
    again = refine_native(native.clone(), k.clone(), cameras.clone(), copy.deepcopy(samples),
                          copy.deepcopy(matches), confidence.clone())
    for left, right in zip((result, metric_cameras, scales), again[:3]):
        torch.testing.assert_close(left, right, rtol=0., atol=0.)
    assert identity == again[3]
    for current, previous in zip((native, k, cameras, confidence), original):
        torch.testing.assert_close(current, previous, rtol=0., atol=0.)
    for current, previous in zip(samples, original_samples):
        for key in current:
            torch.testing.assert_close(current[key], previous[key], rtol=0., atol=0.)
    for current, previous in zip(matches, original_matches):
        for key in ("xy_i", "xy_j"):
            torch.testing.assert_close(current[key], previous[key], rtol=0., atol=0.)

    # Rotate the native world frame, add an arbitrary origin, and change units.
    # The final coordinates retain the native orientation and subtract C0 only.
    angle = .61
    rotation_z = torch.tensor([[np.cos(angle), -np.sin(angle), 0.],
                               [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]], dtype=torch.float64)
    rotation_x = torch.tensor([[1., 0., 0.], [0., np.cos(.27), -np.sin(.27)],
                               [0., np.sin(.27), np.cos(.27)]], dtype=torch.float64)
    rotation = rotation_z @ rotation_x
    origin = torch.tensor([3., -2., 1.], dtype=torch.float64)
    global_gain = 4.
    transformed = cameras.clone()
    transformed[:, :3, :3] = rotation
    transformed[:, :3, 3] = (cameras[:, :3, 3] / global_gain) @ rotation.T + origin
    native_units = native / global_gain
    metric, changed_cameras, changed_scales, changed = refine_native(
        native_units, k, transformed, samples, matches, confidence)
    assert changed["g0"] == global_gain
    torch.testing.assert_close(metric, native, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(changed_cameras[:, :3, :3], transformed[:, :3, :3], rtol=0., atol=0.)
    torch.testing.assert_close(changed_cameras[:, :3, 3], cameras[:, :3, 3] @ rotation.T, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(changed_scales, torch.ones_like(scales), rtol=1e-12, atol=1e-12)
    reconstructed = world_points(metric, k, changed_cameras)
    native_world = world_points(native_units, k, transformed)
    expected_world = global_gain * (native_world - origin.numpy())
    world_error = float(np.max(np.abs(reconstructed - expected_world)))
    np.testing.assert_allclose(reconstructed, expected_world, rtol=1e-12, atol=1e-12)

    # Different per-view errors exercise nonunit relative scales. Their correction
    # must scale the rays, never the optimized metric camera centers.
    distortion = torch.tensor([1.08, .94, 1.04, .96, 1.12], dtype=torch.float64)
    distorted = native_units * distortion[:, None, None]
    refined, refined_cameras, relative, fit = refine_native(distorted, k, transformed, samples, matches, confidence)
    assert float((relative - 1.).abs().max()) > .02
    predicted_world = world_points(refined, k, refined_cameras)
    native_points = world_points(distorted, k, transformed)
    centers = transformed[:, :3, 3].numpy()[:, None, None, :]
    expected = fit["g0"] * (centers - origin.numpy()) + relative.numpy()[:, None, None, None] * fit["g0"] * (native_points - centers)
    nonunit_world_error = float(np.max(np.abs(predicted_world - expected)))
    np.testing.assert_allclose(predicted_world, expected, rtol=1e-12, atol=1e-12)
    # A wrong implementation that scales the centers too must be detectable.
    wrong = relative.numpy()[:, None, None, None] * fit["g0"] * (native_points - origin.numpy())
    wrong_center_error = float(np.max(np.abs(predicted_world - wrong)))
    assert wrong_center_error > .01
    assert fit["penalized_fit_after"] <= fit["weighted_fit_before"] + 1e-12
    assert fit["normal_equation_max_residual"] < 1e-9
    assert all(gain == fit["g0"] for gain in fit["metric_scales"])

    # Changing only native units and global origin cannot alter the endpoint.
    unit_factor = 3.7
    changed_origin = torch.tensor([-1., 4., 2.], dtype=torch.float64)
    rescaled_cameras = transformed.clone()
    rescaled_cameras[:, :3, 3] = transformed[:, :3, 3] * unit_factor + changed_origin
    invariant = refine_native(distorted * unit_factor, k, rescaled_cameras, samples, matches, confidence)
    for first, second in zip((refined, refined_cameras, relative), invariant[:3]):
        torch.testing.assert_close(first, second, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(invariant[3]["g0"], fit["g0"] / unit_factor, rtol=1e-12)

    # Uneven per-view counts distinguish a pooled midpoint median (21.5) from a
    # median of per-view medians (20.5), or the lower middle sample (21).
    grid = torch.tensor([[x, y] for y in range(8) for x in range(10)])
    counts, next_value, uneven = [8, 8, 8, 8, 10], 1, []
    for count in counts:
        uneven.append(dict(xy=grid[:count].clone(), depth=torch.arange(next_value, next_value + count, dtype=torch.float64)))
        next_value += count
    midpoint = refine_native(torch.ones_like(native), k, cameras, uneven, [], confidence)[3]
    assert midpoint["g0"] == 21.5 and midpoint["gauge_ratio_count"] == 42
    assert midpoint["sample_counts"] == counts and midpoint["sample_exclusions"] == 0

    failures = {}
    invalid_sample = native.clone()
    invalid_sample[2, 1, 1] = -1.
    failures["nonpositive_fit_prediction"] = expected_failure(
        lambda: refine_native(invalid_sample, k, cameras, samples, matches, confidence))
    assert "counts_by_view=[0, 0, 1, 0, 0]" in failures["nonpositive_fit_prediction"]
    bad_measurement = copy.deepcopy(samples)
    bad_measurement[1]["depth"][0] = 0.
    failures["invalid_measurement"] = expected_failure(
        lambda: refine_native(native, k, cameras, bad_measurement, matches, confidence))
    missing = copy.deepcopy(samples)
    missing[0] = dict(xy=grid[:7], depth=torch.ones(7))
    failures["insufficient_view_coverage"] = expected_failure(
        lambda: refine_native(native, k, cameras, missing, matches, confidence))
    unrelated = native.clone()
    unrelated[0, 0, 0] = -1.
    negative_output = refine_native(unrelated, k, cameras, samples, matches, confidence)
    assert negative_output[3]["nonpositive_dense_pixels"] == [1, 0, 0, 0, 0]
    assert negative_output[0][0, 0, 0] < 0. and negative_output[3]["sample_exclusions"] == 0
    detached = refine_native(native.clone().requires_grad_(), k, cameras.clone().requires_grad_(), samples, matches, confidence)
    assert all(not value.requires_grad and value.grad_fn is None for value in detached[:3])
    return dict(
        scope="CPU synthetic endpoint validation only; no model inference, adapter training, or held-source accuracy scores.",
        torch_version=torch.__version__, device="cpu", dtype="float64",
        identity_depth_max_error=identity_error, identical_native_inputs_equal_outputs=True,
        native_inputs_unmodified=True, global_metric_gain=changed["g0"],
        world_point_max_error=world_error, nonunit_scale_world_point_max_error=nonunit_world_error,
        wrong_center_scaling_negative_control_error=wrong_center_error,
        native_units_and_origin_invariant=True, native_rotations_preserved=True,
        pooled_midpoint_median=midpoint["g0"], candidate_specific_sample_exclusions=0,
        nonpositive_outside_fit_preserved=True, outputs_detached=True,
        expected_failures=failures, identity_diagnostics=identity, distorted_diagnostics=fit,
        pass_all=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = checks()
    if args.verify:
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for key in ("identity_depth_max_error", "world_point_max_error", "nonunit_scale_world_point_max_error",
                    "wrong_center_scaling_negative_control_error", "global_metric_gain", "pooled_midpoint_median"):
            np.testing.assert_allclose(result[key], previous[key], rtol=1e-10, atol=1e-12)
        if not previous["pass_all"]:
            raise ValueError("Stored final calibration validation is not passing")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: result[key] for key in ("pass_all", "identity_depth_max_error", "world_point_max_error", "pooled_midpoint_median")}))
