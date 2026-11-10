"""Independent pose/intrinsics switches; retain the released visual objective."""
import math

CONDITIONS = {"both": (1.0, 1.0), "pose_only": (1.0, 0.0),
              "intrinsics_only": (0.0, 1.0), "visual": (0.0, 0.0)}


def effective_config(base, pose_scale, intrinsics_scale, steps, seed, learning_rate):
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [pose_scale, intrinsics_scale]):
        raise ValueError("Prior scales must be finite and within [0,1]")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= .001 or steps < 0:
        raise ValueError("Invalid learning rate or step budget")
    result = dict(base)
    result.update(tco_steps=steps, seed=seed, tco_lr=learning_rate,
                  lambda_pose=base["lambda_pose"] * pose_scale,
                  lambda_intrinsics=base["lambda_intrinsics"] * intrinsics_scale)
    assert result["lambda_mv_consistency"] == .2 and result["mv_rgb_weight"] == 1
    assert result["mv_depth_weight"] == result["mv_normal_weight"] == 0
    return result


def validate_loss_components(config, losses):
    """Reject inactive residuals or an objective inconsistent with declared weights."""
    assert all(math.isfinite(v) for v in losses.values())
    if config["lambda_pose"] == 0:
        assert losses["pose_rot_loss"] == losses["pose_trans_loss"] == 0
    if config["lambda_intrinsics"] == 0:
        assert losses["intrinsics_loss"] == 0
    assert losses["depth_loss"] == losses["redundancy_loss"] == 0
    expected = (config["lambda_pose"] * (losses["pose_rot_loss"] +
                config["pose_translation_weight"] * losses["pose_trans_loss"]) +
                config["lambda_intrinsics"] * losses["intrinsics_loss"] +
                config["lambda_mv_consistency"] * losses["mv_consistency_loss"])
    # The upstream autocast path can accumulate scalar terms in reduced precision.
    assert math.isclose(losses["total"], expected, rel_tol=.02, abs_tol=1e-5), (losses["total"], expected)
