# Can RGB tracks distinguish an erroneous focal prior from a real change of focal length?

The same supplied intrinsics can describe two different events. Suppose a single camera reports a focal length 12% above its four neighbors. A metadata fault changes the supplied matrix while leaving the image coordinates unchanged. An image zoom changes both. Comparing focal lengths alone gives identical information in these two cases. An RGB correspondence can distinguish them if the camera poses, scene geometry and image coverage constrain its epipolar line.

This development experiment follows the failed single-camera repair and scalar camera-weight tests. It changes only focal supervision. It compares a candidate correction against preserving a legitimate change in field of view. [Camera repair](../results/single_camera_repair/RESULTS.md) · [Camera-gradient bounds](../results/soft_camera_gradient_bounds/RESULTS.md)

## Matched observations

Reuse the public, frozen RGB tracks and calibrated camera convention from nine five-view development windows in Heads, Chess and Fire. Use each of the five view positions. Apply focal multipliers of 1.05, 1.12 and 1.25 and their reciprocals to both focal axes, holding the principal point fixed. For each of these 270 configurations, construct two track-level cases with identical supplied cameras. Add the nine unchanged windows as controls.

- **Metadata error.** Leave the observed RGB coordinates unchanged. The supplied focal multiplier is incorrect.
- **Consistent image transformation.** Apply the same focal multiplier to the selected view's coordinates around its principal point. The supplied focal multiplier is correct. This is an exact geometric control using existing tracks. It does not measure detector performance on resampled real images.
- Retain the same track IDs in both members of a pair. An observation must remain inside the raster both before and after transformation. Preserve the existing disjoint connected-track folds. Use the same image pairs, ordering and match counts in both cases.

## Candidate correction and comparisons

Select the view with the largest absolute deviation of log focal length from the five-view median. Treat a tie or an all-equal rig as an abstention. Propose the median focal lengths of the other four views, with all camera poses, principal points and other intrinsics fixed. This isolates the focal term that whole-camera suppression can discard along with valid pose information.

For each fold, compare the observed and proposed intrinsics on the same incident edges. Use the existing symmetric Sampson residual, a median absolute residual per edge, and an equal-weight median across neighboring cameras. Require at least two neighbors with six valid matches each. Apply the proposal only if both folds independently favor it by at least 0.25 pixels each. Retain the supplied camera otherwise. Include the one-fold rule, unconditional consensus correction, and unconditional retention as executable comparators. Report every abstention.

## Outcomes and interpretation

Measure absolute log-focal error for the selected test view and report the fraction of metadata errors corrected and legitimate zooms wrongly changed. Across each matched pair, either unconditional baseline has exactly half the original log-focal discrepancy as its mean error. Check that identity numerically. Report the paired mean error relative to that baseline for every scene and severity, the worst scene, match coverage and the clean controls. Repeated views and transforms within a window share observations. These are three development scenes.

For a subsequent input-to-gradient test, look for at least 20% lower paired mean error than the better unconditional baseline, no scene with increased paired error, at most 5% of legitimate zooms incorrectly changed, and no edits to the nine unchanged rigs. Publish the outcome even if it fails. Passing would motivate a separate test with perturbed poses, re-detected image features and original TCO gradients. Dense reconstruction and transfer remain separate experiments.
