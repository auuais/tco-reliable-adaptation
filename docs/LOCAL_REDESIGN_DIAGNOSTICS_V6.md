# Local diagnostics supporting a method redesign

These exploratory checks are separate from the frozen 40-job calibration study. They consume no RunPod allocation, inspect only the existing development scenes, and cannot establish an accuracy improvement or a publishable method. The later transfer and held-out allocations remain gated.

## Numerical update comparison

The first local test uses four fresh processes: default attention and math-only attention, each repeated twice on Heads with two 168x224 views, seed 42 and two updates. Rendering is disabled in both policies to isolate the remaining model/constraint path. The original probe and baseline runner are unchanged. Commands and source hashes were saved before execution in the private evidence archive.

Default attention produced identical initial adapters, sampled RNG states and first scalar losses, but first clipped gradients differed by relative L2 **0.005442893**. Final pointmaps differed by up to **0.00966668** in their unaligned coordinates. Math-only attention produced bitwise identical first clipped gradients and final pointmaps in its two executions. These are raw numerical diagnostics, not metric reconstruction errors. See [complete summaries](../results/attention_probe_v6_local/summary.json).

The result associates the changed attention policy with repeatability in this small, rendering-free setting. It does not prove which kernel causes the original variation, establish repeatability at paper resolution, or validate a full 40-step run. A follow-up records actual operator names, tests flash-only execution, and separately restores rendering under math attention. Unsupported/failed cases remain in the denominator and their logs are retained.

PyTorch's documented [SDPA backend context](https://docs.pytorch.org/docs/2.7/generated/torch.nn.attention.sdpa_kernel.html) selects implementations; its [attention documentation](https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.scaled_dot_product_attention.html) cautions that implementations can differ numerically. Enforcing one backend may change outputs, memory and latency. Such a policy must therefore be a named experimental factor, never a silent repair to a published reproduction.

## Fixed correspondence baseline

Before inspecting v6 outcomes, cache SIFT matches from the resized RGB images only. Use mutual nearest-neighbor ratio 0.8, image-only fundamental-matrix RANSAC at 1.5 pixels, and at least 20 matches per retained pair. Partition connected match components by a deterministic hash; the two folds share no component under the declared half-pixel keypoint identity. Correspondences are never recomputed after adaptation. These are standard [OpenCV matching operations](https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html), not a proposed contribution; the executed runtime is OpenCV 4.11.0.

For each fold, compare the world points predicted at fixed matched pixels using bilinear sampling. Divide their Euclidean separation by the median pairwise distance of fixed image-grid anchors from the current pointmap. This removes global translation, rotation and positive scale from the score; it does not remove shape distortion or certify correct geometry. Average pair medians equally so a heavily matched pair does not dominate. Reject nonfinite or collapsed predictions. Require each fold to connect all five views with at least 100 matches, and at least 10 matches per included pair. A failure to meet this coverage rule is an abstention.

The exploratory offline acceptance rule requires both fold scores to improve at least 2% against the fresh zero-step prediction. Freeze proxy scores before joining reference evaluation labels. Report coverage, abstention, within-scene rank association and harmful runs among the would-be accepted updates. Do not fit thresholds, pool dependent pixels into confidence intervals, or claim held-out performance. The two folds can still share scene-level bias; they are not independent datasets.

The image-only preparation found 1 retained pair / 20 matches for Heads, 3 / 285 for Chess and 6 / 228 for Fire. Heads and Chess do not connect all sampled views. These failures must not be hidden by lowering coverage thresholds after inspecting outcomes. A more capable matcher or a different validation design would need a separately named, frozen follow-up. Sparse matching coverage is already a reason not to deploy this gate as a general controller.

## What a research contribution would still need

Existing [Test3R](https://arxiv.org/abs/2506.13750) already adapts reconstruction using geometric consistency. [GeoCond](https://arxiv.org/abs/2609.18465) already estimates reliability and gates refinement. [Info3R](https://arxiv.org/abs/2609.21938) modulates streaming state updates using information signals and dynamically resets state. [Fast Spatial Memory](https://arxiv.org/abs/2604.07350) introduces Fisher-weighted elastic test-time updates. These approaches differ in backbone, supervision and setting, but make generic consistency, confidence gating, adaptive rates or elastic regularization insufficient novelty by themselves.

Our unproven hypothesis concerns the **marginal benefit of a particular external constraint group**: compare a proposal with and without that group from identical parameter, optimizer and RNG snapshots, then measure its effect on fixed independent visual support. Reliable numerical comparisons and support coverage are prerequisites. Demonstrate benefit beyond lower learning rate, fewer steps, normalized/robust prior losses and conventional consistency gating at matched total compute before transferring to MASt3R. This is a testable direction, not a confirmed new method or a promise of conference acceptance.
