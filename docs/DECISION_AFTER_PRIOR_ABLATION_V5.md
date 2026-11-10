# Decision after pose and intrinsics ablation

The [complete 39-run study](../results/prior_ablation_v5/RESULTS.md) favors both camera penalties overall: score 0.9269 and 2/9 harmful executions, versus 0.9506 and 5/9 for pose-only, 0.9391 and 3/9 for intrinsics-only, and 3.8740 and 8/9 for visual-only. This is development evidence from three inspected sequences, not a new method or a held-out claim.

The simple hypothesis that one camera penalty can be removed globally is not supported by this batch. Per-scene mean winners differ, but a post-hoc scene-level oracle over these four fixed settings improves the score by only 1.51%. That restricted diagnostic gives little headroom for a complex selector under the current clean-pose/nominal-intrinsics setting.

The earlier visual-only Fire benefit did not repeat. GPU/CPU/driver differences and unresolved numerical variability prevent assigning a cause. Prioritize reproducibility and calibration before drawing a mechanistic conclusion about gradient conflict or implementing a sophisticated controller.

## Next study order

1. **Local evaluation stability.** Reevaluate identical saved point maps with fixed reference targets and documented CPU/thread settings. Separate variation in evaluation from variation in predictions. Use this as a diagnostic, preserving the original published metrics.
2. **Small same-device repeat study.** Freeze settings and repeat identical seeds on one device for both and visual-only, emphasizing the observed failure cases. Record initial/final predictions, initial LoRA state, all relevant RNG states and loss traces. Change precision/determinism settings only in separately named conditions; account for latency. Do not average changed methods together under a reproducibility label.
3. **Calibration with a common reference.** Separate the image-resize transform from the 525-versus-554.256 focal choice. Keep a single declared evaluation reference within each comparison. Report released-protocol compatibility and any alternative reference protocol separately, with fresh zero-step controls. Never change input calibration and its scoring target together without exposing that confound. Audit pose orthonormality explicitly before any optional correction.
4. **Residual scaling and uncertainty.** Compare normalized focal residuals and ordinary robust/global-weight baselines before a controller. The fixed principal-point offset is a constant, not an optimizable signal. A useful gate must operate without reconstruction ground truth and charge probes, matching, rejection and rollback to its runtime.
5. **Publication gate.** Advance only if a useful improvement survives held-out scenes, a second public dataset and uncertain/estimated priors, with cost-matched baselines and stable repeats. The current component ablation alone is insufficient. Delay MASt3R transfer until that evidence exists.

The next rental session should remain a separately frozen, locally prepared **USD 3 maximum** within the existing prior-ablation allocation. The current session cost approximately USD 0.66 and all GPU compute is stopped. No follow-up GPU jobs have been started. Read BUDGET.md for the overall USD 200 ceiling and retained-storage costs.

## Handoff constraints

Keep all historical runners/manifests/evidence immutable, including v5. Its execution commit is `568f92b7a5e6af982f3b3872a53928398db7688d`; later commits contain collection and audit reporting. New hypotheses require new study IDs and manifests. Account for the documented Windows CRLF/Linux LF source-byte distinction without discarding raw hashes. CI now installs the CPU loss-test dependencies, fetches the exact upstream commit, verifies both frozen manifests and reproduces the camera-parameterization evidence.
