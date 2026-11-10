# The current proxy does not reliably rank finer prior benefits

All **39 saved predictions and 36 planned contrasts** were scored using the frozen learned support. Coverage is complete. The proxy often recognizes the large benefit of adding a prior to unstable visual-only adaptation, but it performs poorly when distinguishing both-prior adaptation from a single-prior baseline. **Do not deploy this proxy as a constraint-selection controller.**

| Add a constraint to this baseline | Proxy/reference sign agreement | Additions favored by two-fold rule | Harmful among favored | Harmful if always added | Rank association |
|---|---:|---:|---:|---:|---:|
| Pose → visual | 8/9 | 8/9 | 2/8 | 3/9 | 0.617 |
| Intrinsics → visual | 7/9 | 9/9 | 3/9 | 3/9 | 0.450 |
| Pose → intrinsics | 3/9 | 1/9 | 0/1 | 4/9 | −0.400 |
| Intrinsics → pose | 4/9 | 2/9 | 1/2 | 2/9 | −0.133 |

Each contrast matches scene and optimizer seed, with three development scenes and three seeds per family. Favoring an addition requires at least 2% improvement on both eligible proxy folds relative to the without-group endpoint. A harmful addition worsens accuracy **or** completeness by more than 1% relative to that endpoint. The two definitions differ intentionally: an average error improvement can coexist with a harmful individual metric. Signed agreement uses the combined normalized reconstruction score, so it need not agree with the harm count.

For the finer comparisons, the rule's mean reference-score gains were only 0.000311 when adding pose to intrinsics and 0.004937 when adding intrinsics to pose, versus 0.012170 and 0.023662 from always adding those respective groups in this batch. Thus the rule also discards useful corrections. These small differences are not established effects given the separately measured execution variability.

This is an **exploratory comparison of separate 40-step trajectories**, not a shared-gradient, same-state one-step intervention. V5 labels had already been inspected before the analysis was frozen. All proxy outputs were saved before their metric join, but that does not turn previously inspected development data into held-out evidence. Independent seeds, execution variation, shared scene bias and correlated contrast families prevent simple population or causal interpretations.

The result strengthens the need for a redesigned validation signal and stronger simple baselines. Sharing gradients can remove a numerical confound between proposals; it cannot repair a proxy that ranks genuine improvements incorrectly. Later MASt3R, held-out and verification allocations remain withheld. No additional GPU inference or RunPod spending was needed.

See the [frozen plan](../../manifests/prior_marginal_proxy_v8b.json), [all contrast outcomes](summary.json), and [scores frozen before reference joining](proxy_scores.json).
