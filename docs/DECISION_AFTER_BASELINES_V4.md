# Decision after the first 39-run baseline screen

All 39 jobs completed and passed both remote and local validation. The raw archive is checksummed, and the rental GPU is stopped. [Full results and individual seed outcomes](../results/baselines_v4/RESULTS.md) remain available.

## What changed our understanding

LR 0.0003 had the lowest average combined relative distance error: 0.9227 versus 0.9473 for the released LR 0.001, a **2.6% reduction in that prespecified score**. Mean accuracy/completeness errors were 1.460/1.727 cm versus 1.528/1.797 cm. Harmful executions fell from 3/9 to 1/9 under the declared >1% threshold. Average normal consistency decreased from 0.7794 to 0.7705.

This is not a uniformly better setting. Most of its aggregate benefit comes from Chess; released LR 0.001 had better mean distance errors on Heads and Fire. Three seeds with unresolved numerical variability do not establish statistical superiority. The 2.6% development improvement does not meet our provisional 5% held-out advancement target and is not a publication contribution by itself.

The more informative result is the reversal across scenes. Visual-only adaptation achieved mean accuracy/completeness errors of **1.189/1.721 cm on Fire**, compared with **1.538/1.873 cm** for the same LR with camera priors. On Heads, visual-only errors were **3.931/6.978 cm**, compared with **1.291/1.385 cm** with camera priors. Every visual-only Heads run was harmful. Clean reference priors can therefore be valuable in one scene while constraining a better visual-only solution in another, under this particular objective and aligned evaluation protocol.

This observation motivates an input-dependent policy but does not show that one can select the better update without ground truth. The real contribution must be that selection mechanism and its demonstrated cost-adjusted benefit, beyond ordinary confidence or residual gates.

## Next bounded experiment

Before implementing a sophisticated controller, perform a **pose-versus-intrinsics component ablation** on the same development scenes. Use LR 0.0003 and 40 steps, keeping the visual loss weight at 0.2. Conditions:

1. Both camera-prior groups at released weights.
2. Pose only, intrinsics weight zero.
3. Intrinsics only, pose weight zero.
4. Visual-only, both prior weights zero.

Three scenes x three optimizer seeds x four conditions plus three fresh zero-step controls gives **39 jobs**. Freeze a new runner and manifest before execution; do not edit v4. The next rental session has a provisional **USD 3 cap within the USD 20 prior-ablation allocation**, with a 70-minute batch deadline. Prepare all code and tests locally first. All four conditions are rerun together because the physical GPU may change after a restart and numerical execution variability is unresolved.

Test that each zeroed weight removes only its own residual, and record the effective optimizer and loss weights. Keep all seeds and failures. Continue to explicit compute-budget and corrupted-prior tests after identifying the component interaction. Do not reinterpret the joint pose/intrinsics v4 result as proof about either component separately.

## Controller branch and stop rule

Use fixed, disjoint visual tracks/pairs to test whether a short with-group versus without-group probe predicts the direction of actual improvement offline. Ground-truth reconstruction error labels this diagnostic only; it cannot enter the online rule. Compare against residual magnitude, native confidence, geometric conditioning, gradient agreement and random selection. Include matching, probes and rejected steps in latency.

If those signals cannot distinguish beneficial from harmful constraints beyond simple gates, stop the counterfactual-control branch. If they can, implement bounded acceptance and rollback, then freeze the method before the reserved scenes and a second dataset. MASt3R transfer follows a demonstrated VGGT benefit; it is not the next expense.

## Local work and cost

The offload engineering probe completed one full-resolution update on the RTX 3080 at 8.54 GiB allocated memory, with matched zero-step outputs bitwise identical. It has not yet validated 40 steps or adapted-output equivalence. Use it for carefully labelled local debugging before considering it a replacement for cloud runs.

All 18 test sequences are prepared locally: 17,000 frames and 86 benchmark-selected views. The four reserved scenes have not been used for method metrics. The completed cloud session cost approximately **USD 0.67** from elapsed time and the quoted roughly USD 1/hour rate; observed balance change was USD 0.58 with delayed billing, not a settled invoice. Both stopped volumes currently cost roughly USD 0.26/day together; permanent deletion of the migrated duplicate awaits specific user confirmation. No GPU compute remains running.
