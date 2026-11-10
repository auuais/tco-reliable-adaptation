# Repeatability and controlled calibration: stage contract

Authorized 2026-11-11. Prepare locally, execute the next diagnostic GPU stage within **USD 3 total**, archive complete evidence, and commit code and results to the public repository. Preparation, CPU scoring and method redesign incur no RunPod compute cost.

The user explicitly chose to **wait for a redesigned method before spending later allocations if these diagnostics do not yield a reliable improvement**. MASt3R transfer (USD 25), held-out/cross-dataset evaluation, and independent verification are therefore gated. Storage/contingency is a reserve, not an experiment or a target to spend. Keep the existing USD 200 overall preferred ceiling and account for retained storage separately. No claim of a publishable improvement follows automatically from completing a stage.

## Immediate sequence

1. Freeze and run a CPU-only evaluator study: four existing saved predictions, each in three fresh processes at one and four numerical-library threads (24 evaluations). Include the problematic Heads visual-only run and Fire visual-only runs from v4 and v5, plus the v5 Fire both-prior control. Preserve original metrics. Verify prediction and reference hashes. Report variation rather than hiding failed stability thresholds.
2. Freeze a separately instrumented same-device adaptation-repeat study and a calibration study. Distinguish identical-seed executions from different optimizer seeds. Record initial predictions, LoRA initialization, final adapters, RNG states and loss trajectories. The historical v5 runner and evidence remain immutable.
3. Isolate K-prior changes: released K, axis-aware resize with the released nominal focal, and axis-aware resize with the registration RGB focal (525). Keep images, pose priors, optimization and the primary evaluation reference fixed. An alternate common calibration reference, if implemented, is a separately named sensitivity analysis applied to every condition, including zero-step controls. It is not independently verified physical calibration truth.
4. Collect and validate all planned runs, archive to two local drives, and stop paid compute before interpretation or report writing. Include aborted attempts and setup time in the USD 3 limit. No repeat rental may reset that cap.
5. Decide from complete development evidence. A simple calibration correction is useful engineering evidence but not automatically a novel method. If results are unreliable or the advantage is insufficient, redesign locally before releasing any later allocation. Compare normalized/robust constraints and simple baselines before a sophisticated controller. Held-out labels remain unavailable to selection and tuning.

## Decision criteria

For CPU scoring, predeclare distance-metric relative range and normal-consistency absolute range thresholds of 1e-6. These are numerical diagnostic flags, not significance tests. Four purposively selected predictions cannot establish stability of every possible input or platform.

For GPU studies, retain the existing equal-scene normalized accuracy/completeness score and individual-metric harm criterion (>1% worse than fresh zero-step control). Report all seeds and repeated executions, loss traces, latency and memory. Separate within-seed execution noise from between-seed variation. Do not attribute a change to hardware without a controlled hardware comparison.

The frozen design has **40 jobs / 1,480 updates**: three fresh zero-step controls; 27 both-prior runs across three scenes, three seeds and three calibration choices; plus ten additional executions completing the three-repeat both/visual groups on Heads and Fire at seed 42. Two repeat-0 runs are shared with the calibration design by predeclared identity. Calibration summaries always use repeat 0, never whichever repeat performs best. Order is shuffled deterministically inside seed blocks.

A calibration setting passes the conservative development diagnostic only with at least 5% aggregate score improvement over released K, no increase in harmful cells, no scene-mean regression greater than 1%, and an absolute score gain larger than the maximum observed same-seed both-prior score range. These are development criteria, not significance tests or publication guarantees. Calibration repair alone does not satisfy the requirement for a novel method. Full per-step focal residuals and Smooth-L1 slopes will show whether changing K actually changes the optimization signal.

All 40 saved predictions will additionally receive CPU scoring against the same alternate axis-aware 525-focal reference, including zero-step controls. This sensitivity analysis retains released pose handling, changes no adaptation output, and reports separately. Depth is recovered by the exact inverse of the transform used to generate the stored reference, then reprojected; reconstruction of the original released reference must agree within 1e-6 metres before accepting the alternate reference. This is calibration sensitivity, not independent ground truth.

Local integration checks passed seven two-view 224-pixel jobs, including one 40-update run (43 updates total), at 8.06 GiB peak allocation on the RTX 3080. All three zero-step calibration variants produced bitwise identical predictions, also matching the historical v5 zero-step check. A later one-update check verifies the added per-step residual capture and actual supplied-K metadata. These reduced-resolution checks establish pipeline behavior only.

## Budget status at stage opening

No GPU was started for this stage. Prior v5 session cost approximately USD 0.66; that is historical, separately accounted spending. The v5 A40 pod is stopped with no persistent volume and no idle charge. The older retained 20 GB stopped volume continues to cost approximately USD 0.13/day until an authorized storage decision. Live prices/balance must be checked before startup. The existing account balance is not the overall research budget.
