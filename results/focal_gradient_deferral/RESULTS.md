# Every tested focal ambiguity mattered within six optimizer updates

All 360 counterfactual focal choices produced different Smooth L1 slopes on at least one affected focal axis by update index 5 (the sixth update). Zero avoided the visual decision over the 40-update schedule. The registered threshold for funding a standalone timing experiment was 30%. We close that efficiency branch on these windows.

| First update requiring a decision, counted from zero | 0 | 1 | 2 | 3 | 4 | 5 | 6–39 | Never |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate pairs | 192 | 22 | 62 | 42 | 39 | 3 | 0 | 0 |

At the initial prediction, 168/360 choices gave identical clipped slopes on both axes. Skipping the check permanently on that basis would miss a subsequent difference in all 168. The result also held separately in Chess, Fire and Heads. A difference is the first point at which equal adapter updates can no longer be guaranteed from the focal loss alone. Measuring an actual change in dense reconstruction after that point requires another model run.

The comparison reuses 480 actual optimizer updates from 12 five-view full-model TCO runs across three scenes and four calibration conditions. It perturbs each supplied focal target in each direction at three scales. These 360 paired comparisons share trajectories. They measure the opportunity to defer a decision, without timing a matcher or fitting alternative trajectories. Exact FP32 replay reproduced every logged residual and clipped slope. A separate CPU test with upstream TCO intrinsics loss, a shared trainable parameter, gradient clipping and Adam reproduced the eager optimizer's finite trajectory for decisions required immediately, required later or never required.

## Replay

Run `python scripts/focal_gradient_deferral.py verify` with NumPy. The public trace contains the two predicted focal coordinates, targets, residuals and clipped slopes of every view and step. The summary includes per-candidate first differences, source log and archive hashes, and breakdowns by scene, condition, factor and direction. Use `python scripts/test_focal_gradient_deferral.py` with the pinned TCO checkout and CPU test dependencies to repeat the optimizer mechanism check. Additional RunPod cost was $0. [Registered analysis](../../docs/FOCAL_GRADIENT_DEFERRAL.md) · [Source runs](../canonical_sensor_factorial/RESULTS.md)
