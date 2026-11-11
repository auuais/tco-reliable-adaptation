# Relative focal predictions eliminated false edits on the design windows

The revised method uses the four neighboring cameras to estimate each focal axis's common model bias. It changes a supplied focal length only when the corrected model prediction and both track folds favor the replacement. On the original three design windows it kept all 60 valid zooms under mild pose noise and corrected 43/60 faults. Mean relative focal-gradient error was 0.05346, **74.86% below** the better unconditional policy. All registered checks passed. [Fixed decision](../../docs/RELATIVE_FOCAL_CONFIRMATION.md) · [All paired decisions](summary.json)

| Mild pose noise | Faults corrected / 60 | Valid zooms changed / 60 | Mean relative focal-gradient error |
|---|---:|---:|---:|
| Raw VGGT + both track folds | 39 | 2 | 0.09976 |
| Both track folds | 43 | 9 | 0.07040 |
| Relative VGGT + both track folds | 43 | 0 | 0.05346 |
| Relative VGGT alone | 60 | 0 | 0.00000 |

**The simpler baseline was the strongest method.** Relative VGGT alone selected the correct target for every one of the 30 faulty configurations and 30 genuine zoom configurations before pose-seed duplication. It uses the existing VGGT output without running a matcher. On the same images, adding the geometry condition vetoed 17 true corrections in the primary group. The geometry-dependent rule also vetoed 6/30 on clean poses and 28/60 under stronger pose noise. No unique outlier exists in the hidden-change controls, so all these rules missed those changes.

These windows informed the redesign. The exact identification rate is a development result under one fixed optical input scale, a 12% intervention, and four consistent neighboring focal priors. It cannot establish that genuine lens changes will be protected on other cameras, that all-view calibration errors can be identified, or that dense accuracy will improve. The next test repeats both relative methods with frozen settings on the registered offset-25 windows. MASt3R and held-out funds stay reserved until a reconstruction improvement is established. Run `python scripts/relative_focal_confirmation.py` to replay the table. Additional cloud spending was $0.
