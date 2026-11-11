# Does the RGB reliability rule improve the reconstructed surface?

The rule reduced pixelwise squared log-depth error by 3.35% relative to the fixed quarter step. On the same targets it increased registered surface error by 9.25% against the strongest local comparator and by 10.66% against historical frozen VGGT. The registered surface criterion failed on both reference conventions. These are three previously examined development scenes.

| Method | Registered surface score | Released surface score |
|---|---:|---:|
| Cross-fitted four weights | 1.06873 | 1.01817 |
| Fixed quarter step | 1.04005 | 0.99614 |
| Cross-fitted single weight | 1.01393 | 0.99104 |
| Original geometry with supplied focal | 0.97941 | 0.98430 |
| Best local comparator | 0.97826 | 0.98181 |
| Historical frozen RTX 3090 | 0.96577 | 0.95780 |

Lower is better. Scores average accuracy and completeness relative to the local frozen FP32 comparator, weighting scenes equally. Against the strongest local comparator, weighted updates reduced registered error 2.83% on Heads but increased it 5.57% on Chess and 24.22% on Fire. All 18 surface measurements used the existing similarity alignment, rigid ICP and nearest-neighbor evaluator.

A supplementary depth comparison found that the weighted rule reduced error by 4.90% relative to the unchanged depth. A uniform shift with the same mean reduced error by 3.43%. Applying the mean only to eligible pixels reduced it by 2.95%. Shuffling the adjustments within the eligible pixels increased error by 1.19%. These outcomes distinguish where the depth changes help at a pixel level. They do not identify which part of the surface evaluator contributes to the loss.

[Registered surface comparison](../../docs/MOTION_RELIABILITY_SURFACE.md), [surface scores and all scene comparisons](weighted_surface_scores.json), [per-target spatial controls](motion_spatial_placement.json), and [two-disk inventory](weighted_surface_archive.json). Cloud cost was $0. Paid transfer and held-out allocations stay reserved.
