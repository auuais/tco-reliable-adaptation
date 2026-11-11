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

## Does assigning a correction to the right pixel improve surface metrics?

All methods below use the same images and predicted cameras. The permutation preserves exactly the multiset of log-depth changes on eligible pixels in each target. Surface scores normalize accuracy and completeness by the frozen FP32 baseline. Smaller scores and distances are better. Higher normal consistency indicates closer surface orientation.

| Placement | Axial error reduction | Registered surface score | Released surface score | Registered accuracy (m) | Registered completeness (m) | Registered normal consistency |
|---|---:|---:|---:|---:|---:|---:|
| Learned spatial weights | 4.90% | 1.06873 | 1.01817 | 0.01894 | 0.01656 | 0.74092 |
| Uniform shift across image | 3.43% | 0.99442 | 0.99849 | 0.01707 | 0.01680 | 0.75926 |
| Uniform shift on eligible pixels | 2.95% | 0.99212 | 0.98183 | 0.01732 | 0.01639 | 0.75193 |
| Permuted among eligible pixels | -1.19% | 0.94090 | 0.88895 | 0.01971 | 0.01135 | 0.67745 |

The permutation produced a lower composite surface score than the learned placement on every scene and both references. On the registered reference, the mean reference-to-prediction distance fell from 16.56 to 11.35 mm, while prediction-to-reference distance rose from 18.94 to 19.71 mm and normal consistency fell by 0.0635. This is one fixed random permutation per target. It motivates checking metric rankings across seeds and sampling densities. It is not a reliable reconstruction improvement or an advancement result. [Registered placement comparison](../../docs/MOTION_PLACEMENT_SURFACE.md) and [all measured scores](placement_surface_scores.json).

The weighted rule also reduced corresponding-pixel 3D mean squared error against direct focal by 0.63% under the shared similarity and 1.12% when the similarity was refit. On released depth these gains were 0.33% and 0.80%. Thus the adverse registered surface ranking against direct focal is absent from paired Euclidean error after similarity fitting. [Paired 3D errors and fitted similarities](weighted_paired_geometry.json).
