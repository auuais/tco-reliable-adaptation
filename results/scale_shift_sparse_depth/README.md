# Sparse scale-and-shift result

All eighteen predeclared development reconstructions and 36 reference evaluations completed locally. Lower registered surface scores are better.

| Method | Centered grid | Shifted grid |
|---|---:|---:|
| Earlier depth-aware interpolation | 0.873839 | 0.856790 |
| Scene scale and shift only | 1.072797 | 1.072688 |
| Per-view scale and shift only | 0.924176 | 0.931582 |
| Per-view scale and shift plus local depth-aware correction | 0.844799 | 0.824435 |

The joint candidate improved by 3.32% and 3.78% against the strongest earlier control. Every development scene improved. The centered-grid gains were 4.32% on Chess, 0.77% on Fire and 6.25% on Heads. The shifted-grid gains were 4.83%, 0.95% and 7.20%. Historical comparison, component safety, no-scene-regression and released-reference checks passed. The 5% gain check failed on both grids, so neither advances.

Scene-only calibration did not explain the benefit. Per-view calibration alone remained weaker than the earlier local depth-aware method. Calibration combined with the local correction produced the strongest measured development reconstruction so far. This result concerns the original three development scenes with supplied poses and measured neighboring depth. Held-out and cross-dataset performance of this candidate remain unmeasured. The earlier held-out experiment is observed evidence for the earlier method.

No depth needed the registered positivity floor. Fitted per-view additive shifts reached approximately 12 centimeters. All coefficients were fitted from sparse measured ranges. Dense labels were used only after prediction sealing. Full point maps and compact replay evidence are verified on two physical disks. Use `python scripts/replay_scale_shift_sparse_depth.py` to reproduce all eighteen updates, coefficient diagnostics and score aggregation from public input archives. No rental was used.
