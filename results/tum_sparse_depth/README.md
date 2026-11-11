# TUM confirmatory result

The redesigned per-view scale-and-shift plus local depth-aware correction was evaluated on all fifteen reserved TUM targets, three sequences and both sparse grids. All 48 reconstructions were sealed before opening target depth. The common reference compensates the depth/RGB timestamp difference. Lower surface scores are better, with each sequence normalized by ordinary VGGT and weighted equally.

| Method | Centered grid | Shifted grid |
|---|---:|---:|
| Ordinary native VGGT | 1.000000 | 1.000000 |
| Calibrated intrinsic matrix, model poses | 1.318312 | 1.318312 |
| Supplied poses, unchanged depth | 1.172586 | 1.174238 |
| Per-view measured scale | 0.885010 | 0.888540 |
| Spatial interpolation | 0.587640 | 0.595968 |
| Uncalibrated depth-aware interpolation | 0.592567 | 0.588134 |
| Scene scale-and-shift only | 1.171935 | 1.167565 |
| Per-view scale-and-shift only | 0.755834 | 0.761525 |
| Calibrated depth-aware candidate | 0.588805 | 0.588190 |

The primary candidate was 0.20% worse than spatial interpolation on the centered grid and 0.01% worse than the uncalibrated depth-aware method on the shifted grid. All three advancement checks failed on both grids. Against the strongest respective controls, centered per-sequence gains were plus 1.68%, minus 3.71%, and plus 5.36% on long_office_household, structure_texture_far and nostructure_texture_near_withloop. Shifted gains were minus 1.78%, minus 3.00%, and plus 11.30%.

The candidate reduced metric mean absolute log-depth error to 0.02010 and 0.01992, versus 0.02099 and 0.02063 for the respective strongest surface controls. It improved accuracy but worsened completeness enough to fail component safety. The development benefit therefore did not establish a consistent cross-dataset advantage. These comparisons use the same known benchmark poses, calibration and sparse ranges.

The 5% internal spending target had also blocked zero-rental confirmation. That scope was corrected before inference. All TUM execution used local hardware. Original development and held-out decisions remain recorded as failed. Later rental allocations are reserved. The full public replay is `python scripts/replay_tum_sparse_depth.py`.

Fresh-process inference reproduced every depth, intrinsic, extrinsic and native-point tensor exactly on all three sequences. Nine independent surface checks used float64 alignment and Open3D core KNN. Accuracy and completeness agreed within the predeclared tolerances in all nine checks. All six candidate normal-consistency checks passed. All three ordinary-native normal checks failed, with absolute discrepancies between 0.00464 and 0.03365. Independent verification is therefore partial.

A coordinate audit found that every native coordinate was representable as bfloat16 and 34.9%, 44.6% and 57.5% of valid native points were duplicates. Candidate maps had no duplicate points at the stored precision. Degenerate neighborhoods and precision-sensitive normal estimation are a possible explanation for the native normal discrepancies. The audit does not prove the full cause. The original normal values and failed verification flags are preserved for further investigation. The primary distance-based rejection is supported by the independent checks.

Native model archives, every sparse measurement, full prediction checksums and score records are public. Full point maps and captures are backed up on two physical disks. No RunPod compute was used for this experiment.

The normal-score follow-up reproduced the discrepancy with both the default and iterative eigensolvers. For a fixed aligned cloud, SciPy and Open3D core neighbor queries agreed. Across the two alignments, 43.9%, 62.0% and 72.1% of the sampled native normal-estimation neighborhoods changed. The fractions whose absolute normal dot fell below 0.99 were 9.4%, 23.5% and 36.1%. Mean aligned point displacement ranged from roughly 0.002 to 0.686 millimeters. The geometry was heavily quantized and nearest-neighbor boundary ties were common. This demonstrates neighborhood and normal-score sensitivity to numerical alignment differences on the archived native maps. It supports a mechanism for the discrepancy, while leaving all original normal values and failed tolerance checks intact.

Run `python scripts/verify_tum_verification.py` to check the independent-distance flags, the three normal failures, all twelve exact repeated tensors, and the identities and backend comparisons of the numerical diagnostics. A successful reporting check certifies consistency of the recorded evidence. It does not convert the failed normal tolerances into passes.
