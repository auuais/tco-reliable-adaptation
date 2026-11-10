# Coordinate precision changes the zero-step comparator

All 12 frozen CPU rescoring jobs completed. Each scene used exactly the same five captured model depth maps, poses, intrinsics and reference points for every arithmetic mode. No weights were updated and no new inference or cloud time was used. This isolates the last depth-to-world conversion in **our** zero-step TCO path. It does not test the authors' published zero-shot runner.

| Scene | Mean accuracy error, saved → FP32 (cm) | Mean completeness error, saved → FP32 (cm) | Mean normal consistency, saved → FP32 |
|---|---:|---:|---:|
| Heads | 1.362 → 1.340 | 1.290 → 1.393 | .719 → .803 |
| Chess | 1.889 → 1.852 | 2.433 → 2.604 | .664 → .742 |
| Fire | 1.631 → 1.624 | 1.897 → 2.099 | .663 → .723 |

Lower is better for each distance; higher is better for normal consistency. A finer coordinate grid improved normals and accuracy but worsened completeness here. Dense nearest-neighbor coverage after alignment is not monotone in coordinate precision. A rise in normals after adaptation cannot be attributed entirely to learning when the arithmetic also changes.

All three original GPU zero maps had 100% BF16-representable coordinates. Executing the upstream CPU BF16 unprojection reproduced two maps bit for bit; the Fire output was not identical. Independent NumPy float64 unprojection and upstream CPU FP32 produced similar metrics; both recovered all 15 known cameras by the prespecified dense pinhole checks. Neither the original nor CPU BF16 maps passed the strict camera diagnostic in any of the 15 views. For upstream FP32, maximum recovered intrinsic entry error was 0.000044, maximum extrinsic entry error below 0.00000008, and maximum depth error below 0.00000027 in model units.

**Does this change the calibration decision?** No. In a separately labeled post-hoc sensitivity, changing only the denominator from the stored zero to reconstructed FP32 moves the released/axis-nominal/525 scores from .9674/.9467/1.0162 to .9344/.9142/.9826. The nominal improvement relative to released remains about 2.2%, below the original 5% minimum. Do not substitute this analysis for the frozen primary gate. The change does affect which executions count as harmful relative to zero (3/3/5 becomes 2/2/5).

For future method comparisons use the same final unprojection precision on both sides, record it, and retain individual metrics. The earlier five-sequence subset used a historical comparator; this three-scene control does not retroactively correct its scores. Independent reruns of the full paper benchmark are still outstanding.

[All 12 rows and numerical checks](summary.json) · [Post-hoc denominator sensitivity](denominator_sensitivity.json) · [Frozen experiment](../../manifests/pointmap_precision_v9.json)
