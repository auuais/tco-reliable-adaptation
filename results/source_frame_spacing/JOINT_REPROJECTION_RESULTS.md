# Can RGB reprojection distinguish useful camera and depth updates?

The joint fit minimized symmetric pixel reprojection errors on one set of RGB matches and used a disjoint set of matched features for acceptance. The supplied-pose version reduced validation loss by 16.66% on Heads, 3.80% on Chess and 38.28% on Fire. These are three reused development windows. Held-out feature tracks within an image window do not constitute held-out scenes. All four model variants passed the match-based acceptance check in every scene. [Protocol](../../docs/JOINT_RGB_REPROJECTION.md) and [fit parameters and fold costs](joint_reprojection_predictions.json).

| Output model | Registered surface score | Released surface score | Corresponding squared log-depth error | Corresponding normal error (degrees) |
|---|---:|---:|---:|---:|
| Initial camera/depth ensemble | 0.97383 | 0.98627 | 0.009073 | 37.509 |
| Fit poses only | 0.97856 | 1.00166 | 0.009073 | 37.509 |
| Fit depth only | 1.00177 | 1.00769 | 0.009051 | 37.493 |
| Fit both from RGB | 1.03833 | 1.05479 | 0.008893 | 37.455 |
| Fit both from RGB and supplied poses | 1.02020 | 1.04642 | 0.008908 | 37.451 |

Lower values are better. The primary RGB-plus-prior fit improved mean corresponding depth error by 1.83% and normal error by 0.059 degrees, but both directional surface distances worsened on both references. Registered accuracy distance rose from 16.63 to 17.45 mm and completeness distance from 16.65 to 17.40 mm. On the registered reference, the combined surface score worsened in every scene, by 7.34% on Heads, 2.02% on Chess and 5.00% on Fire. The original transfer criterion failed. [Canonical scores and raw components](joint_reprojection_scores.json) and [per-target paired geometry](joint_reprojection_paired.json).
