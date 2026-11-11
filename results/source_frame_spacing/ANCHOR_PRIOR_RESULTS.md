# Does model disagreement tell us when to use another measurement?

## RGB depth corrections

Partitioning the existing RGB triangulation rule by both correction size and disagreement across five image orders increased held-scene axial error by 0.042% relative to the original rule. Changes in error were +0.32% on Heads, +0.16% on Chess and -0.35% on Fire. A full correction improved 26.8% of low-disagreement and 25.3% of high-disagreement eligible reference pixels. These pixel counts reuse images and cannot be treated as independent observations. The registered 2% depth screening criterion failed, so the additional rule did not advance to whole-surface scoring. [Protocol](../../docs/ANCHOR_DISAGREEMENT_BENEFIT.md) and [fold weights, observations and bounds](anchor_disagreement_benefit.json).

## Weighting a supplied camera rig

The separate camera-fusion test used the existing synthetic one-degree and 2%-of-rig-diameter perturbations. Mean angular gains were 3.6% to 10.9% and center gains were 3.8% to 19.5% across the three scenes. All arms used the same median depth, intrinsics and starting pose ensemble. The table compares the registered reference-camera errors across all ten camera pairs per scene and the canonical dense surface score on each reference. Lower values are better.

| Supplied pose weight | Relative rotation error (degrees) | Relative camera-center displacement RMSE | Registered surface | Released surface |
|---|---:|---:|---:|---:|
| None | 1.740 | 0.0483 | 0.97383 | 0.98627 |
| Adaptive | 1.622 | 0.0429 | 0.97028 | 0.98343 |
| Half | 1.038 | 0.0285 | 1.04260 | 0.99634 |
| Full | 1.253 | 0.0343 | 1.26793 | 1.12199 |

Adaptive fusion improved the surface score by 0.36% on the registered reference and 0.29% on the released reference relative to the same median-depth model without the prior. It was 0.47% worse than the historical frozen model and 3.12% worse than the strongest registered local comparator. The half-weight baseline achieved the best camera measures among these four arms and degraded dense reconstruction. The preregistered whole-surface criterion failed. [Fusion protocol](../../docs/ANCHOR_PRIOR_FUSION.md), [pose errors recovered from the actual saved point maps](anchor_prior_pose_errors.json), and [whole-surface scores](anchor_prior_scores.json).

## Magnitude-matched controls

View-specific weights improved on applying the same scene-mean angular and center gains to each view in every scene on both references. Average gains were 0.32% registered and 0.17% released. The stronger comparison was a small constant gain. Adaptive scored 0.97028 registered versus 0.97061 for the best constant (10%); on the released reference it scored 0.98343 versus 0.97990 for the best constant (20%). These amounts correspond to a 0.034% gain and a 0.36% loss for adaptive weighting. Each best constant was identified after scoring and is an optimistic comparator. The evidence does not support a substantial gain from calibrating individual camera weights. [Control protocol](../../docs/ANCHOR_PRIOR_GAIN_CONTROLS.md) and [all 36 reference evaluations](anchor_prior_control_scores.json).
