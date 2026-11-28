# Three-run surface fusion on archived Full40 predictions

**Development gate failed.** The three development rooms supplied one prespecified triplet each. Every room used its first seed (42) as anchor.

Each entry gives mean accuracy error (prediction to reference) followed by mean completeness error (reference to prediction). Both use nearest-neighbour distances after alignment, and smaller is better. The comparative percentages give the arithmetic mean of within-room ratios.

| Room | Native single, mm | Reprojected single, mm | Native anchor, mm | Fusion, mm | Fusion vs native | Fusion vs reprojected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 13.612 / 14.734 | 13.728 / 14.538 | 14.481 / 15.793 | 14.565 / 14.911 | +7.01% / +1.20% | +6.10% / +2.57% |
| chess/seq-03 | 16.445 / 20.784 | 16.413 / 20.426 | 16.542 / 21.143 | 16.776 / 20.745 | +2.01% / -0.19% | +2.21% / +1.56% |
| fire/seq-03 | 15.691 / 18.586 | 15.736 / 18.114 | 15.403 / 18.846 | 15.475 / 18.075 | -1.38% / -2.75% | -1.66% / -0.21% |
| Equal-room mean | 15.249 / 18.035 | 15.292 / 17.693 | 15.475 / 18.594 | 15.605 / 17.910 | +2.55% / -0.58% | +2.22% / +1.30% |

## Coverage, disagreement and matched controls

| Room | Evaluable / emitted | Three-source coverage | Anchor fallback in fusion | Singleton fallback for 42 / 43 / 44 | Per-view median log-depth range |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 688,882 / 1,015,280 | 94.12% | 0.808% | 0.00% / 3.61% / 3.08% | 0.0483 to 0.0649 |
| chess/seq-03 | 716,290 / 1,015,280 | 95.90% | 0.116% | 0.00% / 2.28% / 1.94% | 0.0297 to 0.0486 |
| fire/seq-03 | 697,344 / 1,015,280 | 95.27% | 0.075% | 0.00% / 1.92% / 2.88% | 0.0422 to 0.0594 |

The supplied camera rig placed the independent runs into one coordinate system. The anchor camera grid stayed fixed for all reprojected singles and for fusion. All nine native scores matched their archived distances. Three copies of the anchor gave the identical saved point bytes as its singleton in each room. A second, separately launched scorer reproduced the four registered Chess control and fusion rows.

Across rooms, all three sources covered 94.12% to 95.90% of pixels. Fusion fell back to the anchor in 0.075% to 0.808%. The candidate was therefore able to combine outputs across nearly the entire image. Depending on the view, the median supported farthest depth exceeded the nearest by 3.0% to 6.7%.

The disagreement column gives the smallest and largest of the five per-view medians of the supported log-depth spread. Exponentiating the spread gives the largest-to-smallest supported depth ratio. The reprojected-single control includes camera-grid resampling and anchor fill. A log median favors the middle estimate; it has no signal identifying the most accurate seed when two seeds share a bias. These controls do not independently identify the contribution of the fixed anchor camera estimates.

Historical synchronized model times summed across a room's three ordinary Full40 trajectories averaged 225.3 s. The fusion kernel in the CPU worker averaged 1.12 s. These components were measured separately. They do not establish a matched end-to-end latency or a comparison to 120 ordinary updates.

## Decision

Accuracy beat the native single mean in 1 of three rooms. The gate required lower accuracy error than both single-run baselines, mean completeness within +3% of native, at least two rooms improving on native accuracy and a passing sham.

The registered follow-up stops here. Three-trajectory fusion did not earn late-checkpoint experiments, a 120-update competitor or fresh held-out evaluation.

The three rooms were previously used in development. They support a pilot decision only. Local CPU work incurred USD 0 rental.

[Room-level decision](b_development_decision.json) · [Scorer rows and independent repeats](b_development_rows.json) · [Support, fits and reference-isolation evidence](b_development_predictions.json)
