# Ten-update early refresh on three development rooms

42 of 45 registered model slots delivered scored reconstructions. Each room used the same five input images for all arms and its three assigned seeds. Accuracy measures distance from the predicted surface to the measured surface. Completeness measures the reverse distance. Smaller is better.

## Paired comparisons

Each percentage is the mean of the three within-room ratios, using the median of three matched seeds per method and room. A timing ratio below one favors Refresh1. The limits for the primary comparison with Full10 are +1% accuracy error, +3% completeness error and a shorter scene time.

| Refresh1 compared with | Accuracy change | Completeness change | Scene time change | Peak live change |
| --- | ---: | ---: | ---: | ---: |
| Full10 | -0.82% | -2.72% | -53.94% | -1.03% |
| Cache10 | -0.24% | -0.77% | +9.55% | +29.54% |
| Refresh1Uncached | +0.45% | -0.07% | -26.17% | +0.00% |
| Full40 | Incomplete pairs | Incomplete pairs | Incomplete pairs | Incomplete pairs |

The bootstrap uncertainty below is descriptive on these previously used rooms. Development decisions use the registered point estimates. Formal confirmation requires six fresh visits.

- Refresh1/Full10 accuracy error ratio 0.9918. Its one-sided 95% lower and upper bounds are 0.9439 and 1.0442.
- Refresh1/Full10 completeness error ratio 0.9728. Its one-sided 95% lower and upper bounds are 0.9005 and 1.0542.
- Refresh1/Full10 scene time ratio 0.4606. Its one-sided 95% lower and upper bounds are 0.1806 and 0.7358.

## Absolute measurements

Each entry is a median over the completed seeds shown. A partial Full40 row describes its measured runs only.

| Room | Arm | Scored seeds | Accuracy (mm) | Completeness (mm) | Scene time (s) | Peak live (GiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| chess/seq-03 | Full10 | 42, 43, 44 | 16.823 | 21.424 | 38.629 | 7.822 |
| chess/seq-03 | Cache10 | 42, 43, 44 | 17.726 | 23.634 | 18.507 | 5.976 |
| chess/seq-03 | Refresh1 | 42, 43, 44 | 17.835 | 23.027 | 20.873 | 7.741 |
| chess/seq-03 | Refresh1Uncached | 42, 43, 44 | 17.706 | 23.313 | 28.274 | 7.741 |
| chess/seq-03 | Full40 | 43, 44 | 17.686 | 21.620 | 355.319 | 7.822 |
| fire/seq-03 | Full10 | 42, 43, 44 | 15.505 | 18.449 | 178.405 | 7.822 |
| fire/seq-03 | Cache10 | 42, 43, 44 | 15.685 | 18.468 | 18.388 | 5.976 |
| fire/seq-03 | Refresh1 | 42, 43, 44 | 15.487 | 18.131 | 20.881 | 7.741 |
| fire/seq-03 | Refresh1Uncached | 42, 43, 44 | 15.492 | 18.088 | 28.225 | 7.741 |
| fire/seq-03 | Full40 | 42 | 15.375 | 18.215 | 394.247 | 7.822 |
| heads/seq-01 | Full10 | 42, 43, 44 | 12.886 | 13.904 | 27.194 | 7.822 |
| heads/seq-01 | Cache10 | 42, 43, 44 | 11.818 | 11.723 | 19.256 | 5.976 |
| heads/seq-01 | Refresh1 | 42, 43, 44 | 11.810 | 11.967 | 19.700 | 7.741 |
| heads/seq-01 | Refresh1Uncached | 42, 43, 44 | 11.732 | 11.876 | 26.740 | 7.741 |
| heads/seq-01 | Full40 | 42, 43, 44 | 14.083 | 15.584 | 73.878 | 7.822 |

## Gate and coverage

The equal-step development comparison gave Refresh1 a verdict of **pass**. The registered follow-up schedule is **Narrow**.

Even if every Full40 control had used the entire 8.75 GiB cap, the mean Refresh1/Full40 peak ratio would be at least 0.8847. This bound can rule out the 0.764 memory target. The missing Full40 errors and elapsed times remain unknown.

| Room | Arm | Seed | Terminal outcome | Recorded attempts |
| --- | --- | ---: | --- | ---: |
| fire/seq-03 | Full40 | 44 | timeout | 2 |
| fire/seq-03 | Full40 | 43 | timeout | 2 |
| chess/seq-03 | Full40 | 42 | timeout | 2 |

All costs here are local. RunPod spend was USD 0. [Case-level scores and timing](a_development_rows.json) · [Independent ratios and uncertainty](a_development_independent.json) · [Follow-up decision](a_alternative_gate.json)
