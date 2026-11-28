# Hybrid alternative selection

Status — complete.

The registered rooms are heads/seq-01, chess/seq-03, fire/seq-03. The paired seeds are 42, 43, 44.

The development design pairs the alternative with Refresh1 on identical seeds.

Accuracy measures prediction-to-reference distance. Completeness measures reference-to-prediction distance. Positive error changes are worse. An error pair always gives accuracy first. Each arm's room estimate is its median across three seeds. The aggregate equally weights within-room ratios. A room with any missing seed pair is excluded from the estimate and prevents a selection pass.

Selection requires accuracy error at most +1%, completeness error at most +3% and strictly shorter time than Full10. The reported ranges give the exact 5th and 95th percentiles from descriptive resampling of rooms and paired seeds. Development and stress use previously evaluated rooms.

## Refresh1 versus Full10

9 of 9 seed pairs and 3 of 3 rooms are complete.

| Room | Candidate errors (mm) | Reference errors (mm) | Change (mm) | Error change (%) | Candidate / reference time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 11.810 / 11.967 | 12.886 / 13.904 | -1.076 / -1.937 | -8.35% / -13.93% | 19.70 / 27.19 |
| chess/seq-03 | 17.835 / 23.027 | 16.823 / 21.424 | +1.012 / +1.602 | +6.01% / +7.48% | 20.87 / 38.63 |
| fire/seq-03 | 15.487 / 18.131 | 15.505 / 18.449 | -0.019 / -0.318 | -0.12% / -1.72% | 20.88 / 178.40 |

heads/seq-01 used 688,882 evaluable points. Seed-only bounds were -8.35% to -7.56% for accuracy and -14.21% to -11.28% for completeness. Signed distance from the room's +1% / +3% limits was -1.205 / -2.354 mm (positive exceeds the limit).

chess/seq-03 used 716,290 evaluable points. Seed-only bounds were +2.97% to +7.36% for accuracy and +3.88% to +10.04% for completeness. Signed distance from the room's +1% / +3% limits was +0.844 / +0.960 mm (positive exceeds the limit).

fire/seq-03 used 697,344 evaluable points. Seed-only bounds were -0.60% to -0.12% for accuracy and -1.75% to +0.05% for completeness. Signed distance from the room's +1% / +3% limits was -0.174 / -0.871 mm (positive exceeds the limit).

Equal-room error changes were -0.82% and -2.72%. The hierarchical bounds were -5.62% to +4.44% and -9.95% to +5.42%, respectively. Speedup was 2.171× and peak allocated memory changed -1.03%.

Full10 point gate — passed.

## Refresh1 versus Cache10

9 of 9 seed pairs and 3 of 3 rooms are complete.

| Room | Candidate errors (mm) | Reference errors (mm) | Change (mm) | Error change (%) | Candidate / reference time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 11.810 / 11.967 | 11.818 / 11.723 | -0.008 / +0.244 | -0.07% / +2.08% | 19.70 / 19.26 |
| chess/seq-03 | 17.835 / 23.027 | 17.726 / 23.634 | +0.109 / -0.607 | +0.61% / -2.57% | 20.87 / 18.51 |
| fire/seq-03 | 15.487 / 18.131 | 15.685 / 18.468 | -0.198 / -0.337 | -1.26% / -1.82% | 20.88 / 18.39 |

heads/seq-01 used 688,882 evaluable points. Seed-only bounds were -0.80% to +2.32% for accuracy and +1.57% to +4.23% for completeness.

chess/seq-03 used 716,290 evaluable points. Seed-only bounds were -0.67% to +1.91% for accuracy and -3.19% to -0.50% for completeness.

fire/seq-03 used 697,344 evaluable points. Seed-only bounds were -1.90% to -0.89% for accuracy and -5.30% to -1.36% for completeness.

Equal-room error changes were -0.24% and -0.77%. The hierarchical bounds were -1.30% to +1.13% and -3.00% to +1.91%, respectively. Speedup was 0.913× and peak allocated memory changed +29.54%.

## Narrow versus Full10

9 of 9 seed pairs and 3 of 3 rooms are complete.

| Room | Candidate errors (mm) | Reference errors (mm) | Change (mm) | Error change (%) | Candidate / reference time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 11.565 / 11.716 | 12.886 / 13.904 | -1.321 / -2.188 | -10.25% / -15.73% | 18.33 / 27.19 |
| chess/seq-03 | 17.920 / 23.819 | 16.823 / 21.424 | +1.097 / +2.395 | +6.52% / +11.18% | 18.43 / 38.63 |
| fire/seq-03 | 15.603 / 18.409 | 15.505 / 18.449 | +0.097 / -0.040 | +0.63% / -0.22% | 18.26 / 178.40 |

heads/seq-01 used 688,882 evaluable points. Seed-only bounds were -11.19% to -8.91% for accuracy and -16.14% to -13.64% for completeness. Signed distance from the room's +1% / +3% limits was -1.449 / -2.605 mm (positive exceeds the limit).

chess/seq-03 used 716,290 evaluable points. Seed-only bounds were +3.95% to +7.07% for accuracy and +8.03% to +12.84% for completeness. Signed distance from the room's +1% / +3% limits was +0.929 / +1.752 mm (positive exceeds the limit).

fire/seq-03 used 697,344 evaluable points. Seed-only bounds were +0.19% to +0.72% for accuracy and -1.41% to +1.65% for completeness. Signed distance from the room's +1% / +3% limits was -0.058 / -0.594 mm (positive exceeds the limit).

Equal-room error changes were -1.03% and -1.59%. The hierarchical bounds were -7.08% to +4.77% and -10.96% to +8.09%, respectively. Speedup was 2.393× and peak allocated memory changed -21.77%.

Full10 point gate — passed.

## Narrow versus Cache10

9 of 9 seed pairs and 3 of 3 rooms are complete.

| Room | Candidate errors (mm) | Reference errors (mm) | Change (mm) | Error change (%) | Candidate / reference time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 11.565 / 11.716 | 11.818 / 11.723 | -0.253 / -0.007 | -2.14% / -0.06% | 18.33 / 19.26 |
| chess/seq-03 | 17.920 / 23.819 | 17.726 / 23.634 | +0.194 / +0.185 | +1.10% / +0.78% | 18.43 / 18.51 |
| fire/seq-03 | 15.603 / 18.409 | 15.685 / 18.468 | -0.083 / -0.059 | -0.53% / -0.32% | 18.26 / 18.39 |

heads/seq-01 used 688,882 evaluable points. Seed-only bounds were -2.26% to -1.31% for accuracy and -0.71% to +1.47% for completeness.

chess/seq-03 used 716,290 evaluable points. Seed-only bounds were +0.27% to +1.11% for accuracy and +0.68% to +2.59% for completeness.

fire/seq-03 used 697,344 evaluable points. Seed-only bounds were -1.12% to -0.15% for accuracy and -3.96% to +0.21% for completeness.

Equal-room error changes were -0.52% and +0.14%. The hierarchical bounds were -1.80% to +0.56% and -1.59% to +1.42%, respectively. Speedup was 1.020× and peak allocated memory changed +2.40%.

## Narrow versus Refresh1

9 of 9 seed pairs and 3 of 3 rooms are complete.

| Room | Candidate errors (mm) | Reference errors (mm) | Change (mm) | Error change (%) | Candidate / reference time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| heads/seq-01 | 11.565 / 11.716 | 11.810 / 11.967 | -0.245 / -0.251 | -2.07% / -2.10% | 18.33 / 19.70 |
| chess/seq-03 | 17.920 / 23.819 | 17.835 / 23.027 | +0.085 / +0.793 | +0.48% / +3.44% | 18.43 / 20.87 |
| fire/seq-03 | 15.603 / 18.409 | 15.487 / 18.131 | +0.116 / +0.278 | +0.75% / +1.53% | 18.26 / 20.88 |

heads/seq-01 used 688,882 evaluable points. Seed-only bounds were -3.55% to -1.46% for accuracy and -2.65% to -2.10% for completeness.

chess/seq-03 used 716,290 evaluable points. Seed-only bounds were -0.78% to +0.95% for accuracy and +2.54% to +4.00% for completeness.

fire/seq-03 used 697,344 evaluable points. Seed-only bounds were +0.75% to +0.85% for accuracy and +0.34% to +1.60% for completeness.

Equal-room error changes were -0.28% and +0.96%. The hierarchical bounds were -1.93% to +0.81% and -1.47% to +2.90%, respectively. Speedup was 1.116× and peak allocated memory changed -20.95%.

The selection rule retained Narrow.
