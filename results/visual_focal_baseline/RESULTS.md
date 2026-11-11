# Classical visual self-calibration on the same 33 image batches

We fitted image-pair fundamental matrices and ran the iterative focal solver of Kocur, Kyselica and Kúkelová. Our five-view implementation aggregates successful positive estimates from at least two neighbors. It preserves the supplied poses and principal points and uses supplied focal lengths as the solver priors. We evaluated both continuous estimates and a version that selects the closest of the two allowable focal candidates for the one outlier. [CVPR 2024 authors' implementation](https://github.com/kocurvik/robust_self_calibration) · [Frozen benchmark settings](../../manifests/visual_focal_baseline.json)

| Method, primary 120 cases | Relative focal-gradient error | Mean absolute log-focal error across all five cameras |
|---|---:|---:|
| Keep metadata | 0.2343 | 0.01133 |
| Always use the neighbor median | 0.2126 | 0.01133 |
| VGGT, closer candidate | 0.1140 | 0.00264 |
| Both RGB folds with supplied poses | 0.0704 | 0.00491 |
| Iterative solver, closer candidate | 0.1962 | 0.01058 |
| Iterative solver, continuous | 0.7266 | 0.07430 |

The two-candidate solver corrected 16/60 faults and changed 12/60 legitimate zooms. Continuous estimation also changed cameras away from the deliberately applied outlier. On completely clean configurations its all-camera log-focal error was 0.04937, versus zero for the methods that retained the supplied calibration. The comparison applies to our sparse indoor data and chosen multi-neighbor implementation. The CVPR paper evaluates other settings.

### Sensitivity to the frequency of faulty metadata

Within the primary mild-pose condition, the two-fold rule's conditional mean focal-gradient errors were 0.03388 for valid outliers and 0.10692 for faulty outliers. Holding the scene and pose mix fixed, it matches or beats both unconditional policies if approximately **8.6% to 78.5%** of visible outliers are faults. It matches or beats all seven evaluated policies over **8.6% to 73.8%**. The one-fold comparator ties it in focal-gradient error here. These are intersections of observed conditional risks. The field frequency of focal faults is unknown. The original false-edit criterion still fails. [All risks by noise level](prevalence_sensitivity.json)

The pinned PoseLib solver was rerun locally on 33 public sets of derived match coordinates. The verifier checked 93 calibration configurations and 930 gradient arrays against the original 465 paired cases. No further GPU rental was required. [Commands and source data instructions](../../docs/REPLAY_FOCAL_ZOOM.md) · [Machine-readable outputs](summary.json)
