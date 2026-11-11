# What happens when the four source cameras move closer to each target?

The primary nearby-frame update produced **9.63% higher surface error than the strongest local control and 11.04% higher error than the historical native frozen VGGT**. All five advancement checks failed. The three-scene development comparison provides no basis for spending the $25 MASt3R allocation. [Sealed surface results](temporal_scores.json)

## Source coverage with measured geometry

The targets were frames 25, 225, 425, 625 and 825 of Heads/01, Chess/03 and Fire/03 from 7-Scenes. All 15 target RGB arrays, reference point maps and validity masks reproduced the preceding experiment exactly. Each comparison supplied four source images per target. In the dispersed condition the 15 targets also served as sources. Each temporal condition used 60 additional distinct source images. All four spacing-control measurements below use the same subset of valid target pixels that lacked support from the first source in the original experiment. Each target has equal weight.

| Four-source policy | Mean target-to-source baseline | Eligible with two sources at natural target depth | Recovery of injected short depth | Recovery of injected long depth | Reduction in natural squared log-depth error |
|---|---:|---:|---:|---:|---:|
| Original dispersed | 0.984 m | 0.37% | 1.06% | 0.04% | 0.20% |
| ±4, ±12 frames | 0.082 m | 86.13% | 88.01% | 85.80% | 19.84% |
| ±8, ±24 frames | 0.161 m | 78.44% | 80.36% | 74.54% | 17.17% |

The source cameras and depths in this table came from the dataset. Injected errors multiply the target depth by exp(−0.08) or exp(+0.08). Recovery means the estimated correction exactly reversed that change. On clean targets, the shorter-spacing rule changed 0.0084% of previously unsupported pixels. The dispersed comparison reproduced the full earlier reference-camera/reference-depth results. [Prospective reference test](../../docs/SOURCE_FRAME_SPACING.md) · [Source hashes and camera checks](inputs.json) · [All scenes, target states and support partitions](reference_control.json)

## Runtime and controls for a deployable update

The RTX 3080 ran 33 five-image frozen VGGT forwards. All nine dispersed depth, intrinsics and pose arrays matched the preceding RTX 3080 captures bit for bit. The three dispersed forwards took 12.99 seconds. The complete short-spacing configuration used 18 forwards and took 67.66 seconds of measured forward time. The medium-spacing configuration, including the same three original forwards, took 66.51 seconds. Peak allocated GPU memory across the 33 forwards was 4.34 GiB. The short-spacing primary depth decision used another 15.71 seconds of measured CPU time. These component timings exclude data preparation, output archiving and surface evaluation. This frozen camera/depth inference omits the unused point head. The measured memory does not estimate the requirements of full TCO training. [Frozen captures](frozen_heads.json) · [Prediction manifest and per-target timings](temporal_predictions.json)

All deployed arms saw RGB, the same supplied focal calibration when applicable and frozen outputs. Each temporal bundle shared one original target. Its depth units were aligned using the median ratio of dispersed and temporal predicted depths at that target. The original target poses anchored the scene. The simple controls include unmodified registered temporal depth with its native focal, that same depth with the supplied focal, and a geometric-mean blend of the original and temporal predicted depths. Single-source and two-source updates also used exactly those temporal images. The dataset's depths and poses entered only the separate reference controls and scoring. [Prospective model comparison](../../docs/TEMPORAL_DEPTH_REDESIGN.md)

## Scoring and reproducibility

### Do the source depths or camera poses explain the loss?

The four arms below use exactly the same nearby RGB frame IDs and identical validity masks. They substitute measured or predicted camera geometry and depth independently. The last column is restricted to pixels where all four arms had two eligible sources. Its denominator is the unchanged target's log-depth error on those pixels. Positive values mean lower error. This diagnostic uses an offline metric scale to compare predicted depths with sensor depths.

| Camera rig | Source depth | Two-source coverage, natural target | Short error exactly corrected, all valid | Long error exactly corrected, all valid | Clean pixels changed | Natural error reduction, all valid | Natural error reduction, common support |
|---|---|---:|---:|---:|---:|---:|---:|
| Measured | Measured | 90.64% | 91.02% | 89.99% | 0.04% | +15.89% | +23.00% |
| Measured | Frozen VGGT | 89.51% | 60.97% | 38.53% | 34.76% | −2.69% | −3.54% |
| Frozen VGGT | Measured | 90.64% | 90.14% | 88.66% | 0.40% | +15.15% | +21.82% |
| Frozen VGGT | Frozen VGGT | 89.52% | 60.66% | 37.69% | 37.59% | −3.22% | −4.05% |

With frozen cameras, substituting measured source depths improved the natural common-support result by **25.87 percentage points**. With frozen source depths, substituting measured cameras improved it by **0.52 points**. With measured cameras the source-depth substitution improved it by 26.54 points. With measured source depths the camera substitution improved it by 1.18 points. On Heads, Chess and Fire respectively, frozen cameras with measured depths reduced common-support error by 37.49%, 9.55% and 18.40%. Using frozen depths as well increased it by 7.03%, 4.67% and 0.45%. The two-source rule frequently changed already-correct depths when its source depths came from the same model. This identifies a limitation of this update on these sequences. The experiment does not isolate a unique physical cause of the predicted-depth errors. [Registered diagnostic](../../docs/NEARBY_CAMERA_DEPTH_CONTROL.md) · [Per-target counts, camera substitutions and replay](nearby_camera_depth.json)

### Surface and pixel evaluation

| Method | Short spacing, registered | Short spacing, released | Medium spacing, registered | Medium spacing, released |
|---|---:|---:|---:|---:|
| Local dispersed frozen FP32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Dispersed, supplied focal | 0.9794 | 0.9843 | 0.9794 | 0.9843 |
| Best previous local depth control | 0.9783 | 0.9818 | 0.9783 | 0.9818 |
| Historical native frozen RTX 3090 | 0.9658 | 0.9578 | 0.9658 | 0.9578 |
| Unmodified temporal shape, native focal | 1.1900 | 1.1054 | 1.1487 | 1.0700 |
| Unmodified temporal shape, supplied focal | 1.1995 | 1.1418 | 1.1362 | 1.0895 |
| Geometric-mean depth blend | 1.0342 | 1.0312 | 1.0222 | 1.0108 |
| Single source, original starting depth | 1.1292 | 1.0841 | 1.1327 | 1.0673 |
| Two sources, temporal starting depth | 1.1784 | 1.1232 | 1.1211 | 1.0759 |
| **Two sources, original starting depth (primary)** | **1.0724** | **1.0463** | **1.0613** | **1.0176** |

Against the best local control, the primary registered score worsened by 2.13% on Heads, 17.47% on Chess and 9.06% on Fire. The geometric-mean blend also beat it on both references with the same images. The medium-spacing sensitivity result worsened registered surface error by 8.49% against the strongest local control. The full comparison includes the three historical TCO strategies, averaging the preceding runs over both random seeds and both base metadata-fault directions within each scene. [All methods, accuracy, completeness and scene breakdowns](temporal_scores.json)

The surface evaluator aligns each five-target scene by similarity and ICP and measures both point-cloud accuracy and completeness. The score divides each distance by the archived local frozen FP32 distance, averages the two ratios and gives each scene equal weight. A smaller score is better. A separate pixelwise comparison fixes the original direct-focal similarity for every arm and measures squared 3D displacement at corresponding pixels. These answer different questions. Across all valid pixels, the short-spacing two-source update increased fixed-alignment error by 0.9% on the registered reference. The geometric-mean blend reduced it by 2.4%. [Pixelwise results on both depth conventions](temporal_pixels.json)

The RGB tensors, reference geometry, model heads and 36 predicted point maps were copied to two physical disks and checked by SHA-256. The public repository contains compact decisions, manifests and scored evidence. RunPod spending for this study was $0. The calibration-stage conservative estimate remains $1.82 of its $3 cap. [Asset inventory](asset_archive.json) · [Prediction inventory](archive.json) · [Budget ledger](../../docs/BUDGET.md)

## Could independently measured RGB motion replace predicted source depth?

The next local control used 60 image pairs and 120 forward/reverse DIS flow estimates. Matching used the OpenCV 4.11.0 medium preset and took 4.51 CPU wall-clock seconds inside the matching calls. Triangulation used a one-pixel flow-cycle tolerance, a one-degree minimum ray angle and a one-pixel reprojection tolerance. The two-sided rule also required depth agreement between earlier and later source frames. Corrected depth was limited to ±0.08 in log units. No source depth entered triangulation. The evaluation used original natural targets plus known short, long and clean targets. [Registered experiment](../../docs/NEARBY_RGB_CORRESPONDENCE.md) · [Kroeger et al., Dense Inverse Search](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Kroeger2016ARXIV.pdf) · [OpenCV implementation](https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html)

| Camera rig and flow rule | Valid target coverage | Natural squared log error reduction | Edited natural depths improved | Injected short error removed | Injected long error removed | Mean absolute log error introduced on clean depth |
|---|---:|---:|---:|---:|---:|---:|
| Frozen, any supported source | 67.17% | −20.96% | 26.02% | 11.77% | 52.74% | 0.0321 |
| Frozen, two-sided support (primary) | 24.63% | −3.84% | 32.88% | 10.68% | 22.65% | 0.0098 |
| Measured, any supported source | 60.50% | −19.38% | 31.14% | 30.40% | 28.66% | 0.0279 |
| Measured, two-sided support | 17.93% | +0.55% | 41.83% | 13.18% | 15.24% | 0.0054 |

With frozen cameras, two-sided triangulation increased natural error by 7.89% on Heads, 2.32% on Chess and 1.30% on Fire. The known perturbations were easier to correct than the original model's errors. Ground-truth cameras yielded a 0.55% aggregate reduction on natural targets for the two-sided rule. The registered decision to proceed to a whole-surface redesign failed. These observations concern this motion estimator, support rule and source spacing on three previously examined scenes. [Per-target flow outcomes](nearby_motion_scores.json) · [Flow capture](nearby_rgb_flows.json) · [Sealed depth estimates](nearby_motion_estimates.json) · [Two-disk inventory](motion_archive.json)

### Did motion point in the wrong direction, or was the step too large?

Multiplying the bounded log-depth correction by a single scalar reduced the average error for small positive steps. At each target, the sum of squared log-depth errors is a quadratic in that scalar. Its coefficients reproduced both measured endpoints. The table gives the reduction in natural squared log-depth error against the unchanged dispersed prediction. The last column lets each target choose its own scalar using its measured depth and gives an upper bound for this class of updates.

| Rig and RGB rule | Pooled optimal step | Pooled optimal gain | Gain using step from other two scenes | Per-target oracle gain |
|---|---:|---:|---:|---:|
| Frozen, any supported source | 0.213 | 1.65% | 1.49% | 3.13% |
| Frozen, two-sided support | 0.296 | 0.82% | 0.79% | 1.42% |
| Measured, any supported source | 0.247 | 2.34% | 2.20% | 4.83% |
| Measured, two-sided support | 0.553 | 1.58% | 1.42% | 2.42% |

For two-sided frozen geometry, the step learned from the other two scenes reduced error by 1.29% on Heads, 0.17% on Chess and 0.92% on Fire. There is a weak transferable direction in these development frames, and the unit step gives it too much weight. The simpler any-source rule retained more of this signal. These post-hoc depth gains do not establish a surface improvement. A single multiplier per target has little remaining headroom on this metric. [Quadratic diagnostic design](../../docs/MOTION_DEPTH_SHRINKAGE.md) · [Per-target coefficients, leave-one-scene-out fits and endpoint checks](motion_shrinkage.json)
