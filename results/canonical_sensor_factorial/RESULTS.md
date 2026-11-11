# Intrinsics helped on average. Correcting both camera components missed the target

The strongest of the 11 tested methods on both dense references used registered RGB intrinsics with the released poses. Against TCO with its original supplied cameras, this change reduced mean normalized reconstruction error by **6.38%** on the released reference and **12.57%** on the registered RGB reference. Fire was an exception. On that scene the intrinsics change raised released-reference error by **7.43%**. The three development sequences share one dataset and each condition uses one seed. [Full 66-row score table and scene contrasts](summary.json)

The prespecified candidate corrected both the camera frame and the intrinsics. Its error exceeded the intrinsics-only comparator by **5.86%** and **0.88%**, respectively. Its largest scene-level loss against that comparator was 18.54% on the released reference and 7.11% on the registered reference. The advancement rule required at least 5% lower mean error and at most a 1% loss on any scene, on both references. The combined correction failed that rule. The $25 MASt3R allocation remains reserved. [Frozen experiment](../../manifests/canonical_sensor_factorial.json)

| Camera priors or comparator | Released reference | Registered RGB reference |
|---|---:|---:|
| Frozen VGGT, zero updates | 1.0000 | 1.0000 |
| Original TCO with released priors | 0.8745 | 0.9288 |
| Change pose frame only | 0.8863 | 0.8891 |
| Change intrinsics only | **0.8187** | **0.8120** |
| Change pose frame and intrinsics | 0.8666 | 0.8192 |
| Directly use released cameras with initial depth | 1.1066 | 1.3425 |
| Directly use registered cameras with initial depth | 1.1685 | 1.2627 |

Lower is better. Each score first divides accuracy and completeness errors by their corresponding frozen-model errors on the same scene and reference, then averages those two ratios and gives the three scenes equal weight. The four additional depth-blending comparators and absolute errors in metres are included in the full table. Both dense references use the same depth observations. [Scoring definition](../../scripts/evaluate_sensor_tco.py)

### How much depended on the scene?

| Sequence | Intrinsics-only gain vs original TCO, released | Intrinsics-only gain vs original TCO, registered |
|---|---:|---:|
| Heads 01 | 7.74% | 16.12% |
| Chess 03 | 19.66% | 21.60% |
| Fire 03 | −7.43% | 1.60% |

Chess showed the largest interaction. Adding the pose-frame correction to the original TCO priors reduced released-reference error from 0.8109 to 0.7525. Adding it after correcting intrinsics raised error from 0.6515 to 0.7723. A camera transformation can therefore have different effects depending on the intrinsics already supplied. This interaction was observed on a single five-frame Chess window. [Scene tables and factorial effects](summary.json)

### What changed in the estimated depth and cameras?

| Method | Axial depth AbsRel | Median pair rotation error (°) | Median translation-direction error (°) | Mean absolute log focal error |
|---|---:|---:|---:|---:|
| Frozen VGGT | 0.0431 | 1.516 | 2.221 | 0.0195 |
| Original TCO | 0.0372 | 1.633 | 2.158 | 0.0534 |
| Pose frame only | 0.0362 | **1.152** | 2.442 | 0.0527 |
| Intrinsics only | **0.0326** | 2.114 | 2.177 | 0.0076 |
| Both corrections | 0.0328 | 1.346 | **2.037** | **0.0050** |

The camera diagnostics compare recovered camera parameters with separate public RGB structure-from-motion estimates. Pair metrics average the per-scene medians over ten view pairs. Axial depth uses one median scale per five-frame window. Intrinsics-only TCO improved depth and focal estimates relative to original TCO, while its rotation estimate worsened. The cell with the lowest reconstruction error therefore offers different tradeoffs for a camera-tracking application. [All 33 component evaluations](component_metrics.json)

### Execution and scope

All 12 fresh-process jobs finished 40 updates with finite gradients on the same 24 GiB RTX 3090, using the original model path. Model inference and adaptation took 52.31–55.01 seconds per cell. Peak allocated CUDA memory was 11.50 GiB. Within a scene, the four runs shared initial predictions, adapter weights, random state and reference tensors. The 33 prediction fingerprints were public before CPU scoring began. Each raw trajectory was verified on two drives before the provider stopped the GPU. The calibration-stage cost estimate was $0.95–$0.99 against $3. [Prediction seal](seal.json) · [Session and raw checksums](session.json) · [Replay instructions](REPLAY.md)

The intrinsics-only result is a development lead. Repeating it across seeds and testing whether RGB observations can recognize an incorrect focal prior are separate questions. Neither transfer nor held-out generalization has been measured here.
