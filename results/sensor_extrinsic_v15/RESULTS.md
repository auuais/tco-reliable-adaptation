# Camera reference changes which reconstruction method ranks first

A sensor transform fitted on the other two development scenes missed the 5% improvement criterion on both primary references. Its dense reconstruction error was **18.3% higher** than the strongest comparison under the released reference and **19.9% higher** under the registered RGB reference. A third reference using published RGB structure-from-motion poses gave the same decision. The experiment scored 36 sealed point maps under three references, for 108 CPU evaluations. [All measurements and decisions](summary.json) · [Frozen plan](../../manifests/sensor_extrinsic_v15.json)

| Reconstruction | Released reference | Registered RGB reference | RGB SfM reference |
|---|---:|---:|---:|
| Frozen VGGT | 1.0000 | **1.0000** | 1.0000 |
| Direct supplied poses and intrinsics | 0.9615 | 1.2426 | 1.2032 |
| Change only the intrinsics | 1.0725 | 1.2238 | 1.1958 |
| Change only the camera frame | 0.9943 | 1.1801 | 1.1592 |
| Fixed RGB frame and intrinsics | 1.0973 | 1.1972 | 1.1789 |
| RGB frame fitted on other scenes | 1.0807 | 1.1989 | 1.1762 |
| Unchanged TCO | 0.9344 | 1.0091 | **0.9851** |
| Retain half of TCO's relative depth change | **0.9134** | 1.0229 | 1.0026 |

Each entry averages the accuracy and completeness errors divided by frozen VGGT's corresponding error on the same scene and reference. Lower is better. Three scenes receive equal weight. TCO and the half-depth comparison average three original adaptation seeds within each scene. All methods in a column share the same depth samples, validity masks and camera convention. The two alternate references retain registered depth while changing how it is placed in 3D. For RGB SfM we used the published per-view focal length and matched camera-baseline scale. [RGB reference source](https://github.com/tsattler/visloc_pseudo_gt_limitations/tree/22b7c48a0e55b49d0214a8dbb022426f518e2795)

The unchanged TCO benefit relative to frozen VGGT moves from **6.56% lower error** under the released reference to **0.91% higher error** under the registered RGB reference and **1.49% lower error** under RGB SfM. For these fifteen images, the choice of reference changes both the size of the observed effect and the first-ranked method. The released numbers reproduce the earlier camera controls within numerical rounding. The experiment covers three previously examined development sequences, and both alternate pose conventions have calibration uncertainty.

The image-only fit used a shared six-parameter correction to the approximate physical sensor transform, with robust epipolar residuals and a penalty on departures from that calibration. Each transform used RGB matches and supplied metric poses from the other two scenes. The target scene supplied clean camera priors and frozen VGGT depth to the reconstruction. The fitted deviations had magnitudes of 0.94° to 1.55° and 1.14 to 3.61 cm. The signed transform convention was checked on synthetic cameras. An implementation error in the initial empty results container was corrected before any point map or evaluation was written. The final protocol and predictions were committed before dense scoring. [Fits, source hashes and predictions](seal.json)

### Which geometry changed?

Against registered axial depth, the mean per-scene absolute relative error after a single scale alignment was 3.91% for frozen VGGT, 3.79% for unchanged TCO and 3.66% for half-depth. The direct pose substitutions retained frozen depth to a maximum absolute difference of 0.000000012 in this metric. Against published RGB SfM poses, mean scene-median relative rotation error was 1.69° for VGGT, 1.43° for TCO and 1.36° for the fixed RGB substitution. Corresponding translation-direction errors were 1.71°, 1.58° and 2.07°. An improvement on one camera component alone gave little indication of the final point-cloud ranking. These component measures were specified separately as descriptive diagnostics. [All depth and camera metrics](component_metrics.json) · [Component protocol](../../manifests/sensor_component_metrics.json)

The learned shared-transform candidate stops at this clean development screen. No MASt3R, held-out, verification or rental allocation was spent. [Disjoint-frame camera consistency](../sensor_windows_v16/RESULTS.md) · [Budget and release conditions](../../docs/BUDGET.md)
