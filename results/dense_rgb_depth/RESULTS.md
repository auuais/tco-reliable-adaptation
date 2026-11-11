# Dense RGB depth correction with an independent source-view check

The guarded method had **0.51% higher** registered 3D error than the best local control (Direct focal substitution) and **2.15% higher** error than the best historical control (Frozen VGGT, archived RTX 3090). It missed the registered development gate. [Prospective design](../../docs/DENSE_RGB_DEPTH.md) · [Raw surface scores and decision](scores.json)

## Surface reconstruction

Each condition has equal weight. The primary score averages the ratios of accuracy and completeness to the local frozen FP32 model. Smaller values indicate smaller distances. Normal consistency is higher when normals agree. The historical TCO arms average the original two seeds. These are three development sequences, nine distinct RGB batches and twelve controlled cases. The two signs of a metadata fault share the same RGB images.

| Method | Registered score | Accuracy (mm) | Completeness (mm) | Normal consistency | Released score |
|---|---:|---:|---:|---:|---:|
| Frozen VGGT, local FP32 | 1.0000 | 17.25 | 17.27 | 0.7608 | 1.0000 |
| Direct focal substitution | 0.9850 | 17.06 | 16.75 | 0.7587 | 0.9828 |
| One depth scale | 0.9860 | 17.09 | 16.76 | 0.7577 | 0.9850 |
| Guarded per-view scales | 0.9914 | 17.12 | 16.98 | 0.7591 | 0.9939 |
| RGB, two sources | 1.0334 | 18.74 | 16.52 | 0.7546 | 0.9964 |
| RGB, four sources | 1.0030 | 17.57 | 16.65 | 0.7568 | 0.9778 |
| RGB, two sources with independent validation | 0.9901 | 17.25 | 16.68 | 0.7574 | 0.9753 |
| Frozen VGGT, archived RTX 3090 | 0.9692 | 17.43 | 16.01 | 0.6850 | 0.9597 |
| TCO, retain | 1.0766 | 19.75 | 17.60 | 0.7616 | 1.0241 |
| TCO, consensus | 1.0546 | 19.36 | 17.12 | 0.7624 | 1.0029 |
| TCO, adaptive | 1.0081 | 18.26 | 16.47 | 0.7637 | 0.9665 |

| Scene | Guarded score | Difference from best local control | All four source views support a patch |
|---|---:|---:|---:|
| chess/seq-03 | 0.9815 | 0.02% higher | 0.15% |
| fire/seq-03 | 1.0326 | 1.46% higher | 10.85% |
| heads/seq-01 | 0.9561 | 0.00% lower | 0.00% |

## Do the changes move pixels closer to the reference?

All methods use the baseline global similarity for this diagnostic. An oracle chooses the best of the same five candidate depths directly from the reference at each pixel. It measures the available range of corrections. The inference methods select depths using only RGB, frozen depth and frozen cameras.

| Method | Changed valid pixels | Improved among changed | Mean RMSE (mm) | Squared-error reduction outside sparse support |
|---|---:|---:|---:|---:|
| RGB, two sources | 5.91% | 24.66% | 149.52 | -1.02% |
| RGB, four sources | 2.23% | 22.65% | 148.68 | -0.36% |
| RGB, two sources with independent validation | 0.88% | 23.38% | 148.44 | -0.14% |
| Reference oracle | 65.74% | 100.00% | 114.40 | 39.88% |

Selecting the best available depth using reference geometry lowered mean corresponding-pixel RMSE from 148.28 to 114.40 mm. These values concern the corresponding-pixel metric and do not give a ceiling on nearest-neighbor surface error. Sparse support means within eight pixels of a fitting-fold RGB match. The proportions of changed pixels are averaged by condition. The fraction improved among changes pools the changed pixels. [Pixel measurements for both references](pixel_check.json)

## Positive control with known cameras and axial depth

Only this diagnostic uses reference cameras and source depths. The target starts at its true depth multiplied by exp(−0.08), 1 or exp(+0.08). Here “true” denotes the reference depth used to inject the error. RGB-D noise, resampling, registration error and changing appearance can still affect agreement between images. The clean column counts false changes over all valid pixels. The other fractions distinguish the area with test support from exact recovery on that area.

| Rule | Short start eligible | Short exactly recovered among eligible | Clean falsely changed | Long start eligible | Long exactly recovered among eligible |
|---|---:|---:|---:|---:|---:|
| Nearest source | 29.82% | 32.41% | 17.07% | 2.23% | 29.37% |
| RGB, two sources | 7.84% | 31.19% | 3.68% | 0.59% | 20.45% |
| RGB, four sources | 3.39% | 30.70% | 1.40% | 0.35% | 27.39% |
| RGB, two sources with independent validation | 3.39% | 12.76% | 0.54% | 0.35% | 10.90% |

Eligible fractions average the twelve conditions. Exact recovery among eligible pixels pools their counts. The original rule requires all five hypotheses to pass the source-depth visibility check. For parallel cameras viewing a front-facing plane at depth z, a long initial depth of exp(0.08)z has an extreme candidate of exp(0.16)z, about 1.174z. That candidate exceeds the 1.10z visibility limit and can exclude the pixel even when the correct depth is one of the other four candidates. The clean result also exposes a selection problem among eligible pixels. The guarded rule changes 30.4% of them even at the reference depth. The next local comparison will separate this photometric ambiguity from the visibility restriction by testing source-depth agreement and visibility anchored at the starting depth. [Positive-control protocol](../../docs/PHOTOMETRIC_POSITIVE_CONTROL.md) · [Counts by scene, start and rule](positive_control.json)

## Runtime, evidence and reproduction

The local CPU took 5.54 seconds per five-view condition for the RGB scoring, rule selection and guarded point construction, using cached frozen heads. This timing excludes model inference, file I/O and reference evaluation. Historical TCO ran on another GPU. The new experiment added **$0** to the RunPod bill. Two physical drives hold verified copies of all 24 distinct point maps (264,682,424 bytes per copy). [Archive checksums](archive.json) · [Per-pixel choices](choices.zip) · [Input and output hashes](predictions.json)

Run the following from the repository root with the CPU validation dependencies. The public checks replay the discrete predictions, metric tables and pixel counts. Full image-based reproduction also needs the 7-Scenes inputs, frozen heads and offline reference maps described in the parent experiment. In a scratch checkout with an empty output folder for this study, run the inference script first, followed by each evaluator in evaluate mode. [Parent inputs and evaluation](../fixed_camera_depth_scale/RESULTS.md)

```sh
python -m unittest discover -s scripts -p test_dense_rgb_depth.py
python scripts/score_dense_rgb_depth.py verify
python scripts/dense_rgb_pixel_check.py verify
python scripts/photometric_positive_control.py verify
```
