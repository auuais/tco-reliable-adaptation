# Can agreement between predicted depth maps refine a frozen reconstruction?

The two-view depth check had **0.70% higher** registered surface error than the best local control (Depth, two views, all candidates visible) and **2.21% higher** error than the best historical control (Frozen VGGT, archived RTX 3090). It missed the prospective 5% development threshold. [Registered design](../../docs/VISIBILITY_DEPTH_EVIDENCE.md) · [Measured surface scores](scores.json)

Pixelwise source selection and consistency across depth maps are established multi-view stereo techniques. TCO already uses photometric or geometric compatibility while adapting a shared network. Our controlled comparison tests how a specific visibility rule in our local depth search changes which errors can be observed, and how results change when the supporting views use predictions from the same frozen model. [Schönberger et al., ECCV 2016](https://demuc.de/papers/schoenberger2016mvs.pdf) · [MVSNet, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper.pdf) · [TCO, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Learning_3D_Reconstruction_with_Priors_in_Test_Time_CVPR_2026_paper.pdf)

## Known depth errors with reference source geometry

These tests supply the reference camera and source depths as a positive control. A short target starts at exp(−0.08) times reference depth and a long target at exp(+0.08) times reference depth. Exact recovery returns to the reference depth. The clean false-edit rate includes all valid target pixels, including abstentions. All twelve conditions receive equal weight. The final two columns pool the eligible pixels.

| Cue and view rule | Short exactly recovered, all pixels | Long exactly recovered, all pixels | Clean false edit | Short exact, eligible | Long exact, eligible |
|---|---:|---:|---:|---:|---:|
| RGB, four fixed views, all candidates visible | 0.43% | 0.04% | 0.54% | 12.76% | 10.90% |
| RGB, four fixed views, visibility at starting depth | 0.43% | 0.23% | 0.95% | 10.92% | 14.42% |
| RGB, two supporting views, visibility at starting depth | 3.90% | 2.68% | 6.89% | 14.87% | 16.46% |
| Depth, two supporting views, all candidates visible | 23.27% | 2.80% | 0.58% | 92.99% | 76.53% |
| Depth, two supporting views, visibility at starting depth | 22.86% | 14.58% | 0.81% | 87.28% | 89.48% |
| Depth, first supporting view, visibility at starting depth | 47.43% | 37.32% | 4.12% | 91.14% | 91.73% |

The primary rule had two eligible source views for 26.21% of short-start pixels and 16.29% of long-start pixels. Exact recovery conditional on support should be read alongside the smaller absolute recovery fractions. The original four-view RGB control reproduces every published histogram. [Full 12-way factorial, including every scene](positive_control.json)

## Frozen model inputs and dense surface evaluation

Here the model supplies the camera poses and source depths. All methods share the original local FP32 denominator and twelve development conditions. Smaller distance scores are better. Registered accuracy and completeness are nearest-neighbor distances in millimeters after the published alignment. Normal consistency is higher when the two surfaces agree.

| Method | Registered score | Accuracy (mm) | Completeness (mm) | Normal consistency | Released score |
|---|---:|---:|---:|---:|---:|
| Frozen VGGT, local FP32 | 1.0000 | 17.25 | 17.27 | 0.7608 | 1.0000 |
| Direct focal substitution | 0.9850 | 17.06 | 16.75 | 0.7587 | 0.9828 |
| One depth scale | 0.9860 | 17.09 | 16.76 | 0.7577 | 0.9850 |
| Guarded per-view scales | 0.9914 | 17.12 | 16.98 | 0.7591 | 0.9939 |
| RGB, two sources | 1.0334 | 18.74 | 16.52 | 0.7546 | 0.9964 |
| RGB, four sources | 1.0030 | 17.57 | 16.65 | 0.7568 | 0.9778 |
| Previously tested four-view RGB check | 0.9901 | 17.25 | 16.68 | 0.7574 | 0.9753 |
| Depth, one supporting view | 1.0240 | 18.06 | 16.79 | 0.7445 | 0.9860 |
| RGB, two supporting views | 1.0820 | 20.24 | 16.23 | 0.7464 | 1.0098 |
| Depth, fixed four views | 0.9867 | 17.09 | 16.76 | 0.7581 | 0.9832 |
| Depth, two views, all candidates visible | 0.9838 | 17.07 | 16.70 | 0.7572 | 0.9798 |
| Depth, two supporting views | 0.9906 | 17.14 | 16.78 | 0.7556 | 0.9822 |
| Frozen VGGT, archived RTX 3090 | 0.9692 | 17.43 | 16.01 | 0.6850 | 0.9597 |
| TCO, retain | 1.0766 | 19.75 | 17.60 | 0.7616 | 1.0241 |
| TCO, consensus | 1.0546 | 19.36 | 17.12 | 0.7624 | 1.0029 |
| TCO, adaptive | 1.0081 | 18.26 | 16.47 | 0.7637 | 0.9665 |

| Development scene | Guarded score | Difference from the strongest aggregate local control | Primary rule eligible, full raster |
|---|---:|---:|---:|
| chess/seq-03 | 0.9778 | 0.18% lower | 52.53% |
| fire/seq-03 | 1.0368 | 2.10% higher | 50.84% |
| heads/seq-01 | 0.9573 | 0.10% higher | 7.52% |

## Do the local updates point in the right direction?

These comparisons use the same fixed alignment as the direct-focal predictions. The oracle chooses among the same five depth values using the corresponding reference pixel. It has no access restriction and reproduces the previous depth-candidate diagnostic. The final column excludes the eight-pixel neighborhoods of sparse fitting matches. Positive values indicate reduced squared error.

| Method | Changed valid pixels | Improved among changed | Registered RMSE (mm) | Squared-error reduction outside sparse support |
|---|---:|---:|---:|---:|
| Depth, two supporting views | 2.00% | 43.39% | 148.21 | 0.05% |
| Depth, one supporting view | 14.60% | 46.45% | 148.49 | -0.48% |
| RGB, two supporting views | 9.39% | 22.44% | 150.69 | -2.22% |
| Depth, fixed four views | 0.37% | 40.58% | 148.27 | 0.00% |
| Depth, two views, all candidates visible | 1.08% | 53.07% | 148.14 | 0.14% |
| Reference oracle | 65.74% | 100.00% | 114.40 | 39.88% |

The registered checks gave the following outcomes. Local gain failed, historical gain failed, component limits passed, scene criterion failed, released-depth check failed. The nine RGB batches come from three previously examined 7-Scenes sequences. Relative rankings in this development study cannot establish performance on other scenes, datasets or backbones. [Condition-level pixel evidence](pixel_check.json)

## What changes when predicted sources replace reference depth?

Each row uses the same four target-depth states and the same starting-depth visibility and two-view check. The first two numerical columns use the intersection of pixels eligible under all four camera/source combinations. The natural column measures whether a change moves scaled model axial depth closer to the registered reference. One reference-derived scale per batch converts arbitrary model units to meters. These are diagnostic interventions.

| Camera rig | Source depths | Short exact on common pixels | Long exact on common pixels | Natural improved among edits on common pixels | Natural log-depth MSE reduction, all valid pixels |
|---|---|---:|---:|---:|---:|
| Reference | Reference | 89.03% | 90.37% | 81.33% | 0.61% |
| Reference | Frozen | 78.36% | 54.04% | 26.28% | -0.33% |
| Frozen | Reference | 58.67% | 65.84% | 39.28% | -0.04% |
| Frozen | Frozen | 78.05% | 73.56% | 31.08% | -0.12% |

On the common eligible pixels of natural targets, the share of changes that helped was 81.3% with the reference rig and source depths. Replacing only source depths with frozen predictions reduced that share to 26.3%. Replacing only the camera rig gave 39.3%. With both predicted inputs it was 31.1%. Each rate conditions on the pixels the corresponding rule actually changes. The eligible region is shared, but those changed subsets can differ. The common region contains 22.40% of valid natural-target pixels and only 5.98% of the squared log-depth error that the five depth candidates could remove with reference access, averaging condition-level fractions.

The common intersection is recomputed for each target starting condition, since it affects projection and visibility. Percentages conditional on eligibility or a change pool pixels across the twelve conditions. Mean squared log-depth error gives conditions equal weight and uses a different gauge and metric from the surface table. [Source-replacement protocol](../../docs/DEPTH_SOURCE_TRANSPLANT.md) · [Counts, MSE and scene results](source_transplant.json)

## Can the depth candidates produce a better surface?

The following three point maps receive the registered reference at decision time. All candidate choices minimize corresponding-pixel squared error at fixed alignment, then the original surface evaluator realigns and scores each constructed map. The same map is used for both depth references.

| Reference-assisted selection | Registered surface score | Released surface score |
|---|---:|---:|
| Any candidate at any valid pixel | 0.8134 | 0.7357 |
| Any candidate where the first source proposes a change | 0.9773 | 0.9568 |
| Accept or reject the first source’s actual proposal | 0.9820 | 0.9732 |

Only the inference results in the first table of surface scores govern spending. A reference-assisted surface result witnesses what that particular selection attained. It supplies no global optimum for the nearest-neighbor metric. [Surface diagnostic design](../../docs/DEPTH_GRID_SURFACE_DIAGNOSTIC.md) · [Scores and per-scene tables](grid_surface.json)

## Does a correction transfer beyond the observed pixels?

Fit one bounded depth scale per tile on alternating 16×16-pixel training blocks and evaluate it on the other block color, swapping roles for the second pass. The evaluation pixels never received a first-source proposal. All four fitting conditions share these exact test pixels. For the random control, the number of training labels in each tile matches the observed source-proposal sites. Positive values indicate lower corresponding-pixel squared error on the registered reference.

| Tiles per image | Reference at all training pixels | Reference at proposal sites | Reference at matched-count random sites | Actual source proposals |
|---|---:|---:|---:|---:|
| 1×1 | 1.99% | -34.08% | 1.95% | -1.58% |
| 2×2 | 8.31% | -13.05% | 7.83% | -2.97% |
| 4×4 | 13.30% | -10.60% | 8.07% | -10.38% |
| 8×8 | 20.39% | -0.30% | 11.61% | -10.19% |

The test covers 85.40% of valid pixels. At the registered 4×4 resolution, exact reference labels at proposal sites produced 10.60% higher error than identity on the withheld pixels. The same number of random reference labels per tile gave 8.07% lower error. The locations where the first source asks for a correction therefore transfer poorly to the rest of the tile on these development sequences. A subsequent local test should distinguish bias from source visibility from bias introduced by using only nonzero proposals. [Cross-fit protocol](../../docs/SPATIAL_DEPTH_TRANSFER.md) · [Both references and scene breakdowns](spatial_transfer.json)

## Runtime and replay

Calculating the full factorial and constructing the primary point map took 8.01 seconds per condition on average on the local CPU after the original frozen pass. This includes the unused comparison arms and is not a standalone deployment latency measurement. Cloud cost was **$0**. Both physical drives contain 45 unique prediction files (495,983,815 bytes per drive). [Prediction checksums](archive.json) · [Dense corrections](choices.zip) · [Per-view counts and timing](predictions.json)

With the project CPU dependencies, the following commands check the numerical controls, choices and reported metrics. Full replay uses the public data and prepared inputs from the previous study. In a scratch checkout with an empty output folder for this experiment, run `visibility_positive_control.py evaluate`, `run_visibility_model.py infer`, `score_visibility_model.py evaluate` and `score_visibility_model.py pixels` from the scripts directory or using the scripts path from the repository root. [Parent experiment and data](../dense_rgb_depth/RESULTS.md)

```sh
python -m unittest discover -s scripts -p test_visibility_depth_evidence.py
python scripts/visibility_positive_control.py verify
python scripts/run_visibility_model.py verify
python scripts/score_visibility_model.py verify
python scripts/depth_source_transplant.py verify
python scripts/depth_grid_surface.py verify
python scripts/spatial_depth_transfer.py verify
```
