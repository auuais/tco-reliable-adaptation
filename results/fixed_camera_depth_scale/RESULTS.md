# Can five depth scales improve frozen VGGT reconstruction?

The guarded method produced **0.65% higher registered 3D error** than the strongest local control, selected focal, unchanged depth. Its error was **2.29% higher** than the strongest historical control (Archived frozen VGGT, native). It failed the preregistered development gate. [Registered protocol](../../docs/DEPTH_SCALE_PROTOCOL.md) · [Full measurements](scores.json)

## Geometry on the two depth references

Lower scores and distances are better. The score averages accuracy and completeness relative to frozen local FP32 for each condition and then averages the twelve conditions equally. The distances and normal consistency below use the RGB-registered depth reference. Higher normal consistency is better.

| Method | Registered score | Accuracy (mm) | Completeness (mm) | Normal consistency | Released score |
|---|---:|---:|---:|---:|---:|
| Frozen VGGT, FP32 | 1.0000 | 17.25 | 17.27 | 0.7608 | 1.0000 |
| Selected focal, unchanged depth | 0.9850 | 17.06 | 16.75 | 0.7587 | 0.9828 |
| One shared depth scale | 0.9860 | 17.09 | 16.76 | 0.7577 | 0.9850 |
| Five depth scales | 0.9903 | 17.11 | 16.95 | 0.7588 | 0.9934 |
| Five scales with validation | 0.9914 | 17.12 | 16.98 | 0.7591 | 0.9939 |
| Archived frozen VGGT, native | 0.9692 | 17.43 | 16.01 | 0.6850 | 0.9597 |
| Archived TCO, retain | 1.0766 | 19.75 | 17.60 | 0.7616 | 1.0241 |
| Archived TCO, consensus | 1.0546 | 19.36 | 17.12 | 0.7624 | 1.0029 |
| Archived TCO, adaptive | 1.0081 | 18.26 | 16.47 | 0.7637 | 0.9665 |

The historical frozen arm uses the native mixed-precision RTX 3090 coordinates. Each historical TCO arm averages two independent 40-update trajectories per condition. The local arms share frozen RTX 3080 depth and pose heads. Their corrected coordinates are built in float64 and saved in float32. The frozen control uses the original float32 unprojection. Hardware and arithmetic differences matter for interpreting cross-execution comparisons. [Matched arithmetic control](../matched_focal_heads/RESULTS.md)

## RGB validation and dense accuracy

The held-back RGB matches accepted 9 of 12 proposals. Each proposal changes at most five scalar values and keeps the frozen poses and within-view depth shape. A veto returns the selected focal with identity depth scales. The validation fold contains separate matched features from the same five images. The three development sequences were examined in earlier experiments.

| Scene | Condition | Focal direction | Held-back loss reduction | Decision | Guarded registered score |
|---|---|---|---:|---|---:|
| heads/seq-01 | Metadata fault | Shorter | 5.71% | accepted | 0.9824 |
| fire/seq-03 | Metadata fault | Shorter | 1.73% | accepted | 1.0153 |
| chess/seq-03 | Metadata fault | Shorter | 0.08% | validation veto | 0.9818 |
| heads/seq-01 | Metadata fault | Longer | 5.71% | accepted | 0.9824 |
| fire/seq-03 | Metadata fault | Longer | 1.73% | accepted | 1.0153 |
| chess/seq-03 | Metadata fault | Longer | 0.08% | validation veto | 0.9818 |
| heads/seq-01 | Valid zoom | Shorter | 11.39% | accepted | 0.9480 |
| fire/seq-03 | Valid zoom | Shorter | 4.47% | accepted | 0.9987 |
| chess/seq-03 | Valid zoom | Shorter | 1.54% | accepted | 1.0156 |
| heads/seq-01 | Valid zoom | Longer | 1.13% | accepted | 0.9764 |
| fire/seq-03 | Valid zoom | Longer | 3.03% | accepted | 1.0160 |
| chess/seq-03 | Valid zoom | Longer | 0.23% | validation veto | 0.9831 |

The weakest scene relative to the strongest aggregate local control was heads/seq-01, with 1.69% higher error. The two metadata directions reuse the same unmodified images. The experiment contains nine unique five-view RGB batches across three scenes. Its twelve conditions do not supply twelve independent scene replicates.

## Corresponding-pixel diagnostic

RGB-registered depth gave a condition-normalized corresponding-pixel mean 3D distance of 1.0039 for native frozen coordinates, 1.0000 for frozen FP32 and 1.0022 for the guarded result. The respective raw mean distances were 76.61, 76.33 and 76.59 mm.

Released depth gave a condition-normalized corresponding-pixel mean 3D distance of 1.0012 for native frozen coordinates, 1.0000 for frozen FP32 and 0.9959 for the guarded result. The respective raw mean distances were 82.58, 82.46 and 82.51 mm.

This secondary diagnostic aligns the point maps by one global similarity in double precision and compares predictions with depth at the same pixels before ICP. It supplies another measure of reconstruction quality on these registered inputs. The spending decision uses the originally registered surface metrics. [Paired-pixel measurements and RMSE](pixel_diagnostic.json)

## How much can rigid camera corrections recover?

Using the reference coordinates for this diagnostic, we fitted increasingly flexible transforms to exactly the same corresponding pixels. The values are the mean of condition-level optimal RMSEs in millimeters on the direct-focal predictions. Each added degree of freedom can only lower the fitted squared error.

| Reference | One similarity | Independent translations | Independent poses, shared scale | Independent similarities | Independent affine transforms |
|---|---:|---:|---:|---:|---:|
| Registered RGB | 148.28 | 145.31 | 143.80 | 142.87 | 139.72 |
| Released | 149.30 | 144.64 | 142.90 | 141.87 | 138.74 |

Even with reference geometry available, independently aligning and scaling each view reduces the average registered pixel RMSE by only 3.65%. Allowing independent scales after the rotations and translations accounts for 0.94 mm. The bound applies to corresponding-pixel RMSE. It explains the limited scope of one scale per view on this measure. The next local diagnostic should locate the remaining error within each image and measure how much of it is visible to RGB correspondences before fitting a spatially varying depth correction. [Experimental design](../../docs/DEPTH_GEOMETRY_HEADROOM.md) · [Replayable view statistics](geometry_headroom.json)

## Decision and cost

- At least 5% lower error than the best local control — failed.
- At least 5% lower error than the best historical control — failed.
- Neither distance component more than 2% worse than the local control — passed.
- Every scene at least as accurate as the local control — failed.
- Released-depth score at least as good as the local control — failed.

The local correction averaged 0.052 s per five-view request on the CPU, including dense point creation. Prior measurements on the same RGB batches averaged 4.34 s for the instrumented frozen RTX 3080 pass and 57.80 s for fresh CPU matching and track export. Their sum, 62.19 s, is a component-based estimate. Model initialization, dataset I/O and offline reference evaluation are excluded. The historical GPU measurements use different hardware. A head-to-head latency claim would require a common end-to-end benchmark.

Local execution added **$0 RunPod cost**. The two archives contain the same 27 unique point maps, 297,511,567 bytes per drive. [Prediction seals and timing](predictions.json) · [Archive checksums](archive.json) · [Budget](../../docs/BUDGET.md)

## Reproduction

From the repository root, run the following commands with the CPU dependencies used by GitHub Actions. The second command refits both bounded solvers from the published match systems, checks validation on the disjoint fold, imports the original controls and rebuilds the tables from the measured dense distances.

```sh
python -m unittest discover -s scripts -p test_fixed_camera_depth_scale.py
python scripts/score_depth_scale_study.py verify
python scripts/diagnose_depth_scale_pixels.py verify
python scripts/depth_geometry_headroom.py verify
```

To recompute distances from pixels and depths, use a separate scratch checkout with the dense pilot RGB and depth inputs and the frozen head archives. Generate predictions into an empty study output directory with `python scripts/run_depth_scale_study.py`, then run `python scripts/score_depth_scale_study.py evaluate` in the pinned TCO environment. [Dense input reconstruction](../dense_focal_pilot/REPLAY.md) · [Frozen head capture](../matched_focal_heads/RESULTS.md) · [Public sparse match systems](match_systems.zip)
