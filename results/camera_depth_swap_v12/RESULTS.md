# Camera and depth changes interact on Heads

We crossed the initial and adapted camera estimates with the initial and adapted depths for the same 13 completed TCO trajectories. Each reconstruction uses the first camera as its coordinate frame. We removed one common scale from both the final depths and final camera centers, using the median ratio between final and initial depth. A fifth condition uses the supplied clean cameras directly with initial depth, rescaling the trajectory by the median camera baseline. All outputs use the same numerical precision. The five conditions give 65 paired outputs and 45 distinct point maps.

| Camera estimates | Depth | Normalized error | Accuracy error (cm) | Completeness error (cm) | Harmful cases / 9 |
|---|---|---:|---:|---:|---:|
| Initial | Initial | 1.0000 | 1.6051 | 2.0320 | 0 |
| Adapted | Initial | 0.9498 | 1.6448 | 1.7424 | 4 |
| Initial | Adapted | 1.1291 | 1.8346 | 2.0721 | 4 |
| Adapted | Adapted | 0.9343 | 1.5114 | 1.7960 | 2 |
| Supplied priors | Initial | 0.9615 | 1.6565 | 1.7380 | 3 |

The table uses repeat zero on three scenes and three optimizer seeds. Distance and normalized error are lower for better reconstructions. Harmful cases exceed the FP32 zero-step error by over 1% on either distance. The source model predicts the same initial geometry across seeds within each scene, so the three direct-prior cases on Heads repeat one deterministic prediction. The raw rows include four additional same-seed executions. Across both reconstructed endpoints the maximum discrepancy from the previous matched-precision distance measurements was 0.105%, below the prespecified 0.5% tolerance.

| Scene | Camera effect at initial depth | Depth effect at initial cameras | Interaction | Combined effect |
|---|---:|---:|---:|---:|
| Heads | +0.0143 | +0.5056 | −0.4487 | +0.0712 |
| Chess | −0.0909 | −0.0963 | +0.0031 | −0.1842 |
| Fire | −0.0738 | −0.0222 | +0.0119 | −0.0841 |

Each entry is a change in normalized error from the initial reconstruction. The interaction equals the combined change minus the two single-factor changes. On Heads, adapted depths had substantially higher error with the initial cameras. Their compatibility improved with the adapted cameras. On Chess, each change independently lowered error and the interaction was small. These contrasts describe the stored endpoints under the chosen scale convention. They do not identify what would happen if camera or depth parameters were frozen during training.

[Experiment design](../../manifests/camera_depth_swap_v12.json) · [All numeric outcomes](summary.json) · [Prediction hashes](seal.json) · [Two-drive archive record](archive.json)
