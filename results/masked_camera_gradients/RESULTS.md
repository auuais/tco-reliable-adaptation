# Whole-camera omission moved the update away from the clean-prior gradient

At the frozen VGGT camera-head output, mean normalized gradient error was **0.530 with the unchanged corrupted priors and 1.009 after alert-based omission**, a 90.2% increase. Omission increased error for all three scenes and all four fault profiles. Eliminating all external camera supervision on the same alerts scored 0.929, and uniformly omitting one camera on those alerts scored 0.936. Even supplying the true faulty index to the omission rule scored 0.805. The branch missed every prespecified criterion for a dense pilot. [Recorded gradients and decision](summary.json) · [Frozen analysis](../../manifests/masked_camera_gradients.json)

| Fault | Unchanged corrupted priors | Omit flagged camera |
|---|---:|---:|
| Rotation | 0.379 | 0.978 |
| Translation | 0.625 | 1.037 |
| Focal length | 0.310 | 0.966 |
| Mixed | 0.807 | 1.053 |

The comparison uses the derivative of the upstream TCO pose and intrinsics losses with respect to the five nine-parameter camera outputs. The reference derivative uses the clean calibrated supplied priors on the same pretrained predictions. For each method we divide the norm of the gradient difference by the clean gradient norm. The detector raised 154 alerts on 180 corrupted inputs. One of three clean inputs, Heads, triggered omission of the first camera and had error 1.563. The five-view reference changes when the first camera is omitted; filtering a view also removes healthy components of its measurement. These observations motivate checking how much of an update should be changed, independently of the detector's ability to name a suspect. [Scoring code](../../scripts/audit_masked_camera_gradients.py)

The test used three development windows at frame offset zero. Alert thresholds came from the two other scenes at offsets 50, 100 and 150. It measured camera-head gradients on CPU in FP32. It did not measure LoRA parameter updates, rendering gradients or reconstruction errors. No held-out data or cloud compute was used.

## Replay

The compact inputs contain only derived initial camera encodings and RGB matches. The runner reads the unmodified pose files from the public scene or sequence ZIPs and validates all 15 clean poses in the exact FP32 representation consumed by the loss. During preflight, rebuilding the archived FP64 translations on a different numerical stack differed by at most 2.23 × 10⁻¹⁶. Both original and calibrated FP32 encodings matched by SHA-256 before the gradient experiment. The CPU decoder was also checked against the captured camera poses and intrinsics. [Input provenance](inputs.json)

Use `python scripts/audit_masked_camera_gradients.py verify` to recompute the decision from 1,098 published derivative arrays. For a full run, move the summary and gradient trace aside in a scratch checkout and run `python scripts/audit_masked_camera_gradients.py run --archives /path/to/7scenes-zips` with the pinned TCO checkout. [Derivative arrays](gradient_traces.zip) · [Public-data instructions](../sensor_public_replay/README.md)
