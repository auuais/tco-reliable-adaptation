# Camera-head checkpointing with native depth batches

**Camera-head checkpointing saved 1.52 GiB of peak allocated memory in every matched five-view comparison.** It moved the ten-view memory failure from the depth head to RGB rendering during the second update. The five-view evidence was inconclusive at the 1% relative margin on both datasets. All 90 locked five-view runs completed, including the Courtyard revisit. RunPod compute cost was **USD 0**. [Decision](decision.json) · [Handoff and acceptance rules](../../docs/TCO_COURTYARD_CAMERA_HANDOFF.md)

## Courtyard diagnosis

CPU rescoring verified all six archived outputs. The largest historical accuracy gap was 318 mm after similarity alignment and 308 mm after ICP. Changing the depth batch size produced a small raw prediction difference before the first optimizer update. At a fixed batch size, the checkpointed and ordinary runs had identical raw outputs at step zero, with differences first observed after one real update. Matched native repeats alone differed by 149–266 mm in final accuracy error. Fresh reruns at the original three seeds found differences in both directions, leaving the causal contribution of depth-head checkpointing unresolved. All 12 separate-process replays reproduced the original final readout bit for bit. [Courtyard report](COURTYARD.md) · [Trace data](diagnosis.json)

## Qualification

The candidate checkpointed the gradient-bearing camera head and retained native depth execution. It kept five views in one depth-head batch and ten views in two batches of eight and two. The original joint rendering loss and all 24 paired LoRA stages remained active. In the pinned CPU fixture, all tested camera outputs and gradients were bitwise identical. The reduced-raster GPU fixed-adjoint tests also matched all 1,536 gradient and adapter-update tensors bitwise. The full-size zero-update predictions and the dummy-reference leak probe matched exactly. Real-loss benchmark runs used the original objective. [Qualification](qualification.json)

## Ten views

Camera checkpointing **did not complete the two-update gate**. In the profiled native run, memory at the first adaptation camera-head exit was 7.025 GiB. The corresponding camera-checkpointed run held 5.501 GiB, a reduction of 1.524 GiB. Native execution failed on a 260 MiB allocation in the first depth-head chunk, with 8.69 GiB allocated at the failing operation. The candidate completed the depth head, the renderer, and the first Adam update. During the second update, both clean candidate attempts and the profiled attempt failed on a 24 MiB allocation in RGB rendering, with 8.66 GiB allocated. The common allocator cap was 8.75 GiB, and CUDA reported zero free device memory at the candidate failures. The two clean attempts took 916 and 111 seconds to reach terminal status. The ten-view full-length pilot and all 18 locked slots were gated off. The earlier one-image depth method remains the demonstrated ten-view fit on this card. [Capacity gate](screen.json) · [Previous ten-view study](../tco_full40_recompute_v1/README.md#ten-view-capacity)

The five-view two-update probes completed at 7.822 GiB allocated for native TCO and 6.302 GiB for camera checkpointing, with times of 14.43 and 14.54 seconds. These probes established feasibility. Full 40-step timing and quality use the locked results. [Engineering receipts](initial_evidence.json)

## Five-view 7-Scenes confirmation

All 42 registered runs finished. The table gives absolute three-seed median mean surface distances and scene times.

| Scene | Native accuracy (mm) | Checkpointed accuracy (mm) | Native completeness (mm) | Checkpointed completeness (mm) | Native / checkpointed time (s) |
|---|---:|---:|---:|---:|---:|
| Chess | 25.64 | 25.63 | 29.23 | 28.76 | 101.64 / 76.37 |
| Fire | 17.61 | 17.55 | 20.48 | 20.46 | 112.42 / 76.47 |
| Heads | 13.29 | 12.39 | 14.80 | 13.13 | 341.20 / 76.77 |
| Office | 33.08 | 36.23 | 39.27 | 43.04 | 104.55 / 75.46 |
| Pumpkin | 17.98 | 18.30 | 21.89 | 22.61 | 80.78 / 74.62 |
| RedKitchen | 16.52 | 17.39 | 19.18 | 20.09 | 105.64 / 75.93 |
| Stairs | 23.12 | 22.97 | 78.05 | 77.13 | 166.21 / 75.55 |

Mean normalized accuracy error rose **1.27%** and completeness error rose **0.50%**. The one-sided 95% upper bounds were **+4.27%** and **+4.17%**, respectively. Both lower bounds were below the 1% margin. Both decisions are inconclusive. Resampling matched seeds within each sampled room increased the upper bounds to +14.65% and +17.16%. Office had the largest ratio increase in both errors. Peak allocated memory fell **19.43%**, from 7.822 to 6.302 GiB, and peak reserved memory fell **14.75%**. Mean normalized scene time was **36.07% shorter** in this Windows workload, with substantial variation in native runtime. The lower-memory variant took slightly longer on ETH3D. [Scene ratios and intervals](decision.json)

## Five-view ETH3D confirmation

All 42 registered runs finished. The table gives absolute three-seed median mean surface distances. Accuracy is prediction-to-reference distance. Completeness reverses the direction. Lower is better for both.

| Scene | Native accuracy (mm) | Checkpointed accuracy (mm) | Native completeness (mm) | Checkpointed completeness (mm) |
|---|---:|---:|---:|---:|
| Delivery Area | 190.64 | 317.36 | 180.00 | 373.34 |
| Electro | 151.12 | 162.87 | 110.57 | 137.62 |
| Facade | 189.94 | 148.02 | 179.24 | 139.75 |
| Kicker | 24.93 | 25.79 | 24.02 | 24.54 |
| Office | 21.56 | 24.95 | 21.87 | 24.49 |
| Playground | 245.10 | 256.76 | 227.47 | 232.68 |
| Terrains | 40.32 | 37.71 | 37.00 | 34.49 |

The equal-weight mean ratio across scenes implies accuracy error **+9.95%** and completeness error **+17.07%**. The one-sided 95% upper bounds were **+26.64%** and **+43.94%**, respectively. Both lower bounds were below the 1% margin, so the two decisions are inconclusive and both point estimates missed the target. Resampling matched seeds within each sampled scene gave sensitivity upper bounds of +27.45% and +39.53%. The candidate reduced allocated memory by **20.26%** and reserved memory by **17.68%**, with scene time **1.88% longer**. Delivery Area contributed the largest increase. [Replayed decisions and leave-one-scene-out values](decision.json)

## Courtyard revisit with native depth batches

| Seed | Native accuracy / completeness (mm) | Camera-checkpointed accuracy / completeness (mm) |
|---:|---:|---:|
| 7 | 193.47 / 163.43 | 669.10 / 534.86 |
| 77 | 925.64 / 915.67 | 742.66 / 626.83 |
| 777 | 240.85 / 198.25 | 606.86 / 519.67 |

The clean camera-only variant increased both errors in two of three matched seeds. The native accuracy error alone ranged from 193 to 926 mm. The median accuracy error rose from 241 to 669 mm and completeness error rose from 198 to 535 mm. Keeping the native depth batches did not resolve the Courtyard instability in this revisit. This scene was registered as a diagnostic and did not enter the ETH3D confirmation estimate. [Six revisit receipts](evidence.json)

## Pairing and inference

For each scene, divide the candidate median over seeds 7, 77 and 777 by the matched native median. Average the scene ratios equally within each dataset. A pass requires both the point estimate and one-sided 95% upper bound to allow at most a 1% increase in each mean distance. Resample scenes in paired blocks, with 10,000 replicates and seed 20261125. The sensitivity calculation also resamples matched seeds within each sampled scene. Both scorers use the same fixed evaluation mask, Umeyama alignment and rigid point-to-point ICP with a 0.1 m threshold. A missing primary pair leaves its dataset decision inconclusive.

The 14 primary windows use different frame indices from the previous full-40 benchmark. The recorded project audit found no exact index overlap in its identifiable prior manifests and captures. All locations have prior project exposure and nearby views can share content. These windows measure performance on new inputs in known locations. Use the raw input, camera, depth and decoded reference digests in the registration to reproduce them. [Initial registration and exposure audit](../../manifests/tco_courtyard_camera_v1.json) · [Locked registration](../../manifests/tco_courtyard_camera_locked_v1.json)

## Run coverage and cost

The diagnostic and engineering phase had 35 completed cases and five explicit CUDA OOM attempts. The first CPU archive attempt stopped on an import-path error before decoding data. The corrected runner reproduced the archived results, and the registration retains the original attempt digest and source amendment. Four locked native 7-Scenes runs timed out at the 20-minute limit. All four completed on unchanged retries. The final result includes 84 confirmation runs, six Courtyard revisit runs and 18 gated ten-view slots. Local elapsed time from first launch to final retry was **6 hours 36 minutes**, with **135 terminal attempt receipts**. No RunPod compute was used. [Engineering evidence](initial_evidence.json) · [Locked evidence](evidence.json)

## Local reproduction

Use the recorded offline Docker image, local VGGT checkpoint, public data and common allocator. The launcher starts a fresh isolated process per case. The final CPU replay uses only the public scalar results, artifact digests and production receipts. [Driver and launch commands](../../docs/TCO_COURTYARD_CAMERA_HANDOFF.md) · [Raw archive verification](archive_receipt.json)

```powershell
python scripts/tco_courtyard_camera_protocol.py verify --registration initial
python scripts/tco_courtyard_camera_protocol.py verify --registration locked
python scripts/run_tco_courtyard_camera_local.py initial --cohort rescore
python scripts/run_tco_courtyard_camera_local.py initial --cohort trace --cohort readout --cohort trace_clean
python scripts/run_tco_courtyard_camera_local.py initial --cohort diagnosis
python scripts/run_tco_courtyard_camera_local.py initial --cohort fixture --cohort qualification
python scripts/run_tco_courtyard_camera_local.py initial --cohort gate --cohort screen5
python scripts/run_tco_courtyard_camera_local.py initial --job gate_camera --retry
python scripts/run_tco_courtyard_camera_local.py locked --cohort confirmation --cohort revisit
python scripts/run_tco_courtyard_camera_local.py locked --cohort confirmation --cohort revisit --retry
python -B -S scripts/replay_tco_courtyard_camera.py
python -B -S scripts/test_tco_courtyard_camera_replay.py
```
