# Full 40-step TCO with selective depth-head recomputation

**Selective depth-head recomputation made ten-view, full 40-step TCO fit on the 10 GiB RTX 3080.** All nine candidate runs completed across three rooms. Every tested ten-view policy without recomputation ran out of memory on both attempts in each room. The five-view comparisons were inconclusive at the registered 1% relative margin for both datasets. All 108 scored cases completed. [Decision and bounds](decision.json) · [Per-attempt evidence](evidence.json)

## Native memory limit

The instrumented five-view reference finished all 40 steps in 77.29 seconds. Peak allocated memory was 7.82 GiB and peak reserved memory was 8.73 GiB. The separate uninstrumented reference attempt reached the 20-minute wall-clock limit. Its terminal receipt identifies a timeout.

The instrumented ten-view run completed the initial prediction and the first joint transformer and camera-head calls. At the depth-head entrance, 7.03 GiB was allocated. The first eight-frame depth chunk then failed at its high-resolution fusion convolution during update zero. CUDA requested 260 MiB with 8.69 GiB already allocated under the 8.75 GiB cap. The uninstrumented ten-view attempt failed at the same convolution. [Native profiles](profile.json) · [Attempt receipts](initial_evidence.json)

## Two-update factorial

All eight tests used ten 392 × 518 views of Heads/seq-01 and seed 42. Chunk size controls the number of views per depth-head call. Every transformer call and rendering loss still receives all ten views. Recomputing a chunk during backward avoids holding its internal activations from the original forward pass.

| Maximum frames per chunk | Ordinary execution | Selective recomputation |
|---:|---|---|
| 8 | Forward OOM | Backward OOM |
| 4 | Forward OOM | Backward OOM |
| 2 | Forward OOM | Completed at 8.41 GiB peak allocated and 6.90 s/update |
| 1 | Forward OOM | Completed at 8.29 GiB peak allocated and 6.54 s/update |

One frame per chunk ranked first by the registered two-update time. Across three unscored full-length runs the median was **156.52 seconds** with **8.29 GiB peak allocated** and **8.75 GiB peak reserved**. The full-length same-chunk control hit a CUDA OOM. The locked design uses one frame per chunk with recomputation and retains every chunk-only setting as a ten-view control. [Screen evidence](screen.json) · [Locked registration](../../manifests/tco_full40_recompute_locked_v1.json)

## Numerical qualification

The pinned PyTorch 2.7.1 CPU fixture reproduced depth, confidence and all four tap gradients exactly when it toggled recomputation at each fixed chunk size. With one image per chunk, the reduced-resolution GPU fixture also matched exactly across two fixed-adjoint updates. This covered 1,536 gradient and updated parameter tensors, both render orders, and the final predictions. Swapping the evaluation surface and mask left the prediction hash identical.

Changing from the native chunk size to one image per chunk introduced numerical differences before recomputation. At full-size zero-step inference the largest relative L2 difference across the exported predictions was 0.034%. The fixed-adjoint test showed that gradients can be more sensitive than predictions to this chunk-size change. The registered five-view tests measure how much these differences matter after 40 real updates. [Qualification receipts](qualification.json)

Two native runs with seed 42 also finished independently on each of Heads and Chess. Their accuracy errors were 14.09 and 14.94 mm on Heads, and 15.26 and 15.51 mm on Chess. Completeness errors were 15.60 and 16.40 mm on Heads, and 18.42 and 19.60 mm on Chess. These repeats measure ordinary variation in the reference pipeline. The locked comparisons use fresh seed-matched reference cases and retain the 1% margin. [Repeat measurements](profile.json)

## Ten-view capacity

Each row summarizes three fresh processes. All nine candidate runs saw all ten frames at 392 × 518 and completed 40 updates across the 384 LoRA tensors in the full decoder. The timer includes GPU placement, adaptation and final reconstruction. Geometric scoring runs after the timer.

| Room | Completed | Median time | Peak allocated | Median peak reserved | Accuracy (mm) | Completeness (mm) |
|---|---:|---:|---:|---:|---:|---:|
| Heads | 3/3 | 155.48 s | 8.29 GiB | 8.69 GiB | 9.04 | 10.56 |
| Office | 3/3 | 155.83 s | 8.29 GiB | 8.73 GiB | 14.02 | 16.45 |
| Pumpkin | 3/3 | 155.51 s | 8.29 GiB | 8.75 GiB | 16.91 | 27.73 |

Each of the four chunking-only policies ran out of memory on both first-seed attempts in each room, for 24 explicit CUDA OOM receipts. The remaining 24 control slots were conditional on completing the first seed. The candidate fit all three windows under the same cap. Scene times ranged from 155.20 to 156.48 seconds across the nine runs, and peak reserved memory ranged from 8.69 to 8.75 GiB. The ten-view table gives absolute distances to the aligned reference surface. The ten-view comparison establishes capacity.

## Five-view development

All 27 runs completed in Heads, Chess and Fire. Each comparison averages room-level ratios of three-seed medians. Lower is better for both distance errors.

| Comparison | Scene time | Peak allocated | Peak reserved | Accuracy error | Completeness error |
|---|---:|---:|---:|---:|---:|
| One-frame recomputation vs native execution | +4.22% | −9.33% | −15.88% | +2.42% | +3.41% |
| One-frame ordinary execution vs native execution | +0.11% | +0.10% | −9.29% | +10.76% | +9.49% |
| One-frame recomputation vs one-frame ordinary execution | +4.11% | −9.42% | −7.27% | −7.11% | −5.32% |

The output and gradients were identical in the same-chunk fixed-adjoint fixture. These real-loss runs include variation from renderer and optimizer trajectories. The development comparisons establish the resource tradeoff and describe the observed distances. The confirmation decision uses its own rooms and paired uncertainty bounds.

## Five-view 7-Scenes confirmation

All 24 cases completed. The candidate reduced the mean normalized accuracy error by **4.42%** and completeness error by **0.73%**. The one-sided 95% upper bounds on error change were **−0.85%** for accuracy and **+4.26%** for completeness. Accuracy cleared the registered 1% margin. Completeness was inconclusive. In the secondary bootstrap that also resampled matched seeds within rooms, the upper bounds were +4.67% and +8.38%.

| Room | Native accuracy | Recomputed accuracy | Native completeness | Recomputed completeness |
|---|---:|---:|---:|---:|
| Office | 21.50 mm | 18.68 mm | 26.42 mm | 23.85 mm |
| Pumpkin | 18.93 mm | 18.83 mm | 25.71 mm | 25.69 mm |
| RedKitchen | 31.56 mm | 31.25 mm | 28.26 mm | 30.70 mm |
| Stairs | 31.98 mm | 30.98 mm | 64.34 mm | 63.24 mm |

The candidate took 8.34% longer and used 9.33% less peak allocated and 15.66% less peak reserved memory. The candidate and reference each had three fresh seeds per room. Each room contributes equally to the dataset ratios, and the primary bootstrap resamples rooms with both metrics together. Four rooms limit the precision of the bound.

## Five-view ETH3D confirmation

All 48 cases completed. Mean normalized error rose **6.32%** for accuracy and **2.53%** for completeness. The one-sided 95% upper bounds were **+24.56%** and **+13.86%**, while the lower bounds were **−6.43%** and **−6.52%**. Both observed increases exceeded 1%. Both uncertainty intervals crossed the margin, so both tests were inconclusive. The secondary bootstrap that also resampled paired seeds gave upper bounds of +22.10% and +15.54%.

| Scene | Native accuracy | Recomputed accuracy | Native completeness | Recomputed completeness |
|---|---:|---:|---:|---:|
| Courtyard | 329.21 mm | 573.02 mm | 331.15 mm | 479.16 mm |
| Delivery area | 146.89 mm | 155.63 mm | 153.98 mm | 155.57 mm |
| Electro | 52.86 mm | 49.10 mm | 54.12 mm | 49.91 mm |
| Facade | 345.42 mm | 324.41 mm | 284.14 mm | 270.72 mm |
| Kicker | 37.13 mm | 30.79 mm | 35.72 mm | 30.48 mm |
| Office | 23.23 mm | 25.47 mm | 22.07 mm | 24.79 mm |
| Playground | 269.70 mm | 270.04 mm | 283.53 mm | 286.81 mm |
| Terrains | 33.11 mm | 30.16 mm | 33.32 mm | 29.37 mm |

Courtyard had the largest relative increase in both errors and is included in the eight-scene primary result. The candidate took 10.41% longer and used 7.14% less peak allocated and 13.39% less peak reserved memory.

## Decision and coverage

| Test | Accuracy change | Completeness change | One-sided 95% upper bounds | Verdict |
|---|---:|---:|---:|---|
| 7-Scenes, four rooms | −4.42% | −0.73% | −0.85%, +4.26% | Accuracy passed, completeness inconclusive |
| ETH3D, eight scenes | +6.32% | +2.53% | +24.56%, +13.86% | Both inconclusive |

The five-view tests use 392 × 518 inputs for 7-Scenes and 350 × 518 for ETH3D. Each scene has fresh matched runs at seeds 7, 77 and 777. Accuracy measures distance from prediction to reference, and completeness measures the reverse, after Umeyama alignment and rigid ICP. For each scene, divide the candidate's three-seed median by the native three-seed median. The dataset statistic is the mean of these scene ratios. A pass requires both the point estimate and the one-sided 95% upper bound to allow at most a 1% increase in each error. The replay uses 10,000 paired scene resamples and also provides paired scene-and-seed bounds and leave-one-scene-out results. The confirmation scenes were reused from the previous benchmark.

The locked roster contains nine completed ten-view candidate cases, 27 completed development cases and 72 completed confirmation cases. The 12 ten-view control cases each produced two explicit OOMs. One ETH3D Office candidate process exited with code 139 on its first attempt after logging iteration 33. The unchanged retry completed all 40 steps and supplied the registered score. The engineering stage also recorded nine OOM attempts and a separate native five-view wall-clock timeout. Local execution from first launch to final retry took **5 hours 3 minutes**. RunPod compute cost was **USD 0**. [Locked case receipts](evidence.json) · [Engineering receipts](initial_evidence.json)

## Local reproduction

The local launcher uses the pinned offline image, the verified VGGT checkpoint, read-only input data and one fresh process per case. Measurements use the same expandable allocator and 87.5% process cap in all arms. RunPod compute cost is **USD 0**. [Handoff](../../docs/TCO_FULL40_RECOMPUTE_HANDOFF.md) · [Registration](../../manifests/tco_full40_recompute_v1.json) · [Preflight](preflight.json)

```powershell
python scripts/tco_full40_recompute_protocol.py verify
python scripts/run_tco_full40_recompute_local.py initial --cohort fixture --cohort reference --cohort profile --cohort screen
python scripts/run_tco_full40_recompute_local.py initial --job promote_c1_on_r1 --job promote_c1_on_r2 --job promote_c1_on_r3 --job promote_c1_off_r1
python scripts/run_tco_full40_recompute_local.py initial --job fixed_c8_off --job identity_c8_off --job fixed_c1_off --job identity_c1_off --job fixed_c1_on --job leak_c1_on --job identity_c1_on
python scripts/run_tco_full40_recompute_local.py initial --cohort repeat
python scripts/run_tco_full40_recompute_local.py initial --export
python scripts/check_tco_full40_recompute.py profile
python scripts/check_tco_full40_recompute.py screen
python scripts/check_tco_full40_recompute.py qualification --candidate c1_on
python scripts/tco_full40_recompute_protocol.py verify --registration locked
python scripts/run_tco_full40_recompute_local.py locked --cohort ten --cohort development --cohort confirmation
python scripts/run_tco_full40_recompute_local.py locked --cohort ten --cohort development --cohort confirmation --retry
python scripts/run_tco_full40_recompute_local.py locked --export
python scripts/replay_tco_full40_recompute.py
python scripts/test_tco_full40_recompute_replay.py
```

The verified local archive contains the raw prediction arrays, memory traces, run logs, registrations and public results. The public replay checks scores, thresholds and attempt receipts without a GPU. [Verified archive receipt](archive_receipt.json)
