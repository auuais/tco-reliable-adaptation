# TCO stall diagnosis, five-view quality, and ten-view capacity on an RTX 3080

The study uses the same local 10 GiB RTX 3080, pinned VGGT checkpoint, TCO runtime, native image grouping, and 40 real Adam updates. The five-view quality test compares unmodified local TCO against whole-camera checkpointing. The ten-view experiment adds two renderer and optimizer lifetime interventions to that checkpoint. Rental compute was **USD 0**. [Handoff and acceptance rules](../../docs/TCO_NATIVE_STALLS_QUALITY_CAPACITY_HANDOFF.md) · [Prospective registration](../../manifests/tco_native_stalls_quality_capacity_v1.json)

## Five-view stall diagnosis

All 12 clean development slots finished. In the longest run, native Chess seed 42 took **1,032.22 seconds** from model placement through FP32 reconstruction. Its independent same-seed replica took **88.72 seconds** at the same **7.822 GiB** peak allocation. Scoring took 13.61 and 11.34 seconds, respectively. The corresponding whole-camera checkpoint ran in 81.19 seconds and allocated 6.302 GiB. Both checkpointed Chess runs used 6.302 GiB, and all four native cases used 7.822 GiB. [Stage 1 evidence](native_stalls.json)

Separate event traces covered native Chess, checkpointed Chess, native Courtyard, and native Heads. Each confirmed all 40 Adam updates on the CUDA stream and reached scoring, serialization and a valid terminal receipt. None reproduced the multi-minute delay. The fresh fast traces therefore show the normal path, while historical timeout stacks and the slow clean case point to delays inside reconstruction without isolating a driver, kernel or allocation mechanism. A printed loss row precedes backward and Adam. Even row 39 supplies no evidence that the last update finished. [Historical receipt audit](historical_stalls.json) · [Trace and host telemetry](host_telemetry.json)

Independent CPU rescoring verified the eight saved development native reconstructions. Native replicas with identical seeds and inputs still produced different aligned geometry. The individual millimetre and percentage differences are reported by seed, separate from the held-out estimates. [Repeatability pairs and rescores](native_stalls.json)

One checkpointed Chess execution completed model work before an in-memory batch-layout receipt check failed. The verifier compared a list of tuples with the serialized list of lists. The launcher was interrupted during its first retry. Both attempt receipts were retained, the check was corrected and committed, and the same registered case then finished with all 40 verified updates. The source correction and its one affected retry limit are frozen in an additive amendment. [Receipt correction](../../manifests/tco_native_stalls_quality_capacity_receipt_correction_v1.json)

## Five-view reconstruction quality

The locked comparison completed **90/90** cases on the first attempt. It used seven 7-Scenes and eight ETH3D locations with fresh frames and three matched seeds per location. Whole-camera checkpointing passed the zero-update identity check, the deterministic signed-gradient/Adam comparison on two matched updates, the sign-flip negative control, and the dummy-reference substitution test. The production-backend comparisons are reported separately. [Quality qualification](quality_qualification.json) · [Candidate and roster lock](../../manifests/tco_native_stalls_quality_capacity_locked_v1.json)

The percentages are relative changes in mean directed surface error after Umeyama similarity alignment and rigid ICP. Lower values are better. The interval columns report the separate one-sided 95% bounds from paired scene-and-seed resampling. The acceptance threshold is an estimate and upper bound of at most **+1.00%**, before rounding.

| Dataset | Error | Estimated change | Lower bound | Upper bound | 1% verdict |
|---|---|---:|---:|---:|---|
| 7-Scenes | Accuracy | +1.09% | −2.54% | +8.21% | Inconclusive |
| 7-Scenes | Completeness | +2.29% | −3.02% | +7.91% | Inconclusive |
| ETH3D | Accuracy | −5.88% | −14.36% | +0.85% | Pass |
| ETH3D | Completeness | +4.71% | −13.35% | +27.90% | Inconclusive |

The joint 1% claim remains inconclusive with complete coverage. Of the four development native-repeat pairs, four exceeded 1% on accuracy and three on completeness. Independent CPU rescoring of eight native development outputs and four confirmation outputs produced a maximum discrepancy of **2.68 × 10⁻¹¹ mm**. The development spread and the confidence intervals describe different measurements. [Quality estimates, absolute scores and sensitivity analysis](quality_decision.json)

Checkpointing lowered peak allocated memory by **1.520 GiB in every one of the 45 held-out pairs**. The median within-pair reduction was 19.4% on 7-Scenes and 20.3% on ETH3D. The median paired scene time changed by −1.0% and +1.4%, respectively. [Paired resources](quality_decision.json)

## Ten-view capacity with native depth batches

The correctness tests qualified all four checkpointed policies. The protected normal-loss and visualization paths passed. The actual-renderer gradient tests detected the detached-opacity control, and the optimizer verifier detected deliberately reset Adam moments. Each tested lifetime policy passed the two-update matched replay of signed gradients from the actual loss. [Renderer and optimizer qualification](qualification.json)

All ten clean screen attempts hit explicit CUDA out-of-memory errors at 392 × 518 under the 8.75 GiB process cap. Each screen would have required two full Adam updates and a reconstruction with the native depth batches of eight and two.

| Policy | Clean completions | Update and failing operation | Allocation request |
|---|---:|---|---:|
| Native | 0/2 | Update 1, depth head | 260 MiB |
| Whole-camera checkpointing | 0/2 | Update 2, RGB rasterizer | 24 MiB |
| Checkpointing + skip unused normals | 0/2 | Update 2, depth-head positional embedding addition | 794 MiB |
| Checkpointing + clear spent gradients | 0/2 | Update 2, RGB rasterizer | 24 MiB |
| Checkpointing + both changes | 0/2 | Update 2, depth-head positional embedding addition | 794 MiB |

The two separate diagnostic attempts reproduced the failures and each confirmed a completed first Adam update. At the checkpointed run's second RGB entry, the inventory showed **102.63 MiB** of unused target-normal storage and **12 MiB** of spent gradients. The combined trace verified that the old gradient storage dropped to zero before the next forward, while the **24 MiB** of Adam moments stayed live. The combined trace reached a lower recorded peak over the executed prefix, 8.509 GiB versus 8.663 GiB. The depth-head failure came before its second RGB call, so these peaks cover different amounts of work. The shift in first failure does not establish why the actual allocator could not satisfy the larger request. [Screen and trace ledger](evidence.json) · [Host telemetry](host_telemetry.json)

The registered pilot and capacity confirmation remained gated. This is a replicated failure of the tested two-update workload on this runtime and GPU. The ten-view full-40 capacity target remains unestablished. A local follow-up should examine the 794 MiB depth-head temporary and preserve the native eight-plus-two grouping. Any revised method should repeat the two-update capacity check before a full pilot.

## Reproduction and verification

The attempt ledger contains **151 terminal receipts** with 137 valid completions, 12 OOMs, one receipt validation error, and one interruption during the resulting operational correction. Elapsed wall time from the first recorded launch to the last terminal receipt was **4 h 27 min**. The study used only the local RTX 3080. [Independent decision](decision.json) · [Second-disk verification](archive_receipt.json)

```powershell
python -B -S scripts/tco_native_stalls_quality_capacity_protocol.py verify
python -B -S scripts/report_tco_native_stalls.py
python -B -S scripts/replay_tco_native_stalls_quality_capacity_release.py
python -B -S scripts/test_tco_native_stalls_quality_capacity.py
python -B -S scripts/archive_tco_native_stalls_quality_capacity.py verify --check-source
```
