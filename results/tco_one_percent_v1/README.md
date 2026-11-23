# Cached TCO under a 1% reconstruction margin

**Last-12 caching with ten updates passed development and missed the 1% quality margin on both confirmation datasets.** The RTX 3080 resource gains transferred. The true-match boundary adapter also failed its development controls.

The registered target is **at most half the scene latency**, at least **20% lower peak allocated GPU memory**, and at most **1% relative increase in each of surface accuracy and completeness error** against optimized full-decoder TCO with 40 updates. The study uses a single RTX 3080 and public benchmark data. [Technical handoff](../../docs/TCO_ONE_PERCENT_HANDOFF.md) · [Development registration](../../manifests/tco_one_percent_v1.json)

## Development results

All 90 fixed-setting and adapter runs completed without retries. The last-14 configuration used 6.32 GiB against 7.82 GiB for full TCO in the unscored Heads screen, a 19.21% reduction. The protocol excluded it from the scored matrix at the 20% cutoff. [Engineering qualification](qualification.json)

| Method | Speedup vs full 40 | Lower peak allocated memory | Accuracy error change | Completeness error change |
|---|---:|---:|---:|---:|
| Zero-step VGGT | 20.26× | 18.27% | +3.30% | +7.58% |
| Full 5 | 4.05× | 0.00% | +2.43% | +1.25% |
| Full 10 | 2.86× | 0.00% | −3.97% | −6.49% |
| Cached last-12, 10 steps | 4.04× | 23.60% | −2.30% | −5.39% |
| Cached last-12, 15 steps | 3.29× | 23.60% | −2.89% | −5.83% |
| Cached last-12, 20 steps | 2.72× | 23.60% | −3.44% | −5.17% |
| Cached with own-feature adapter, 10 steps | 3.79× | 23.21% | −3.23% | −6.06% |
| Cached with true matches, 10 steps | 3.92× | 23.21% | −3.08% | −5.01% |
| Cached with shuffled matches, 10 steps | 3.88× | 23.21% | −3.36% | −5.85% |

Lower error is better. The simple ten-step cached run used nearly the same time as full five-step TCO and about 1.85 GiB less peak allocation. Heads drove the average quality gain against full 40. On Chess, the cached ten-step method raised accuracy error by 8.00% and completeness error by 10.34%.

The geometric adapter failed the registered mechanism comparison. True matches had higher aggregate errors than the own-feature and shuffled controls. Chess completeness error moved from 23.33 mm with plain caching to 23.70 mm with true matches. Shuffling changed 99.9% of Chess donor edges. Matching supported roughly 65% of Chess patches, 18% of Heads patches and 63% of Fire patches.

The current full-40 Heads medians were 14.48 mm accuracy error and 15.79 mm completeness error. The preceding benchmark measured 11.90 and 12.83 mm. The experiments used the same rooms and nominal optimization settings, with a new allocator and launch order in this study. Both reports therefore retain their own matched baselines. Same-seed ordinary-loss qualification also produced small numerical differences in the reconstructed point maps. [Earlier benchmark](../tco_frozen_prefix_v1/README.md) · [Prior per-room comparisons](../tco_frozen_prefix_v1/allocator_sensitivity_decision.json)

Nine timing-only pilots locked the maximum tested affordable full-decoder budgets to five steps for cached last-12/10 and the own-feature arm, six for cached last-12/15, and ten for cached last-12/20. All nine separately timed and scored six-step cases completed. Cached last-12/10 beat full 5, cached last-12/15 beat full 6, and the own-feature adapter beat full 5. Full 10 beat cached last-12/20. **Cached last-12/10 was the fastest eligible recipe.** [Timing registration](../../manifests/tco_one_percent_time_v1.json) · [Pilot evidence](timing_evidence.json) · [Development evidence](evidence.json) · [Replayed development decision](decision.json)

## Locked confirmation

The recipe was frozen before scoring four 7-Scenes transfer rooms and eight ETH3D physical scenes. The 7-Scenes rooms appeared in an earlier sparse-depth study. The chosen frames were disjoint from its registered capture frames. ETH3D was independent of our method selection. Its first five registered image IDs were read from the public ETH3D release at 350 × 518, with the pinned TCO loader and a remapping from official distorted reference depth to the supplied undistorted images. All six arms, including full four-step and five-step controls, used three seeds per scene. [Confirmation registration](../../manifests/tco_one_percent_confirmation_v1.json) · [Input and camera checks](confirmation_inputs.json) · [ETH3D release](https://www.eth3d.net/datasets)

| Dataset | Speedup vs full 40 | Lower peak allocated memory | Accuracy error change | Completeness error change | One-sided 95% upper error changes |
|---|---:|---:|---:|---:|---:|
| 7-Scenes transfer, 4 rooms | 4.02× | 23.60% | +45.31% | +28.50% | +71.31% accuracy, +52.96% completeness |
| ETH3D, 8 scenes | 4.17× | 23.64% | +181.46% | +240.87% | +364.59% accuracy, +523.18% completeness |

The cached method also lost the time-matched quality comparisons. On 7-Scenes its time was within 0.72% of full 5, with 2.28% higher accuracy error and 4.49% higher completeness error. On ETH3D full 4 fit the time budget and full 5 exceeded it. The cached method increased accuracy and completeness errors by 19.48% and 16.04% against full 4. Each comparison uses the ratio of physical-scene averages after normalizing both arms by the same full-40 medians. [Confirmation replay](confirmation_decision.json)

| Dataset and scene | Full 40 accuracy (mm) | Cached accuracy (mm) | Full 40 completeness (mm) | Cached completeness (mm) |
|---|---:|---:|---:|---:|
| 7-Scenes Office | 20.15 | 33.78 | 23.94 | 40.92 |
| 7-Scenes Pumpkin | 17.86 | 32.55 | 25.13 | 31.99 |
| 7-Scenes RedKitchen | 32.59 | 38.88 | 29.71 | 34.68 |
| 7-Scenes Stairs | 30.97 | 34.71 | 63.67 | 63.05 |
| ETH3D Courtyard | 279.41 | 1568.19 | 251.87 | 1371.60 |
| ETH3D Delivery Area | 146.86 | 189.65 | 154.82 | 190.29 |
| ETH3D Electro | 49.94 | 450.87 | 51.79 | 748.42 |
| ETH3D Facade | 314.21 | 460.73 | 261.26 | 353.95 |
| ETH3D Kicker | 31.19 | 49.81 | 31.55 | 47.84 |
| ETH3D Office | 26.19 | 34.22 | 25.16 | 33.61 |
| ETH3D Playground | 307.75 | 288.47 | 338.40 | 252.86 |
| ETH3D Terrains | 31.51 | 40.30 | 30.19 | 35.94 |

Electro and Courtyard accounted for much of the ETH3D mean increase. Even a descriptive reaverage over the other six scenes gives +31.28% accuracy error and +22.90% completeness error. The registered result includes all eight scenes. Seven of eight had an increase in both errors. The smaller reference errors on some scenes make relative losses larger, and the absolute distances show substantial regressions as well.

At ten updates, restoring full-decoder adaptation reduced both errors on both confirmation datasets. Extending full-decoder adaptation from ten to forty updates reduced both errors further. The latency and allocation savings came with a geometry cost on these data.

## Ten-view memory check

In one separately registered Heads/seq-01 window with frames 0, 100, …, 900, cached last-12 with ten updates completed in **28.33 seconds** with **7.75 GiB peak allocated** and **8.70 GiB peak reserved**. Its aligned surface accuracy and completeness errors were 10.24 mm and 10.72 mm. Full 40 exceeded available CUDA memory during a convolution. Both arms used the same 392 × 518 input size, seed and process cap. This is a one-window feasibility result. Full 40 produced no ten-view reference score. [Stress registration](../../manifests/tco_one_percent_view_stress_v1.json) · [Stress receipts](view_count_stress.json)

## Matched protocol

Each development case uses five 392 × 518 images in Heads/seq-01, Chess/seq-03 and Fire/seq-03, with frame IDs 0, 200, 400, 600 and 800. Three fresh processes per arm and scene use seeds 42, 43 and 44. The same camera poses and intrinsics are supplied to every arm. Reference depth, geometry and validity masks enter only the shared evaluator. The optimized reference collects four head inputs and adapts LoRA in all 24 frame/global stages. All adapted arms use the same TCO loss, Adam at 0.0003, rank-4 suffix LoRA and frozen original weights and heads.

The synchronized timer starts with the scene and loaded model on CPU. It includes placement, matching and cache construction where applicable, every update, final reconstruction and float32 unprojection. Evaluation time is excluded. Surface accuracy is prediction to reference and completeness is reference to prediction. Both are mean distances after shared Umeyama alignment and rigid ICP. Compute each arm's median across seeds within each sequence, divide by the full-40 median in that sequence, then average the ratios with one weight per physical scene.

The three boundary-adapter arms share the same mask and 8,192 learned parameters. The true-match arm forms a feature difference from projected cross-view donor patches. The own-feature arm uses its patch's cached feature within that mask. The shuffled arm preserves each receiving patch's support and permutes donor IDs within each ordered view pair. Graph construction is charged to all three arms.

## Runtime qualification

The first two deterministic full-size identity probes stalled and were stopped. Release-mode full-size identity checks and 224 × 294 fixed-adjoint checks then completed. A release-mode full-40 memory probe stalled in CUDA backpropagation. A test with an 87.5% process memory cap reached an explicit allocation failure. Enabling expandable allocator segments under that cap completed the unscored 40-step diagnostic in 76.62 seconds with 7.822 GiB peak allocated and 8.725 GiB peak reserved. The capped expandable configuration was committed and used for every scored comparison. [Early probes](preliminary_qualification.json) · [Allocator qualification](allocator_qualification.json) · [Pinned PyTorch allocator](https://github.com/pytorch/pytorch/blob/v2.7.1/c10/cuda/CUDACachingAllocator.cpp#L1141-L1147)

All **315 primary scored cases** completed, including 99 development and 216 confirmation runs. The 9 unscored timing pilots, 11 engineering probes under the final allocator, earlier allocator diagnostics, and secondary view-count test are tracked separately. Two initial Stairs confirmation runs timed out. A queued Office run was interrupted when the local serial batch paused for a diagnostic. All three completed on an unchanged fresh-process retry. RunPod compute cost for this study was **USD 0**. [Runtime checks](confirmation_runtime_checks.json) · [Per-attempt confirmation evidence](confirmation_evidence.json)

## Reproduce the decision on CPU

Run the replay from the repository root. It recomputes the timing bracket, threshold decisions, mechanism comparison, locked dataset summaries, and one-sided scene bootstrap bounds from the exported rows. It also checks that each confirmation case used the qualified input and reference hashes.

```powershell
python scripts/replay_tco_one_percent.py
python scripts/test_tco_one_percent_replay.py
```

For local GPU execution, use the checkpoint, image, TCO revision and bfloat16 export compatibility patch in the handoff. The registries carry exact raw input hashes, the selected frames and the runtime settings. The launcher uses fresh offline containers and skips cases with valid completion receipts. The confirmation asset preparer uses official archives and the pinned calibration/undistortion helper. [Development handoff](../../docs/TCO_ONE_PERCENT_HANDOFF.md) · [Asset preparation code](../../scripts/prepare_tco_one_percent_confirmation.py)

```powershell
python scripts/tco_one_percent_protocol.py verify
python scripts/check_tco_one_percent.py
python scripts/run_tco_one_percent_local.py qualification
python scripts/run_tco_one_percent_local.py fixed
python scripts/run_tco_one_percent_local.py boundary
python scripts/run_tco_one_percent_local.py equal_time
python scripts/run_tco_one_percent_local.py confirmation
```

The private evidence archive on the second drive contains the prediction and completion files for both benchmark phases, the qualified engineering cases, and the view-count check. Raw public benchmark assets and model weights stay on the local research disks. [Verified archive receipt](archive_receipt.json)
