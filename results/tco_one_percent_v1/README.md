# Cached TCO under a 1% reconstruction margin

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

## Matched protocol

Each development case uses five 392 × 518 images in Heads/seq-01, Chess/seq-03 and Fire/seq-03, with frame IDs 0, 200, 400, 600 and 800. Three fresh processes per arm and scene use seeds 42, 43 and 44. The same camera poses and intrinsics are supplied to every arm. Reference depth, geometry and validity masks enter only the shared evaluator. The optimized reference collects four head inputs and adapts LoRA in all 24 frame/global stages. All adapted arms use the same TCO loss, Adam at 0.0003, rank-4 suffix LoRA and frozen original weights and heads.

The synchronized timer starts with the scene and loaded model on CPU. It includes placement, matching and cache construction where applicable, every update, final reconstruction and float32 unprojection. Evaluation time is excluded. Surface accuracy is prediction to reference and completeness is reference to prediction. Both are mean distances after shared Umeyama alignment and rigid ICP. Compute each arm's median across seeds within each sequence, divide by the full-40 median in that sequence, then average the ratios with one weight per physical scene.

The three boundary-adapter arms share the same mask and 8,192 learned parameters. The true-match arm forms a feature difference from projected cross-view donor patches. The own-feature arm uses its patch's cached feature within that mask. The shuffled arm preserves each receiving patch's support and permutes donor IDs within each ordered view pair. Graph construction is charged to all three arms.

## Runtime qualification

The first two deterministic full-size identity probes stalled and were stopped. Release-mode full-size identity checks and 224 × 294 fixed-adjoint checks then completed. A release-mode full-40 memory probe stalled in CUDA backpropagation. A test with an 87.5% process memory cap reached an explicit allocation failure. Enabling expandable allocator segments under that cap completed the unscored 40-step diagnostic in 76.62 seconds with 7.822 GiB peak allocated and 8.725 GiB peak reserved. The capped expandable configuration was committed and used for every scored comparison. [Early probes](preliminary_qualification.json) · [Allocator qualification](allocator_qualification.json) · [Pinned PyTorch allocator](https://github.com/pytorch/pytorch/blob/v2.7.1/c10/cuda/CUDACachingAllocator.cpp#L1141-L1147)

RunPod compute cost for this study was **USD 0**.
