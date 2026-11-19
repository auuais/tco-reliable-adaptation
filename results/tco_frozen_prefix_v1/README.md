# Frozen-layer caching makes TCO faster on the RTX 3080

The best quality–efficiency tradeoff was **10 updates with the last 12 VGGT stages trainable**. It was **4.79–5.21× faster** and used **23.6% less peak allocated GPU memory** than full-decoder 40-step TCO with the four-output optimization. Its completeness error stayed within the 1% target. Its surface accuracy error increased by **1.96–2.26%**, which ruled it out under the registered quality criterion. All other cached configurations also missed the accuracy target. [Handoff](../../docs/TCO_FROZEN_PREFIX_HANDOFF.md) · [Frozen registration](../../manifests/tco_frozen_prefix_v1.json) · [Replayed bounds](partial_bounds.json)

## Matched results

Each result uses five 392 × 518 images per scene on Heads, Chess and Fire, with clean poses and intrinsics supplied to TCO and reference depth reserved for scoring. The adapted arms used the same rendering objective, rank 4 and Adam learning rate 0.0003. The code uses frozen VGGT weights and depth and camera heads. The timer starts with the checkpoint and scene inputs on CPU and includes device placement, the initial forward, cache construction, every update, and final 3D output. The same float32 geometry readout is used for every score.

| Trainable stages | Updates | Speedup | Lower peak allocation | Accuracy error change | Completeness error change |
|---:|---:|---:|---:|---:|---:|
| 12 | 5 | 6.30–6.84× | 23.6% | +9.00 to +9.30% | +5.13 to +5.89% |
| 12 | 10 | 4.79–5.21× | 23.6% | +1.96 to +2.26% | +0.27 to +0.99% |
| 12 | 40 | 2.02–2.19× | 23.6% | +3.57 to +3.87% | +0.74 to +1.46% |
| 6 | 5 | 7.23–7.86× | 34.6% | +6.77 to +7.07% | +7.78 to +8.54% |
| 6 | 10 | 6.03–6.57× | 34.6% | +10.02 to +10.33% | +2.11 to +2.79% |
| 6 | 40 | 3.04–3.30× | 34.6% | +9.71 to +10.03% | −2.18 to −1.48% |

All comparisons use the optimized full-decoder 40-step arm as the reference. Positive error changes are worse. Take medians across the three seeds in each scene, then form equal-weight scene ratios. The intervals reflect a missing baseline repeat. They are arithmetic bounds on the registered medians, since the median of three values with two observed values lies between the two observations. They describe the widest possible ratios given the missing measurement. Every cached variant exceeds 1% surface accuracy degradation even at its lowest possible ratio. [Per-run rows and failure receipts](partial_evidence.json)

## Compare with fewer full-decoder steps

Each cell is the median over the three seeds for that scene. Time is seconds and distance errors are mm. All cases in this comparison finished, including zero-step VGGT and the 5- and 10-step four-output controls.

| Scene | Method | Scene time | Peak allocated GiB | Accuracy error | Completeness error |
|---|---|---:|---:|---:|---:|
| Chess | Full 5 | 19.02 | 7.83 | 17.69 | 23.39 |
| Chess | Full 10 | 26.91 | 7.83 | 17.03 | 22.06 |
| Chess | Cached last 12, 10 updates | 17.93 | 5.98 | 17.90 | 23.70 |
| Fire | Full 5 | 18.64 | 7.83 | 15.80 | 19.26 |
| Fire | Full 10 | 27.30 | 7.83 | 15.66 | 18.86 |
| Fire | Cached last 12, 10 updates | 17.67 | 5.98 | 15.73 | 18.42 |
| Heads | Full 5 | 19.14 | 7.83 | 13.97 | 13.96 |
| Heads | Full 10 | 27.01 | 7.83 | 12.44 | 13.18 |
| Heads | Cached last 12, 10 updates | 17.51 | 5.98 | 11.80 | 11.63 |

Against **optimized full 5**, cached last-12 with ten updates was **6.5% faster** and reduced mean accuracy and completeness errors by **5.0% and 6.6%** across scenes. Heads supplied most of the error reduction. Chess worsened by **1.1% and 1.3%** on the two errors. Each of the full 5-step scene medians was slightly slower than the cached ten-step choice. Against **optimized full 10**, the cached candidate was **34.6% faster**, with **0.11% higher accuracy error** and **2.2% lower completeness error** on average. These are direct same-scene ratios with three valid repeats for every arm. The compact evidence also includes the original offloaded 5-, 10- and 40-step controls.

## Attribute the savings

At ten updates, using the cache with the same 12 trainable stages cut latency by **30.4%**. Using it with the same six trainable stages cut latency by **45.7%**. Peak allocated memory moved by less than 0.01% in both comparisons. Thus cache reuse explains a substantial portion of the runtime gain. The smaller trainable portion and staged residency together explain the memory reduction relative to full TCO. Zero-step and fixed-gradient checks matched bit for bit. Ordinary-loss runs also matched early outputs and loss values. Their backward passes showed known numerical variation, including between repeats of the same method. The fixed-gradient check isolates the cache implementation from that variation. Every benchmark timing and surface score used real TCO updates. [Engineering checks and real-loss variation](engineering.json)

## Coverage and replay

The default setup attempted all 135 registered jobs. It completed **128 jobs and 1,975 updates**, including all cached and uncached suffix runs. Eight attempts across seven full-decoder cases hit the 20-minute timeout, including two attempts on the Fire 40-step reference. An outlying completed run remains in the table. Successful launches occupied roughly 171 minutes and timed-out launches about 161 minutes. Those are launcher times and include overhead outside the primary per-scene timer. RunPod compute cost **USD 0**. This is a three-scene development result.

All seven missing controls subsequently finished on the same 3080 using a CUDA native allocator garbage collection threshold of 0.75. This changes runtime behavior, so the primary table uses only the default allocator evidence. A separate sensitivity analysis fills the seven gaps with the recovery measurements and independently replays all 135 cases. It reaches the same decision. In that analysis, the cached last-12 ten-step variant was **5.21× faster** with **23.6% lower peak allocated memory**, **1.96% higher accuracy error** and **0.95% higher completeness error**. The recovery setting was applied only to the seven timed-out jobs. The retest does not isolate the reason those jobs initially stalled. [Recovery receipts](allocator_diagnostics.json) · [Sensitivity decision](allocator_sensitivity_decision.json) · [PyTorch allocator documentation](https://docs.pytorch.org/docs/stable/notes/cuda)

Recompute the registered interval decision from Git on CPU.

```powershell
python scripts/bound_tco_frozen_prefix.py --evidence results/tco_frozen_prefix_v1/partial_evidence.json
python scripts/replay_tco_frozen_prefix.py --allocator-sensitivity
```

For local reproduction, verify the pinned environment and assets in the handoff. Run the engineering fixture, then use the registered serial launcher from the same repository. Use a fresh attempt ID for a failed job. Successful jobs with verified receipts are skipped.

```powershell
python scripts/check_tco_frozen_prefix.py
python scripts/run_tco_frozen_prefix_local.py
python scripts/run_tco_frozen_prefix_local.py --job fire_four_40_44 --attempt attempt02
python scripts/bound_tco_frozen_prefix.py --publish-incomplete
python scripts/diagnose_tco_frozen_allocator.py fire_four_40_44 attempt02
python scripts/diagnose_tco_frozen_allocator.py --summarize
```

Each remaining job can be launched individually in manifest order after a timeout. The recovery analysis uses the recorded failed attempt for each of the seven jobs. [Backup receipt](archive_receipt.json)
