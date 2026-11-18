# Comparing separate MASt3R decoder steps at equal alignment cost

**The split policy missed the registered development threshold.** Across three rooms from Stanford 12-Scenes, it tied the scalar policy for earlier observations and raised mean held depth error by 0.31% for later observations. Even the best choice made with hindsight from each set showed no split advantage. The experiment stopped at the development stage and Office 1 remains reserved. [Handoff and thresholds](../../docs/MAST3R_DECODER_GROUP_HANDOFF.md) · [Execution manifest](../../manifests/native_lora_decoder_groups_v1.json)

## Held sparse depth results

Mean absolute axial depth errors are in mm after native MASt3R alignment, with one metric scale per candidate fitted using adaptation support and query observations. The table weights rooms equally within each temporal direction. Positive gains mean lower error with split multipliers.

| Adaptation packet | Scalar selected | Split selected | Gain from split | Scalar hindsight best | Split hindsight best | Hindsight gain from split |
|---|---:|---:|---:|---:|---:|---:|
| Earlier | 70.324 | 70.324 | 0.00% | 70.324 | 70.324 | 0.00% |
| Later | 64.798 | 64.998 | −0.31% | 64.788 | 64.959 | −0.26% |

| Room | Adaptation packet | Scalar choice | Split choice (early, late) | Scalar error | Split error |
|---|---|---:|---:|---:|---:|
| apt1/living | Earlier | 0.50 | (0.50, 0.50) | 82.431 | 82.431 |
| apt1/living | Later | 0.75 | (0.50, 1.50) | 81.612 | 82.212 |
| apt2/kitchen | Earlier | 0.50 | (0.50, 0.50) | 54.166 | 54.166 |
| apt2/kitchen | Later | 1.00 | (1.00, 1.00) | 53.751 | 53.751 |
| office2/5b | Earlier | 1.50 | (1.50, 1.50) | 74.375 | 74.375 |
| office2/5b | Later | 0.50 | (0.50, 0.50) | 59.030 | 59.030 |

Each policy could choose from seven candidates and used the same previously computed first frozen-backbone LoRA update. The split policy traded two intermediate scalar choices for two asymmetric layer updates. Five choices agreed. In the sixth case, the scalar step of 0.75 beat both asymmetric candidates. Across all cases the expanded scalar selector lowered error relative to the previous four-choice selector by 0.94% for earlier observations and 0.71% for later observations. Both comparisons use the same development rooms. The 1.50 boundary won in one case, so the result applies to the tested grids. [Case scores and fixed controls](held_scores.json) · [Previous study](../native_lora_step_v1/README.md)

The two asymmetric candidates did change aligned depth. On selector pixels their median per-view RMS changes from the full step were 2.49 and 2.76 mm. Their median cosine similarities to the scalar depth displacement from 0.50 to 1.50 were −0.29 and −0.15. These descriptive values show distinct responses in this grid. Neither asymmetric candidate beat the best scalar choice on held depth in any case. Each of the three rooms supplied two paired observations over the same five RGB views. The experiment measures sparse depth on those views. Dense accuracy and independent environments remain open. [Selection and mechanism evidence](selection_seal.json)

## Verification and local resources

- All six cases scored all nine distinct candidates. Each fixed held view retained 53–64 measurements. Choices were sealed before held scoring. The independent replay verified 270 public held-view rows and 540 private selector and held-view rows, including their gauge and overlap mask.
- All five protocol fixtures passed inside the pinned image. GitHub's CPU workflow runs the four portable fixtures and the public replay. The real factor assignment fixture imports the full MASt3R runtime and runs in the pinned image.
- The first bridge case matched all four historical reconstructions, selector scores, and gauges exactly. The additional full-step repeat had zero difference across the metric-depth raster and the same held-view scores.
- The RTX 3080 rendered 34 candidates and one repeat locally. The other 20 endpoints were reused after verifying snapshot, model, trace and output digests. The six launches took 38.0 minutes total, including 28.8 minutes in fresh endpoint rendering and its checks. Peak allocation for a fresh endpoint was 3.62 GiB. All six containers stopped successfully on the first attempt. Local working evidence used 5.39 GB including matching caches. RunPod compute cost was **USD 0**.
- The verified second-disk snapshot includes the compact private evidence for replay. The public repository carries the registration, scores, and receipts. [CPU replay](independent_replay.json) · [Backup receipt](archive_receipt.json)

## Reproduce the decision

Run the public CPU replay in a checkout of the results commit.

```powershell
python scripts/replay_native_lora_decoder_groups.py
```

For a full local rerun, use a clean copy of registration commit `c46a908` and the pinned image and checkpoint. Restore the historical inputs, factors, and reconstructions from the preceding study and validate them against its seals. The launcher uses the pinned Docker engine and one GPU job at a time. The compact second-disk snapshot can support a private replay without rerendering.

```powershell
python scripts/run_native_lora_decoder_groups_local.py
python scripts/native_lora_decoder_groups.py seal
python scripts/native_lora_decoder_groups.py score
python scripts/replay_native_lora_decoder_groups.py --private
```
