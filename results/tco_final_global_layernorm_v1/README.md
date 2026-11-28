# Five-view TCO with pretrained final-block LayerNorm adaptation

The Stanford data support a violation of at least one registered requirement for LN10.

LN10 optimized 4,096 pretrained LayerNorm coefficients in the last global block alongside standard rank-4 LoRA. Each run started from a fresh VGGT checkpoint. L10 and L40 used standard LoRA for ten and forty updates. All three used five RGB views, camera checkpointing, joint RGB+ED rendering and an 8,960 MiB PyTorch budget. [Registered design](../../docs/TCO_FINAL_GLOBAL_LAYERNORM_HANDOFF.md) · [Weight and signed-gradient checks](qualification.json)

## Stanford fresh rooms

The paired comparison scored 54 of 54 slots. Lower errors and shorter times are better. Percent changes give equal weight to rooms. The one-sided upper bounds resample rooms within the two enrolled environments and paired seeds within rooms. They describe variation in these environments. [Room and seed results](primary_decision.json)

| Endpoint | LN10 vs L10 | Upper 95% bound | LN10 vs L40 | Upper 95% bound |
|---|---:|---:|---:|---:|
| Accuracy error | +2.83% | +5.17% | -1.54% | +3.60% |
| Completeness error | +3.02% | +5.83% | +10.36% | +14.41% |
| Scene time | +0.68% | +3.55% | -64.98% | -64.49% |
| Peak live GPU memory | +0.00% | +0.00% | +0.00% | +0.00% |
| Peak reserved GPU memory | +0.00% | +0.09% | -0.46% | +0.19% |

Absolute values are equal-weight means of the three-seed median for each complete room.

| Arm | Accuracy (mm) | Completeness (mm) | Scene time (s) | Peak live (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|
| L10 | 9.539 | 11.279 | 29.16 | 6,437.7 | 7,236.0 |
| LN10 | 9.814 | 11.582 | 29.35 | 6,437.8 | 7,236.0 |
| L40 | 9.799 | 10.495 | 83.80 | 6,437.8 | 7,269.3 |

The equal-room ratio of scene times corresponds to **2.86×** the forty-step processing speed. Both comparisons use full scene timing, including the final FP32 readout.

- Accuracy superiority over ten-step LoRA **observed violation uncertain**.
- Completeness within 3% of ten-step LoRA **observed violation uncertain**.
- Accuracy within 1% of forty-step LoRA **inconclusive**.
- Completeness within 3% of forty-step LoRA **supported violation**.
- Faster scene reconstruction than forty-step LoRA **pass**.

## Same-seed repeatability on Chess

3 of three prespecified duplicate pairs were scorable on Chess/seq-03 at seed 42. Each delta subtracts the original error from its duplicate in millimetres. Positive means worse on the repeat. [Development case scores](development_rows.json)

| Arm | Accuracy change | Completeness change |
|---|---:|---:|
| L10 | +0.477 mm (+2.88%) | +1.611 mm (+7.71%) |
| LN10 | +1.266 mm (+7.94%) | +2.788 mm (+13.98%) |
| L40 | -1.347 mm (-8.61%) | -1.753 mm (-9.10%) |

The ordering of LN10 and L10 reversed on accuracy and completeness across the two executions.
These pairs measure execution variability on one room and seed. The registered fresh-room estimates use their own original runs.

## Outcome-selected follow-up

The registered development follow-up tested a reduced LayerNorm learning rate. [Follow-up decisions](followup_decision.json)

| Endpoint | LN10_low vs L10 | Upper 95% bound | LN10_low vs L40 | Upper 95% bound |
|---|---:|---:|---:|---:|
| Accuracy error | -0.83% | +6.22% | -4.87% | +13.34% |
| Completeness error | -2.29% | +5.98% | -7.46% | +15.86% |
| Scene time | -4.85% | -0.79% | -66.21% | -65.10% |
| Peak live GPU memory | +0.00% | +0.00% | +0.00% | +0.00% |
| Peak reserved GPU memory | +0.03% | +0.45% | -0.55% | +0.00% |

Absolute values are equal-weight means of the three-seed median for each complete room.

| Arm | Accuracy (mm) | Completeness (mm) | Scene time (s) | Peak live (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|
| L10 | 14.546 | 17.225 | 30.26 | 6,437.7 | 7,214.0 |
| LN10_low | 14.433 | 16.842 | 28.79 | 6,437.8 | 7,216.0 |
| L40 | 15.088 | 17.943 | 85.22 | 6,437.8 | 7,256.0 |

The original-rate development diagnostics measured the joint L2 displacement of all four LayerNorm vectors from their pretrained values after ten updates. The clip count is the number of updates whose LayerNorm gradient norm exceeded 1 before clipping. [Paired model diagnostics](followup_development_diagnostics.json)

| Room | Seed | Final joint displacement | LayerNorm clips / updates |
|---|---:|---:|---:|
| chess/seq-03 | 42 | 0.00912793 | 0 / 10 |
| heads/seq-01 | 42 | 0.00834435 | 0 / 10 |

Across 6 matched development room–seed pairs, lowering the LayerNorm learning rate changed mean surface accuracy error by +0.0123 mm and completeness error by -0.1561 mm relative to the original rate. These are arithmetic means of paired differences. The formal development gate uses per-room seed medians.

## Independent ARKitScenes visits

The locked LN10_low candidate reached **supported violation** on ARKitScenes. [Independent paired decisions](replication_decision.json)

| Endpoint | LN10_low vs L10 | Upper 95% bound | LN10_low vs L40 | Upper 95% bound |
|---|---:|---:|---:|---:|
| Accuracy error | -0.80% | +2.87% | +4.39% | +9.30% |
| Completeness error | -0.57% | +2.81% | +6.09% | +12.18% |
| Scene time | +0.52% | +3.38% | -64.78% | -64.46% |
| Peak live GPU memory | +0.00% | +0.00% | +0.00% | +0.00% |
| Peak reserved GPU memory | -0.09% | +0.14% | -0.64% | -0.23% |

Absolute values are equal-weight means of the three-seed median for each complete room.

| Arm | Accuracy (mm) | Completeness (mm) | Scene time (s) | Peak live (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|
| L10 | 23.545 | 38.696 | 28.06 | 6,437.7 | 7,239.3 |
| LN10_low | 23.574 | 38.818 | 28.18 | 6,437.8 | 7,232.7 |
| L40 | 21.926 | 36.474 | 80.02 | 6,437.8 | 7,279.3 |

## Execution and reproducibility

On the local RTX 3080, 151/151 model slots completed across 151 attempts, including engineering controls and repeats. Cumulative model-attempt wall time was 3.28 hours. RunPod compute cost was **USD 0**. [Attempt ledger](final_attempt_evidence.json) · [Archive receipt](final_archive_receipt.json)

| Stage role | Registered | Completed | First attempt completed | Attempts |
|---|---:|---:|---:|---:|
| engineering | 8 | 8 | 8 | 8 |
| development | 18 | 18 | 18 | 18 |
| duplicate | 3 | 3 | 3 | 3 |
| primary | 54 | 54 | 54 | 54 |
| anchor | 6 | 6 | 6 | 6 |
| followup | 6 | 6 | 6 | 6 |
| diagnostics | 2 | 2 | 2 | 2 |
| replication | 54 | 54 | 54 | 54 |

Independent CPU scoring reloaded the saved FP32 predictions for **32 of 32** prespecified room–seed–arm slots. The largest absolute score difference across both directed errors was 1.1e-11 mm.

Replay the decisions on CPU from the repository root using the public files.

```powershell
python -B -S scripts/finalize_tco_final_ln.py verify
```

[Final verdict and evidence digests](final_verdict.json) · [Development](development_summary.md) · [Primary](primary_summary.md) · [Follow-up](followup_summary.md)
