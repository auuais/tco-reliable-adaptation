# MASt3R compensation redesign

The prospective inventory contains 36 fresh states across Heads, Chess and Fire, with both temporal adaptation directions. R1 penalizes the query residual change attributable to refitting the five calibration scales. R0 applies an isotropic support penalty with matching local curvature trace. R2 uses the same R1 forward objective and detaches the scales. P0, P1 and P3 are freshly reconstructed original controls. [Design](../../docs/PROFILED_COMPENSATION_REDESIGN.md) · [Executable inventory](../../manifests/lora_profiled_compensation_execution_v1.json) · [Scoring registration](../../manifests/lora_profiled_compensation_evaluation_v1.json)

The real-model bridge passed on the RTX 3080. The original teacher and zero-adapter outputs matched exactly. P3, R0 and R1 had identical loss and full 20-pair gradients at pristine initialization. The same saved nonzero adapter gave exact R1/R2 forward equality in both folds. Each arm’s two-pair retained and replayed gradients agreed exactly. All three FP32 parameter perturbations passed in each fold. The bridge used 184 decoder calls, zero optimizer updates and 5,319.49 MiB peak allocated CUDA memory. [Engineering report](engineering/bridge01/report.json) · [Verified stop receipt](launches/tco-profiled-compensation-bridge01/terminal.json)

Check the current [progress record](../lora_profiled_v1/progress.json) and local container state before resuming the queue. Completed cases are checked against their byte inventories and skipped automatically. The batch runner stops on a failed or partial case and retains its evidence. Use the same registered attempt rather than creating duplicates without diagnosis. All local compute uses the existing RTX 3080 and costs USD 0 in rental compute.

The six Heads/earlier states completed with 40 optimizer updates across the five trained states. P0, P1 and P3 reconstructed both historical endpoints byte for byte. P1 reproduced the final trained adapter and P3 reproduced every recorded adapter state. P3, R0 and R1 matched the initial loss, gradient and updated adapter bytes. R1 and R2 shared the initial forward computation, with the expected gradient difference. The six runs took about 29 minutes 37 seconds in total, with a peak of 4,722.25 MiB allocated CUDA memory. Their full evidence is verified on the second disk. The serial queue for the remaining cases has started. [First tranche evidence](heads_earlier_tranche01.json)

**After all cases complete**

Use the prospectively registered score path below. It verifies all endpoints and the first-step equivalence of P3, R0 and R1 across all six scene/source conditions. It saves the global prediction seal before any held-source prediction is compared with a measurement.

```powershell
python scripts/score_profiled_compensation.py seal
python scripts/score_profiled_compensation.py score
python scripts/score_profiled_compensation.py replay
python scripts/verify_compensation_public_scores.py --output results/lora_profiled_compensation_v1/independent_replay.json
```

Only R1 is eligible, with at least 1% lower held-source MAE than P1 in each temporal direction, at least two common improving scenes, and lower directional means than P3, R0 and R2 in both directions. A nomination would move the method to the registered surface and repeatability checks. The rental gate remains tied to those later checks.

Publish the score report, independent replay receipt, and aggregate arithmetic check. Keep the sampled measurement and prediction vectors and full endpoint arrays in the noncommercial local evidence archives. The public arithmetic check starts from the 180 reported view means and verifies the paired-room and directional nomination without those arrays. [Related work and comparison scope](../../docs/PROFILED_COMPENSATION_RELATED_WORK.md)

```powershell
python scripts/replay_compensation_published_metrics.py
python scripts/plot_compensation_verdict.py --output results/lora_profiled_compensation_v1/held_source_gains.svg
python scripts/archive_profiled_compensation.py copy --destination E:/ResearchData/TCO/research_results/lora_profiled_compensation_v1 --snapshot verdict-v1 --verdict --receipt results/lora_profiled_compensation_v1/verdict_backup01.json
```
