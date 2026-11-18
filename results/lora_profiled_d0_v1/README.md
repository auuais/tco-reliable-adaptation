# Saved artifact diagnosis of the failed LoRA screen

D0 confirms that the adapters changed the model and its raw depth predictions. The selected full-gradient P3 case is **not merely a scalar rescaling** at the raw-output level: its mean per-pair shape-energy share is 68.47%. This does not establish useful shape correction, actual calibration cancellation, or improved reconstruction. The original 36-state accuracy screen remains negative.

The [prospective design](../../manifests/lora_profiled_diagnosis_design_v1.json) and [executable registration](../../manifests/lora_profiled_diagnosis_execution_v1.json) were frozen before diagnostic tensor access. Code and executable registration were committed at `9e059a48293dc3aee692e470be7a8a8e768d17a7`. The [66-test CPU acceptance](synthetic_checks02.json) and independent source/math review preceded execution. The earlier 63-test receipt is preserved as preparation history; its source predates the final launcher cleanup checks.

**Measured results**

All 30 adapters contain the expected 144 adapted matrices. Their aggregate effective-weight changes range from **0.02013% to 0.03197%** of the corresponding frozen-matrix norm. The calculation is `DeltaW=(1/4) B A`, not the factor-vector norm. Both numerator and denominator aggregate only the adapted matrices.

The fixed functional case was Heads/earlier, chosen by registration order before D0, with all 20 directed pairs. The table uses the fixed 768-pixel grid in each pair's first camera. Each statistic is computed per pair and then averaged equally; energy shares are means of per-pair ratios, not a pooled global energy fraction.

| Arm | Effective weight change | Raw depth change | Scalar-energy share | Remaining shape-energy share |
|---|---:|---:|---:|---:|
| P1 matched D | 0.03197% | 0.52028% | 66.04% | 33.96% |
| P2 detached scales | 0.02302% | 0.28831% | 34.58% | 65.42% |
| P3 differentiated scales | 0.02344% | 0.29057% | 31.53% | 68.47% |
| P0 same-state raw comparator | 0% | 0% | Undefined at zero change | Undefined at zero change |

Raw depth change means `||d1-d0||_2 / ||d0||_2`. Values are in the raw MASt3R self-camera axial model gauge, not meters or accuracy errors. Scalar decomposition uses `a=(d0^T delta)/(d0^T d0)` and `delta_shape=delta-a*d0`; it never modifies a prediction used for accuracy evaluation.

At the permitted adaptation coordinates, the P3 relative change is 0.31177%, with 30.69% scalar and 69.31% remaining shape energy. The coordinate tables remain separate from the fixed image grid. No selected value is nonfinite or nonpositive. No pixel was dropped to obtain these results.

The two frozen P0 raw-forward caches agree exactly at the declared coordinates. Their different supplied priors do not make their native reconstructions repeat controls. This is one fixed-state forward comparison, not a general estimate of model or optimizer variation.

**What the evidence resolves**

| Question | Finding | Interpretation |
|---|---|---|
| Did factors produce a nonzero effective matrix update? | Yes, all declared adapters were measured as scaled BA products | Factor reparameterization cannot explain away the reported matrix changes |
| Did raw geometry change? | Yes; P3 mean relative change is 0.29057%, against zero sampled P0 difference | The ineffective screen was not a complete absence of raw functional change |
| Is the P3 raw change purely scalar? | No; 68.47% mean per-pair energy remains after scalar removal | A pure raw-depth rescaling explanation is insufficient |
| Did the fitted scales actually cancel a useful update? | Not measured in D0 | Raw scalar/shape decomposition is not the fitted solver response |
| Did a useful training change survive fresh inference and native reconstruction? | Not established | D0 uses fresh FP32 raw outputs; it does not evaluate the BF16 training surrogate or native endpoint |
| Has reconstruction accuracy improved? | No qualifying improvement established | The registered M2 nomination remains null |

**Decision**

Use the [selected bounded D1 probe](../../manifests/lora_profiled_d1_design_v1.json) to measure the actual calibration response and compare the training inference path with the saved fresh FP32 path under identical frozen confidence fusion. No learning-rate/rank/step sweep, redesigned training or rental transfer is released. D0 has consumed zero scientific redesigns. D1 requires separate code, synthetic acceptance and an executable registration before its additional data/model access.

**Verification and resources**

- [Complete report](attempts/d0_01/report.json), with all 30 layerwise adapter tables, all pairwise functional results and invalidity counts.
- Independent representative checks in that report use a materialized BA norm and a separate least-squares scalar fit; both agree with the primary calculations.
- [Exact same-runtime replay](verification02.json) passed. The first replay was interrupted by the user pause before final verification; its started/terminal records remain preserved. Replay does not constitute fresh model inference or independent accuracy verification.
- [Source-layout receipt](source_layout.json) documents why only the first tuple branch supplies self-camera axial depth.
- The run used 443.39 seconds wall time, 98.70 seconds process CPU time and approximately 915.79 MiB peak host RSS. There were zero model forwards, zero optimizer updates, no GPU request and USD 0 rental compute.
- Full raw inputs remain in the verified C:/E: M2 archive. The [D0 result/source backup](backup01.json) preserves the complete diagnosis and D1 preparation on the second physical disk; retain the immutable partial receipts as well as the final report.

Local reproduction uses the already registered entry points, fresh container names and fresh attempt/receipt paths. Do not overwrite `d0_01` or the execution registration:

```powershell
python scripts/run_profiled_d0_local.py verify --name tco-m3-d0-verify-FRESH --attempt d0_01 --output results/lora_profiled_d0_v1/verification-FRESH.json
```

Replace `FRESH` with a new lowercase numeric suffix; the launcher accepts only the declared container-name pattern. A full replay needs the local checkpoint and sealed raw caches. Public CI exercises the synthetic implementation boundary, while the original screen's compact accuracy arithmetic is publicly replayable.
