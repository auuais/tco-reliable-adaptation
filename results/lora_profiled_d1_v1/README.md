# D1 calibration and inference-path preservation

The MASt3R adapter reduced calibration query loss in both spatial folds on both inference paths. Refitting scales reduced the norm of the training-path residual change by about 29–31%. The saved FP32 path retained about 98–99% of the fixed-scale change. The gain on this adaptation objective survived the inference transition. The original held-source nomination remains unchanged. [M2 results](../lora_profiled_v1/README.md)

The probe used only Heads/earlier and the original five views. B0 and B1 use the historical BF16 training decoder with FP32 heads, while F0 and F1 use the saved FP32 inference pointmaps. Both paths use the same reproduced pristine confidence during fusion. Zero and one denote pristine and final P3. The exact teacher depth and confidence rasters matched all archived pixels. Both folds passed the gain, normalizer, ridge and count checks before final adapter access. [Teacher gate](attempts/d1_01/teacher_gate.json)

| Contrast | Query loss reduction, fold 0 | Query loss reduction, fold 1 | Residual change retained, fold 0 | Residual change retained, fold 1 |
|---|---:|---:|---:|---:|
| B0 → B1 | 0.9994% | 0.0098% | 71.14% | 68.77% |
| F0 → F1 | 6.2092% | 0.9686% | 98.84% | 97.76% |
| B0 → F0 | −1.5397% | −1.2103% | 104.05% | 100.10% |
| B1 → F1 | 3.8037% | −0.2398% | 99.81% | 100.59% |

Loss reduction is \((J_0-J_1)/J_0\). Residual change retained is the registered ratio \(\|t+c\|/\|t\|\), where \(t\) is the fixed-scale change and \(c\) is the compensation from refitting. The ratio describes direction and magnitude at the two fitted states, and does not turn compensation into a causal accuracy claim.

On the fixed image grid, the mean relative fused depth change was 0.2288% for B0 → B1 and 0.2363% for F0 → F1. The mean non-scalar share of fused-change energy was 51.48% and 76.69%, respectively. These depth measurements use raw model units. For the B1 output direction, the full and same-state detached loss derivatives were −0.255925 and −0.255881 in fold 0, and −0.001407 and −0.080198 in fold 1. All finite-difference acceptance rules passed. The fold dependence motivates the single controlled compensation redesign. [Full report](attempts/d1_01/report.json) · [Redesign specification](../../docs/PROFILED_COMPENSATION_REDESIGN.md)

The local RTX 3080 made five encoder calls and 40 decoder calls, with zero backward calls or updates. Peak CUDA memory was 3,788.34 MiB allocated and 4,136 MiB reserved. The model stage took about 5m 49s, including the parent byte audit, and capture itself took 50.35s. Both CPU passes took 16.73s of combined in-container analysis time. Both the exact fresh CPU replay and the separate diagonal ridge calculation passed. The verified second-disk backup contains 161 files and 80,601,429 bytes. Rental compute cost was USD 0. [Model seal](attempts/d1_01/model_seal.json) · [Exact replay](attempts/d1_01/replay.json) · [Independent arithmetic](attempts/d1_01/independent_arithmetic.json) · [Backup](execution_backup01.json)

The public checkout replays the eight fold/contrast matrix rows without model weights or the private full depth rasters.

```powershell
python scripts/verify_profiled_calibration_secant.py --attempt d1_01 --verify-only
```

## Preparation record

This section preserves the preparation checkpoint before the model diagnostic. This stage follows the negative development screen and the [verified D0 findings](../lora_profiled_d0_v1/README.md). It asks whether the observed P3 geometry change survives fixed confidence fusion, calibration and the change from training inference to saved FP32 inference. It does not train another method or establish an accuracy improvement.

The [prospective design](../../manifests/lora_profiled_d1_design_v1.json) remains unchanged. It permits one Heads/earlier probe, both original spatial folds, five encoder computations and forty decoder calls, with zero optimizer updates and zero rental compute. The original weights and loaded final adapter must remain frozen.

Completed preparation:

| Component | Synthetic tests | Evidence |
|---|---:|---|
| Float64 finite/directional calibration response | 32 | Exact identities, ridge behavior, finite differences and interpretation floors |
| Input byte inventory and strict member loaders | 16 | Role/schema/mutation checks, forbidden member access, source remapping and safe cache loading |
| Fixed sensor-only calibration and query matrices | 16 | Graph-zero equivalence, original residual agreement, both folds and fixed preparation |
| Teacher-first capture and preprocessing | 20 | Teacher rejection stops final-factor access, 5/20/20 call limits, state integrity and shared confidence |

All **84 tests passed in the pinned CPU image**, with no skips; see [preparation01.json](preparation01.json). The [selected actual input inventory](inputs_preparation01.json) also passed [byte and JSON provenance checks](inventory_checks01.json). This check hashed real input files without decoding production arrays. It did not repeat the full parent seal or authorize model execution. The CPU containers exited successfully and were verified stopped; their launch records are retained under `launches/`.

The original 32-test receipt remains as preparation history. New combined acceptance binds all eight current helper/test source files. An independent read-only review found no concrete blocker in the artifact, calibration or capture helpers. See the [preparation backup](preparation_backup01.json) for the verified second-disk copy.

The preparation checkpoint recorded these implementation steps:

1. Implement and test the release/run/replay controller and bounded local launcher, including partial call telemetry on failures, exclusive attempts, immutable output seals and verified shutdown.
2. Freeze and commit the complete executable source/runtime/input registration. Reuse these helpers; do not rerun M2 or D0.
3. Run the local forty-forward diagnostic. Require archived teacher agreement before loading final P3 factors or FP32 comparison caches. Keep all four states on the same pristine-confidence CUDA FP32 fusion path.
4. Preserve both folds and all four registered contrasts, replay the sealed CPU matrices, back up the evidence, then interpret the result before choosing at most one scientific redesign.

The [execution handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) records current continuation steps.
