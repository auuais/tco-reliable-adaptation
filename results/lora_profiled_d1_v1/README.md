# D1 preparation: calibration and inference-path preservation

**Preparation passed; the actual model diagnostic has not run.** This stage follows the negative development screen and the [verified D0 findings](../lora_profiled_d0_v1/README.md). It asks whether the observed P3 geometry change survives fixed confidence fusion, calibration and the change from training inference to saved FP32 inference. It does not train another method or establish an accuracy improvement.

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

Still required before interpreting a result:

1. Implement and test the release/run/replay controller and bounded local launcher, including partial call telemetry on failures, exclusive attempts, immutable output seals and verified shutdown.
2. Freeze and commit the complete executable source/runtime/input registration. Reuse these helpers; do not rerun M2 or D0.
3. Run the local forty-forward diagnostic. Require archived teacher agreement before loading final P3 factors or FP32 comparison caches. Keep all four states on the same pristine-confidence CUDA FP32 fusion path.
4. Preserve both folds and all four registered contrasts, replay the sealed CPU matrices, back up the evidence, then interpret the result before choosing at most one scientific redesign.

The [execution handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) gives interfaces and remaining requirements. No redesigned training, dense confirmation or paid transfer has been released. Whole-milestone progress remains 50%; this preparation is additional work inside the unfinished diagnosis milestone.
