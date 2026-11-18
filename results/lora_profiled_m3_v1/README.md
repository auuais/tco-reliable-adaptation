# M3 preparation evidence

This directory contains input and CPU implementation checks, not dense accuracy results. The unchanged M2 screen is still running. M3 model execution remains conditional on the registered M2 nomination rule, a separate executable M3 registration and endpoint acceptance.

Use [input plan v2](../../manifests/lora_profiled_m3_inputs_v2.json) and attempt `inputs02`. The [preparation contract](../../docs/PROFILED_LORA_M3_PREPARATION.md) is hash-bound by that plan; changing its contents requires explicit versioning. Original M2 sources and centered packets/folds remain unchanged.

| Verified preparation | Evidence |
|---|---|
| Earlier-only shifted grid | 624 selected observations; minimum fold size 15; exactly 45 allowed original files verified |
| Later-only shifted grid | 661 selected observations; minimum fold size 17; exactly 45 allowed original files verified |
| Source reconstruction | All 15 existing centered packets per direction reproduced exactly; zero target-depth files opened |
| Pose stress | Fixed five-degree / 2%-diameter supplied-pose perturbation; identical matrices across all three repeats |
| CPU preparation fixtures | 20 checks, recorded in `input_checks_v2.json` |
| Tensor case bridge | 24 cases checked; all six centered/clean cases exactly match the original M2 loader, including K, cameras, observations, folds and artifact hashes |
| Endpoint code comparison | M3 capture body has the same AST as M2; the selected original-prior-row provider is isolated in the new module |
| Model/dense accuracy | Not tested by this preparation |

The first `inputs01` attempt and input plan v1 are preserved. Its new validation incorrectly demanded nearly ideal orthogonal rotations; the pinned source matrices have small rounding/calibration deviations. Version 2 preserves those matrices and requires finite, invertible, right-handed camera transforms while reporting their orthogonality defects. No projection, model objective or source selection was changed. This was an engineering correction, not the single permitted scientific redesign.

The failed v1 source was reconstructed afterward from the recorded edit and exactly matches the digest already stored in the original v1 plan. Its source text and explicit reconstruction receipt are under `inputs01/historical_source`; this is a verified reconstruction, not a claim of an earlier Git commit or independent source snapshot.

The five compact derived NPZ archives under `inputs02` contain only signed sparse observations, grouped folds and supplied camera packets. They total 88,489 bytes. Original image/depth assets and model checkpoints are excluded. `tensor_bridge_checks.json` is a CPU input-equivalence report, not proof of a real M3 endpoint run.

Replay checks from the repository root:

```powershell
python scripts/check_profiled_m3_inputs.py
python scripts/check_profiled_m3_network.py
python -m unittest discover -s scripts -p 'test_profiled_m3_*.py'
```

The first two commands print checks without overwriting evidence. New output reports require a fresh path. Do not rerun a preparation command against an existing attempt.

`prepare_profiled_m3_inputs.load_prepared_case(plan, "inputs02", scene, side, grid, pose_condition)` returns a verified CPU packet. `profiled_m3_network.load_case(..., device="cpu")` converts it into the original network case schema; the eventual registered GPU runner may select CUDA. Its original-row loader detects input-file or fitting-tensor mutation before native KDP. The common final refinement still uses optimized native cameras.

`profiled_m3_gate.evaluate(registration, packet)` is a pure aggregate calculation over an entire 360-state, 720-reference-row inventory. Its 26 synthetic tests cover complete coverage, paired identities, fixed repetition-zero control selection, exact threshold boundaries, invalid inputs and mechanism comparisons. The three latest tests enforce identical masks and validity counts across both references, matching the existing preparation contract and early-stage gate. It explicitly does not authorize score access or rental release. The 18 network-adapter fixtures use synthetic data and a mocked endpoint to check routing, provenance and mutations; they are not reconstruction experiments.

The new `profiled_m3_points.py` reference-free converter passed 19 synthetic tests on the host and pinned CPU container (`points_cpu_checks.json`). It converts both endpoint conventions to the dense raster and counts fixed-mask validity without filtering. It has not decoded real M3 endpoints or references. The new `profiled_m3_inventory.py` passed 18 fixtures covering the 360-state plan, 30 conditional reuse candidates, 23 serial barriers and failure accounting. Its command descriptions are not an executable launcher or byte verifier. All 78 M3 unit tests passed together.

`runtime_checks/comparison02.json` records two fresh CPU-only processes with byte-identical synthetic dense-kernel reports. Torch intraop/interops 1/1, OpenCV 1 and all four OMP/BLAS environment thread values 1 resolved the tiny ICP roundoff seen in the preserved two-thread probes. The kernel and original cKDTree workers=-1 are unchanged. This accepts the synthetic examples on the pinned runtime; it does not prove universal determinism, real-study accuracy or model repeatability. See the [kernel audit](../../docs/PROFILED_LORA_M3_KERNEL_AUDIT.md) for exact conversion, masks, camera conventions, alignment, metrics and required registration.

The later evaluator preparation adds these completed components:

| Component | Evidence and limits |
|---|---|
| Dense leaf | `profiled_m3_dense.py`, 16 unit fixtures; `dense_leaf_checks/comparison01.json` verifies two fresh pinned CPU processes and exact legacy equivalence on four synthetic cases, including large outliers and undefined reference holes |
| Boundary and metric-depth diagnostics | `profiled_m3_diagnostics.py`, 22 fixtures; consumes exact kernel-aligned points without fitting, preserves common masks, reports primary raw metric axial error and explicit unsupported alternate axial convention |
| Combined conversion/evaluation/diagnostics | `dense_integration_checks/comparison01.json`; two byte-identical fresh-process reports using production raster shapes, a compact generated mask and both endpoints; includes nonpositive-depth accounting without filtering. This is not a full-mask runtime benchmark |
| Reference preparation | [Reference plan v1](../../manifests/lora_profiled_m3_references_v1.json); six existing RGB/reference archive hashes and 15 original pose text files verified, historical reference-camera algebra reproduced. Zero image/depth/prediction/reference archives decoded; mask semantic checks and current reprojection remain gated |
| Economical early-cell rejection | `profiled_m3_stage_gate.py`, 18 fixtures; complete paired rows for available repetitions, immutable r0 control, exact agreement with corresponding full-gate decisions. R0/R1 cannot confirm the full-repeat mechanism or pass the study |

The separate early-stage registration schema omits the future global prediction-seal field; it binds fixed policy/inputs/runtime and takes completed-stage seals separately. A descriptor is not a verified seal. The future controller must establish the bytes and release order, including sealing both initial directions before their first stage review. The reference preparation is likewise not a score release. Its primary camera digest uses the runtime probe's shape/dtype encoding; diagnostics carries an explicitly named compatible digest in addition to its converter-style mask/camera digest. Do not compare hashes produced by different encodings.

Still required if M2 nominates a candidate: a real M3 runner and versioned executable registration, actual centered/clean endpoint acceptance, byte-verifying controller/stage seals and scorer wrapper, exact shared-mask semantic and reference-camera projection checks after authorized decode, actual paired repeats, and the final gate. The verified leaf modules are components, not this complete scientific pipeline. Do not weaken the complete-inventory gate to make an incomplete study pass. No M3 GPU job or rental has been launched. The [successor handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) specifies implementation order and outcome-dependent actions.
