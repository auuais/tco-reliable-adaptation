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

`profiled_m3_gate.evaluate(registration, packet)` is a pure aggregate calculation over an entire 360-state, 720-reference-row inventory. Its 23 synthetic tests cover complete coverage, paired identities, fixed repetition-zero control selection, exact threshold boundaries, invalid inputs and mechanism comparisons. It explicitly does not authorize score access or rental release. The 18 network-adapter fixtures use synthetic data and a mocked endpoint to check routing, provenance and mutations; they are not reconstruction experiments.

Still required if M2 nominates a candidate: a versioned M3 execution inventory, real centered/clean endpoint acceptance, byte-level prediction seals, dense point conversion/evaluation with both references and fixed masks, a separate non-authorizing early-cell failure check for economical stopping, the full paired repeat inventory, and the final gate. Do not weaken the complete-inventory gate to make an incomplete study pass. No M3 GPU job or rental has been launched.
