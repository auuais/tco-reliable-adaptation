# Calibration aware LoRA: first development screen

**Neither proposed full-gradient variant qualified for advancement.** All 30 trained states and six frozen controls completed, their global prediction seal passed, and the registered held-source scorer selected no nominee. Its exact replay and a separately implemented compact arithmetic replay agree. This is a negative development result for the registered eight-step design, not proof that all calibration-aware adaptation methods fail.

The [goal](../../docs/CALIBRATION_AWARE_LORA_GOAL.md), [successor handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) and [live progress](progress.json) define the study. Original weights remained frozen, all cases have valid primary predictions, and no numerical failure occurred. No dense M3 confirmation, unseen-scene validation or rental qualification has been established.

Primary endpoint: fresh native KDP reconstruction followed by the common all-allowed-observation five-scale refinement. Lower MAE is better. Means weight the five views equally within each scene and the three scenes equally within each source direction. Positive gain means lower error than matched-D control P1.

| Arm | Method | Earlier-source adaptation MAE (mm) | Later-source adaptation MAE (mm) | Earlier gain vs P1 | Later gain vs P1 |
|---|---|---:|---:|---:|---:|
| P0 | Frozen model | 67.888878 | 66.656990 | -0.5752% | -0.0966% |
| P1 | Matched original D objective | 67.500586 | 66.592641 | 0% | 0% |
| P2 | Independent scales, detached solve | 67.395494 | 66.741443 | +0.1557% | -0.2235% |
| P3 | Independent scales, differentiated solve | 67.425538 | 66.630935 | +0.1112% | -0.0575% |
| P4 | Graph-coupled scales, differentiated solve | 67.746053 | 66.688241 | -0.3637% | -0.1436% |
| P5 | Graph-coupled scales, detached solve | 67.440577 | 66.773637 | +0.0889% | -0.2718% |

Only P3 and P4 were eligible candidates. The gate requires at least 1% improvement over P1 in **both** directions, improvement on at least two of the same three scenes in both directions, and no numerical failures. P3 improved Chess and Fire in both directions but regressed on Heads and failed both aggregate gain thresholds. P4 failed both aggregate thresholds and improved only Chess in both directions. Neither candidate may advance under these rules.

The full-versus-detached comparisons are inconsistent across directions. P3 versus P2 changes MAE by -0.0446% gain earlier and +0.1656% gain later; P4 versus P5 gives -0.4530% and +0.1279%. These small development differences do not establish a reliable calibration-gradient mechanism or repeatability. Passing derivative checks establishes implementation correctness, not accuracy benefit.

The 1,225 held-source observations span 30 scene/source/view combinations. These are three development scenes with two directions, not six independent scenes or a final generalization test. Both spatial support and query observations trained the adapters. The two sensor-source directions and existing supplied camera priors must remain explicit when describing the result.

Verified evidence:

- [Global 36-state prediction seal](prediction_seal.json), created before held-source scoring.
- [Complete scores and nomination reasons](held_source_scores.json), including per-view/per-scene values, native secondary endpoint and validity counts.
- [Independent compact arithmetic replay](independent_public_replay.json): 420 fields and 180 rows, exact eligibility/ranking/nominee agreement. It does not reconstruct raw overlap masks or prove a fresh model repeat.
- [Replay preparation](independent_replay_preparation01.json): 21 synthetic tests and fixed 1e-12 arithmetic tolerances recorded before real score access; its source was committed at `3255e93` before scoring.
- [Full prediction archive](backups/m2-20261118-verified01.json): 2,281 files, 6,059,018,880 bytes copied to the second physical disk and verified against the source. [Compact score/replay backup](backups/m2-scores-20261118-verified01.json) preserves the resulting verdict and code separately.

Sum of recorded per-state model runtimes: 9,123.026736 seconds (2.534 hours), excluding queue/verifier overhead. Additional rental compute remains USD 0. The local GPU queue is finished.

**Next branch:** this failed screen is archived. The [completed D0 diagnostic](../lora_profiled_d0_v1/README.md) and exact replay establish nonzero, substantially non-scalar raw-output changes, without establishing an accuracy gain. Implement the [selected D1 probe](../../manifests/lora_profiled_d1_design_v1.json) to measure their preservation through fixed confidence fusion, calibration and the training-to-inference transition. Follow the [predeclared surrogate diagnosis](../../docs/PROFILED_LORA_SURROGATE_DIAGNOSIS.md), using only permitted fitting information and the selected artifacts. Distinguish small effective LoRA changes, calibration absorbing the useful change, and changes lost during native reconstruction. No learning-rate sweep, alternate nominee or paid transfer is justified by this result. At most one evidence-based local redesign is permitted across the goal's failure branches; it must be registered before new training. The prepared M3 infrastructure remains conditional.

| Evidence | Verified result |
|---|---|
| Source preparation | 75 source/pose files, 30 packets reconstructed exactly, zero target-depth files opened |
| Spatial folds | 1,225 observations, minimum fold size 11; source-ray and native-cell grouping verified |
| Frozen graph | 60 directed cache files, 10 unordered edges per scene |
| Calibration calculus | CPU derivatives and final metric-gauge/coordinate fixtures pass |
| Real LoRA gradient bridge | gradient02: exact retained/replayed agreement, all three finite differences pass, original state unchanged |
| Full macro-step | 20 pairs, one optimizer update; 23.85 seconds, 4825.71 MiB peak allocated on the local 3080 |
| Endpoint bridge | endpoint01: pristine/zero-LoRA native/final geometry, matching and confidence exactly equal |
| Scientific release | 30 trained states and six frozen controls registered; first v1 P0 retired after a host filename-verifier repair, preserved unscored |
| Scoring preparation | Synthetic-only seal, metric, selection and failure-policy checks; primary final depth, secondary g0 times native depth |
| Registered development screen | 36 states complete; no eligible P3/P4 nominee; registered replay and independent compact arithmetic replay agree |
| Accuracy improvement | No qualifying improvement from this design; publication/generalization claim unsupported |
| Additional rental compute | USD 0 |

See the [engineering diagnosis](../../docs/PROFILED_LORA_ENGINEERING_DIAGNOSIS.md) for the original gradient failure, deterministic fix and Windows colon/U+F03A filename boundary. No acceptance tolerance was relaxed. The first P0's unchanged bytes were independently verified and [archived on C: and E:](archived_cases/scientific_v1_heads_earlier_P0_screen01.json). All 36 scored states used the same scientific-v2 registration; the retired v1 state was not included.

Portable CPU checks:

```sh
python scripts/prepare_profiled_inputs.py --verify
python scripts/check_profiled_splits.py --verify
python scripts/check_profiled_calibration.py --verify
python scripts/check_profiled_final_calibration.py --verify
python -m unittest discover -s scripts -p test_profiled_batch.py
python scripts/test_profiled_scoring.py
```

The canonical-fusion check needs the pinned MASt3R source checked out by CI. The historical matching-cache audit also needs excluded local files; its public hashes do not constitute a complete public reconstruction replay. The [evaluation rules](../../manifests/lora_profiled_evaluation_v2.json) and separate frozen evaluator registration prevent score access before the complete prediction seal.
