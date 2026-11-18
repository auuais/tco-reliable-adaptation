# Calibration aware LoRA implementation checkpoint

The [goal](../../docs/CALIBRATION_AWARE_LORA_GOAL.md), [successor handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) and [live progress](progress.json) define the study. Preparation and real-model engineering acceptance have passed. Scientific accuracy remains untested; all predictions must be sealed before development scoring.

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
| Accuracy improvement | Not established |
| Additional rental compute | USD 0 |

See the [engineering diagnosis](../../docs/PROFILED_LORA_ENGINEERING_DIAGNOSIS.md) for the original gradient failure, deterministic fix and Windows colon/U+F03A filename boundary. No acceptance tolerance was relaxed. The first P0's unchanged bytes were independently verified and [archived on C: and E:](archived_cases/scientific_v1_heads_earlier_P0_screen01.json). All 36 states will use one revised scientific registration.

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
