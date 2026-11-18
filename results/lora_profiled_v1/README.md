# Calibration aware LoRA implementation checkpoint

The [active goal](../../docs/CALIBRATION_AWARE_LORA_GOAL.md) and [successor handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) define the next experiments. Preparation verifies the local runtime, input provenance, spatial folds, pretrained matching graph and differentiable geometry module. The first real-model engineering attempt failed gradient replay acceptance after passing zero-adapter output equality. Scientific adaptation and accuracy runs remain pending. [Machine-readable progress](progress.json)

| Evidence | Result |
|---|---|
| Current checkpoint and three upstream revisions | Verified; every tracked blob matches the pinned source after documented CRLF normalization where required |
| Source reconstruction | 75 source/pose files, 30 packets reconstructed exactly, zero target-depth files opened |
| Sensor folds | 1,225 observations; minimum fold size 11; no source-ray or native-cell leakage |
| Frozen graph | 60 directed cache files audited; 10 unordered edges per scene |
| Geometry derivative | Full gradient -0.069362121, detached gradient -0.497390560; finite-difference discrepancy 1.08e-9 |
| Pristine versus zero-LoRA outputs | All 20 directed pairs equal in gradient01 |
| Real model retained/replay gradients | Failed: relative L2 0.005883 versus maximum 0.0001; equal forward losses are insufficient |
| Real adapter finite differences, full macro-step, endpoint bridge | Not yet passed; first gradient attempt stopped before the remaining checks |
| Scientific states / accuracy improvement | 0 of 36; none established |
| Additional rental compute | USD 0 |

The first environment attempt rejected Windows CRLF checkout differences. The replacement verified exact Git-object content with explicit newline equivalence; upstream files were preserved. See [failure record](engineering_failures.json) and [environment evidence](environment.json).

The no-update arithmetic diagnostic and single-case runner are implemented but unverified on actual execution. Engineering v1 remains immutable; after diagnosing the [failed replay comparison](engineering/gradient01/replay_failure.json), register revised source in a new version and pass both model bridges before scientific release. The [evaluation specification](../../manifests/lora_profiled_evaluation_v2.json) freezes masks, candidate eligibility and decision rules before scores; the global seal/scorer still needs implementation.

Portable CPU checks:

```sh
python scripts/prepare_profiled_inputs.py --verify
python scripts/check_profiled_splits.py --verify
python scripts/check_profiled_calibration.py --verify
python scripts/check_profiled_final_calibration.py --verify
```

The calibration check also needs the pinned MASt3R `sparse_ga.py` source, which CI checks out for comparison with the actual upstream canonical-fusion function. Local `python scripts/profiled_baseline_graph.py --verify` additionally requires the excluded historical matching caches. Their hashes and provenance are public in [the graph audit](baseline_graph.json); this local cache audit is not a public reconstruction replay.
