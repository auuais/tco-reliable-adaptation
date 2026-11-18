# Calibration aware LoRA implementation checkpoint

The [active goal](../../docs/CALIBRATION_AWARE_LORA_GOAL.md) and [successor handoff](../../docs/CALIBRATION_AWARE_LORA_HANDOFF.md) define the next experiments. Current preparation verifies the local runtime, input provenance, spatial folds, pretrained matching graph and differentiable geometry module. New model adaptation and accuracy runs remain pending. [Machine-readable progress](progress.json)

| Evidence | Result |
|---|---|
| Current checkpoint and three upstream revisions | Verified; every tracked blob matches the pinned source after documented CRLF normalization where required |
| Source reconstruction | 75 source/pose files, 30 packets reconstructed exactly, zero target-depth files opened |
| Sensor folds | 1,225 observations; minimum fold size 11; no source-ray or native-cell leakage |
| Frozen graph | 60 directed cache files audited; 10 unordered edges per scene |
| Geometry derivative | Full gradient -0.069362121, detached gradient -0.497390560; finite-difference discrepancy 1.08e-9 |
| Real model gradient/accuracy | Pending |
| Additional rental compute | USD 0 |

The first environment attempt rejected Windows CRLF checkout differences. The replacement verified exact Git-object content with explicit newline equivalence; upstream files were preserved. See [failure record](engineering_failures.json) and [environment evidence](environment.json).

Portable CPU checks:

```sh
python scripts/prepare_profiled_inputs.py --verify
python scripts/check_profiled_splits.py --verify
python scripts/check_profiled_calibration.py --verify
```

The calibration check also needs the pinned MASt3R `sparse_ga.py` source, which CI checks out for comparison with the actual upstream canonical-fusion function. Local `python scripts/profiled_baseline_graph.py --verify` additionally requires the excluded historical matching caches. Their hashes and provenance are public in [the graph audit](baseline_graph.json); this local cache audit is not a public reconstruction replay.
