# One fresh-process MASt3R numerical repeat

The fixed `heads_centered_J` case was repeated in a fresh process using the same configured seed, 40 updates and three final alignments. The original base-state hash, initial gauges/support and initial scalar loss match. All 288 final adapter tensors and all nine native output arrays differ; all native fields also fail the registered rtol 0.00001 / atol 0.000001 check. These failed flags are preserved in result.json and independently recomputed by the public replay.

The initial gradient norms already differ slightly (0.45347884 versus 0.45351410). The largest final adapter-element difference is 0.00235917. The cause has not been isolated. Identical captured RNG states were not demonstrated. Raw camera-matrix differences are not a reconstruction-quality measure because the global gauge can change.

All three repeat point maps were sealed before their six dense-reference evaluations. The table reports the mean of the relative changes in accuracy and completeness; positive means the repeat had higher error. It is descriptive and differs from the primary study's normalization to ordinary MASt3R.

| Final alignment | Registered RGB reference | Released reference |
|---|---:|---:|
| Ordinary | +0.1399% | +0.2663% |
| KD | -0.1504% | -0.0685% |
| KDP-clean | -0.2720% | -0.0327% |

Individual distance-component changes span -0.3651% to +0.3817% across these six comparisons. See metric_comparison.json for all original/repeat metrics and per-component differences. This one paired execution on one development case is not a confidence interval, a population repeatability estimate, or a bound on other scenes or corruption conditions. It does not rescue any failed primary advancement gate.

Both executions are retained. Repeat native outputs, adapters and support are public under outputs; final forwards, correspondence/canonical caches, full point maps and evaluation records are backed up on C and E physical disks. No additional rental compute was used. This repeat is separate from the primary study's 24 states, 960 updates and 105 reconstructions.

```sh
python scripts/replay_mast3r_neural_repeat.py
python scripts/score_mast3r_neural_repeat.py verify
```

These validate saved arrays, identities, comparison flags, archive inventories and six paired metrics. They do not independently rerun training or implement a separate dense evaluator. A full rerun additionally requires the pinned checkpoint, upstream code and public-data preparation.
