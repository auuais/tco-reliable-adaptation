# Full-resolution TCO on the local RTX 3080

The local RTX 3080 finished **5 of 12** first attempts at five views, 518 × 392 pixels and 40 optimizer updates. The implementation staged the initial forward pass and kept selected projection weights in host memory. Every completed attempt peaked at **6.71 GiB of allocated CUDA memory**. Instrumented inference and adaptation took **87–126 seconds** in those five cases. This timing excludes setup and describes successful attempts only. [Registered local configuration](../../manifests/staged_sensor_factorial.json) · [Measured attempts](first_attempts.json)

| Sequence | Completed / 4 | Updates observed in timed-out attempts |
|---|---:|---|
| Heads, sequence 01 | 4 / 4 | — |
| Chess, sequence 03 | 0 / 4 | 21, 24, 37, 38 |
| Fire, sequence 03 | 1 / 4 | 14, 14, 23 |

Each of the seven interrupted attempts reached its 30-minute wall-clock limit. Its last recorded gradient covered all 384 adapter tensors. The partial table establishes local memory feasibility and an execution reliability limit on this machine. Ranking the four camera conditions requires successful runs for the missing cells. The recovery protocol allows each timeout one further attempt under the same settings. [Recovery protocol](../../manifests/staged_sensor_timeout_recovery.json)

The first attempts and their evaluation inputs occupy a 312 MB archive with 246 files, verified on two drives. Each row records the execution revision and working-tree flag from its original run. The separate RTX 3090 experiment used the original full-model execution path and completed all 12 calibration conditions. [Local archive hashes](first_attempts.json) · [Original TCO on the RTX 3090](../canonical_sensor_factorial/RESULTS.md)

### Final outcome after the registered retries

Three of the seven retries completed. Their initial predictions, adapter tensors, RNG state, evaluation inputs and priors matched their respective first attempts. Four runs timed out again. The final coverage was **8/12** conditions, with four completed on Heads, two on Chess and two on Fire. The other four have no 40-update endpoint. The complete factorial comparison therefore remains unavailable on this local execution path. Both drives hold the retry traces and terminal records. [Retry outcomes and archive checksum](registered_timeout_retries.json)
