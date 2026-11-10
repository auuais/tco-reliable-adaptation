# Equal-length RTX 3080 comparison on Heads

Both executions used the same five images, camera priors, seed, initial predictions and adapters, first scalar loss, and 40 upstream TCO updates. Staged execution finished in **89.33 seconds** versus **624.82 seconds** for the original implementation. Peak allocated CUDA memory fell from **8.63 to 6.71 GiB** and peak reserved memory from **10.57 to 8.25 GiB**. These instrumented times exclude model loading and include the same initial capture. The single pair comes from one Windows/WSL workstation. [Matched starting state and timing records](seal.json)

| Reference | Execution | Accuracy error (cm) | Completeness error (cm) |
|---|---|---:|---:|
| Released | Original | 1.20 | 1.22 |
| Released | Staged | 1.51 | 1.73 |
| Registered RGB | Original | 1.12 | 1.10 |
| Registered RGB | Staged | 1.52 | 1.65 |

Lower distance is better. Against the released reference, the staged endpoint's accuracy error rose **26.3%** and its completeness error rose **41.4%**. Against the registered reference, the respective increases were **35.8%** and **50.8%**. Mean normal consistency changed by +0.26% and −0.82% relative to the original on the respective references. The matched execution demonstrates lower measured memory and wall time. Its reconstruction did not maintain the original run's distance scores. [Four dense evaluations](summary.json)

The trajectories diverged after their shared first loss. One run per implementation cannot distinguish a storage or precision effect from ordinary CUDA adaptation variability. The original implementation completed only this one full window in the twelve-cell protocol. All four camera conditions in the larger staged factorial use the same implementation. Transfer still requires evidence of a reliable method gain. [Comparison protocol](../../manifests/staged_heads_bridge.json) · [Original execution outcomes](../sensor_tco_v17/RESULTS.md) · [Separate twelve-cell design](../../manifests/staged_sensor_factorial.json)
