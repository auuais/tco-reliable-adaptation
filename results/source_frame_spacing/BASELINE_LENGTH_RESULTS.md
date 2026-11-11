# Do camera baseline lengths explain the depth loss?

Supplying measured baseline lengths alongside exact rotation and translation direction reduced natural squared log-depth error by 3.99% versus the same exact pose directions with frozen lengths. The advantage over the ordinary frozen-camera update was only 0.77%. Gains over that frozen comparator were 0.04% on Heads, 0.54% on Chess and 1.75% on Fire. These are reference-assisted development measurements across 15 targets.

| Camera rotation and direction | Measured length information | Gain over unchanged depth | Gain over frozen camera update | Gain over same rig with original lengths |
|---|---|---:|---:|---:|
| Frozen | None | 1.60% | 0.00% | 0.00% |
| Frozen | Bundle scale | 1.21% | -0.30% | -0.30% |
| Frozen | Relative lengths | 0.60% | -1.11% | -1.11% |
| Frozen | Both | 1.46% | -0.07% | -0.07% |
| Exact | None | -2.41% | -4.64% | 0.00% |
| Exact | Bundle scale | 1.33% | -0.36% | 3.12% |
| Exact | Relative lengths | -2.23% | -4.39% | 0.12% |
| Exact | Both | 2.34% | 0.77% | 3.99% |

Most of the isolated recovery on the exact rig came from correcting the bundle scale. With frozen directions and rotations, inserting those same measured lengths produced no aggregate benefit over the original camera update. A measured camera rig includes privileged information, and every arm here uses the same DIS image matches and quarter-strength update. These results give little evidence for developing a baseline-length estimator alone. A stronger redesign would have to improve the image measurements or how the update weighs their uncertainty. MASt3R spending remains reserved.

[Registered comparison](../../docs/BASELINE_LENGTH_CONTROL.md) and [per-target results](baseline_lengths.json). The original frozen and joint-pose controls replayed for every target. Cloud cost was $0.
