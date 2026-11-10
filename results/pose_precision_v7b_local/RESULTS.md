# Pose-loss precision: interrupted local engineering comparison

The frozen stop-on-failure protocol stopped at job 9/29. Three zero controls and five full 40-step adaptations completed on the RTX 3080. The sixth adaptation (Heads, FP32 direct pose energy, seed 2026) recorded 23 loss rows before the 900-second limit. It produced no final prediction or evaluation. The other 20 jobs never started. Runtime logs establish the timeout, not its cause.

All completed adaptations were on Heads. The two complete seed pairs do not estimate the three-scene effect, and no planned same-seed repeat group was completed. **The study gate is not evaluable.** We neither impute outcomes nor recommend FP32 pose loss from this subset. No later allocation is released. The separate [40-run calibration study](../repeat_calibration_v6/RESULTS.md) is complete and unaffected.

The [machine-readable record](interrupted.json) includes each planned run's status, the available final metrics and prediction hashes, and the partial trace count. Source, initialization and input evidence for started jobs was checked against the frozen manifest. Running cost: zero cloud dollars. Any resumed comparison must preserve this timeout and declare its retry policy and identical output precision before examining new outcomes.
