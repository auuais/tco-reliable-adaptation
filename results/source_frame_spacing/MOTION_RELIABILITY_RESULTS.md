# RGB reliability weighting on development targets

The four fitted step sizes reduced squared log-depth error by 3.35% relative to the strongest aggregate comparator (the fixed quarter step). Gains were 4.50% on Heads, 4.21% on Chess and 1.30% on Fire. Each prediction used weights fitted on the other two scenes. Across all 15 targets, the weighted rule reduced error by 4.90% relative to leaving the original target depths unchanged. This passes the registered 2% depth criterion for a local surface comparison.

The estimator groups RGB triangulations by correction direction and two-sided support. The direction and support are available before scoring. Training uses reference depths in the other development scenes. These three scenes have informed method development; cross-fitting is an internal check and does not establish independent generalization. The next test must measure accuracy and completeness on the original scene surfaces with both reference conventions and same-input controls. The transfer allocation is still reserved.

[Per-target results, weights and bounds](motion_reliability_weights.json) and [registered design](../../docs/MOTION_RELIABILITY_WEIGHTS.md).
