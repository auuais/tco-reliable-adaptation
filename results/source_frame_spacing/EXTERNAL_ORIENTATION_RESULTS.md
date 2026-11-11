# External orientation experiment

A positive percentage means lower squared log-depth error. The primary condition used one degree of simulated absolute orientation error and averaged three seeds on 15 development targets.

| Scene | Gain over unchanged depth | Gain over frozen cameras |
|---|---:|---:|
| Overall | -1.41% | -3.34% |
| heads/seq-01 | -1.18% | -4.35% |
| chess/seq-03 | -1.15% | -0.64% |
| fire/seq-03 | -1.89% | -5.04% |

The registered criterion for a local surface evaluation failed.

This experiment measures axial depth on development scenes with simulated orientation input. Whole-surface improvement and performance with a real IMU require separate evaluation. Cloud compute cost was $0.
