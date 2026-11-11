# The model-only focal check repeated on new frames

On 15 different frames from the same three 7-Scenes development sequences, relative VGGT corrected **30/30 distinct supplied-focal faults** and preserved **30/30 genuine digital zooms**. Their correct focal targets therefore produced zero focal-gradient error at the original TCO camera heads. Two mild pose-noise seeds produce 60 faults and 60 valid zooms in the primary table. A model-only rule does not depend on these pose seeds. The earlier design windows also gave 30/30 and 30/30 when pose duplicates were removed. [Design-window results](../relative_focal_confirmation/RESULTS.md) · [Registered replication](../../docs/RELATIVE_FOCAL_WINDOWS.md)

| Method under mild pose noise | Faults corrected / 60 | Valid zooms changed / 60 | Mean relative focal-gradient error |
|---|---:|---:|---:|
| Keep supplied focal | 0 | 0 | 0.23239 |
| Always use neighbor median | 60 | 60 | 0.14989 |
| Raw VGGT proximity | 58 | 4 | 0.04659 |
| Two geometric folds | 43 | 8 | 0.08062 |
| Raw VGGT + folds | 41 | 0 | 0.08162 |
| Relative VGGT + folds | 43 | 0 | 0.06760 |
| Relative VGGT alone | 60 | 0 | 0.00000 |

The registered rule requiring agreement with both RGB track folds **failed to replicate all its advancement checks**. Its mean focal-gradient error beat the stronger unconditional policy by 54.9%, but raw VGGT proximity scored better overall. On Chess, always taking the median scored 0.01198 versus 0.08987 for relative VGGT plus folds. Under matching conditions the established iterative visual solver scored 0.12834 with two focal candidates and 0.56929 with continuous focal estimation. [Paired results](summary.json) · [External solver fits](solver_outputs.json)

The predeclared simpler method passed its own four checks and advances to the 72-run dense pilot. It estimates the common prediction bias from the other four cameras and uses the existing model head to choose the outlier's supplied or median focal. It requires one observed focal outlier. All 150 hidden-change rows remained uncorrected, with affected-camera absolute log-focal error of 0.11333. Neither these repeated digital crops nor successful focal selection establish an improvement in the final 3D reconstruction, a general result on real zoom lenses, or transfer to another model.

All 33 new image batches received fresh camera inference and matching. Replay checked 465 conditions, 2,790 original gradient arrays, 93 classical calibration configurations, and 27,900 analytical focal derivatives with a maximum absolute discrepancy of 1.56 × 10⁻⁵. The experiment incurred $0 of additional GPU rental. [Replay instructions](REPLAY.md) · [Dense experiment design](../../docs/DENSE_FOCAL_PILOT.md)
