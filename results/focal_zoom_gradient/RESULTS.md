# Image matches lower focal-gradient error but still overwrite valid zooms

On the 120 primary cases, the two-fold RGB rule reduced mean relative focal-gradient error by **66.89%** compared with the better unconditional policy. It corrected **43/60** corrupted focal priors and changed **9/60** correct zooms (15%). The registered false-edit ceiling was 5%, or at most three edits in this set. The candidate missed this condition. MASt3R transfer and held-out spending remain pending. [Frozen criteria](../../docs/FOCAL_ZOOM_GRADIENT.md) · [Results by case](summary.json)

| Choice of supplied focal | Mean relative focal-gradient error | Mean relative full camera-gradient error | Faults corrected / 60 | Valid zooms changed / 60 |
|---|---:|---:|---:|---:|
| Keep metadata | 0.2343 | 0.6343 | 0 | 0 |
| Always use neighbor median | 0.2126 | 0.6384 | 60 | 60 |
| Pretrained VGGT, closer candidate | 0.1140 | 0.6105 | 56 | 10 |
| One RGB track fold | 0.0704 | 0.5886 | 43 | 11 |
| Both track folds | 0.0704 | 0.5886 | 43 | 9 |

The two-fold rule beat the better unconditional focal-gradient score in all three scenes. Its scores were 0.0250 on Chess, 0.0224 on Fire and 0.1638 on Heads. Absolute VGGT proximity had zero focal-gradient error on Chess and Fire, but scored 0.3421 on Heads. VGGT proximity made fewer metadata mistakes overall (14 versus 26). Whether a metadata mistake changes the TCO update depends on the head's predicted focal length and the saturated regions of the Smooth L1 loss. The retained candidate already had exactly correct focal gradients for 86/120 primary cases.

### What changes with pose uncertainty?

| Pose error | Faults corrected | Valid zooms changed |
|---|---:|---:|
| Supplied calibrated poses | 24/30 | 1/30 |
| 1° and 2% of trajectory diameter | 43/60 | 9/60 |
| 3° and 6% of trajectory diameter | 32/60 | 18/60 |

Pose perturbations use two public seeds at each nonzero severity. These are repeated measurements of 30 physical image interventions in three development windows. A unique supplied focal outlier triggers the method. It abstained on all 15 uniform clean inputs and on all 150 hidden zooms where an incorrect uniform focal concealed the real change. These tests measure external gradients at the frozen camera head. Dense reconstruction accuracy after adaptation is still unmeasured for this method.

### What did the network see?

On all 30 single-view zoom interventions, both focal axes of the VGGT prediction moved in the correct direction. The median predicted change was 1.039 times the true change in log focal. Mean absolute error on the changed camera was 0.0223 log units (0.1133 if the output never responded). Unchanged cameras drifted by 0.0094 log units on average. This result motivates testing relative focal predictions when an absolute pretrained estimate disagrees with the metadata. [All paired predictions](crop_response.json)

All 33 distinct five-view image batches used fresh CPU FP32 VGGT camera inference and fresh DISK/LightGlue matches. No feature identity crossed the two folds. Starting again from the original scene poses and redetected pixels reproduced the 300 visible-case decisions with a maximum fold-score discrepancy of 1.1 × 10⁻¹² pixels. A separate closed-form derivation agreed with 18,600 field-of-view derivatives from original TCO; maximum absolute discrepancy was 1.96 × 10⁻⁵. The local experiment added $0 of cloud spending. [Replay instructions](../../docs/REPLAY_FOCAL_ZOOM.md) · [Analytical verification](analytic_gradient_check.json) · [Classical self-calibration comparison](../visual_focal_baseline/RESULTS.md)
