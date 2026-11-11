# Can agreement with relative VGGT focal predictions protect genuine zooms?

The previous study's two-fold geometric rule mistakenly changed 9 of 60 valid focal outliers with mild pose noise. Its gradient error was lowest on average, while absolute VGGT proximity recovered more correct metadata values. Across 30 paired image crops the frozen network consistently predicted the direction of the intervention. This motivates a revised decision before renting another GPU. These parent results were known at design time. We have yet to calculate decisions of the revised rule. [Parent study](../results/focal_zoom_gradient/RESULTS.md)

## Fixed decision

Use the same supplied two-axis focal metadata, a single ordinary forward pass of the original VGGT camera head and the same two disjoint RGB track folds. Select the single focal outlier and neighbor-median replacement using the parent's fixed rule. If there is no unique outlier, retain the supplied intrinsics.

For each focal axis separately, estimate a multiplicative bias of the model by taking the median of `log(model_focal / supplied_focal)` over the other four cameras. Subtract this bias from the outlier's model log focal. Choose the consensus target only if its mean absolute log distance to this corrected prediction is strictly smaller than the distance to the supplied target. Edit only when both RGB folds also favor that edit by the original 0.25 pixel margin. All other cameras and pose metadata remain as supplied. No scale threshold, fitting, new camera inference, ground truth or paired unzoomed counterpart enters the rule.

The other four supplied focal lengths are the calibration anchor. A shared error across these four cameras and a real unreported focal change in uniform metadata remain outside this design's identification ability. A common scale error in model predictions cancels algebraically, but errors distinctive to the selected image can persist.

## Paired evaluation and next step

Reuse all 465 published conditions without refitting. Report fault corrections, valid zoom edits, target log-focal error and relative original-loss focal-gradient error under all three pose severities. Include the previous two-fold rule, raw VGGT proximity, raw VGGT agreement with two folds, relative VGGT alone, and the two conditional gradient results from the classical comparator. Show the two previously defined controls. The observed five-view encodings must come from the exact input hash for the condition.

In the 120-case primary group require at most 3/60 false edits, at least 20% lower mean focal-gradient error than the better unconditional baseline, no scene worse than its better unconditional baseline, and no worse mean focal-gradient error than both raw VGGT alone and its agreement with two folds. Report the full comparison even if just one check fails. A pass is a reason to repeat the frozen method on fresh offset-25 five-frame windows (25, 225, 425, 625, 825) in each of the same three development sequences before any dense pilot. All four held-out scenes remain reserved. A tie with raw VGGT plus geometry gives no evidence of an advantage from bias correction. This screen has zero incremental RunPod cost and releases no later allocation.
