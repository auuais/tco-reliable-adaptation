# When must TCO resolve an ambiguous focal length?

This is a secondary development analysis of completed full-model runs. We register the analysis before counting first divergences. The fresh-crop accuracy experiment has its own frozen criteria.

TCO's direct intrinsics loss uses pixel-space Smooth L1 with transition at one pixel. For a predicted focal coordinate `p` and possible targets `a` and `b`, compare `clip(p-a, -1, 1)` with `clip(p-b, -1, 1)`. If these slopes are identical on both axes of the affected view, either choice contributes the same gradient to this step. All other inputs and principal points are fixed. A fixed-step optimizer with identical randomness and execution can therefore defer an expensive visual decision until the first slope difference. Subsequent steps must repeat the check while the decision is unresolved. This argument requires that no scheduler, stopping rule or other loss uses the changed scalar loss or focal metadata. The comparator must receive the same images and other evidence whenever it is queried.

## Population and outcomes

Use the 12 sealed RTX 3090 V17 canonical runs, 40 actual updates each. For each run replace one view's two logged target focal coordinates by an alternative scaled by 1.05, 1.12 or 1.25, or by the reciprocal. Cover all five views independently. This gives 360 candidate pairs over three original five-view windows. Convert the alternative to FP32 after scaling in float64. Use exact equality of clipped FP32 slopes and verify logged original residuals and slopes bit for bit.

The primary count is the number of candidates with identical slopes at all 40 observed updates. Also report first-affected update (zero-based), number initially identical but affected later, and full counts by run condition, scene and scale. Repeated views, windows and factors are paired counterfactuals, not independent model runs. Count no wall-clock or energy saving without a timed deployment. Only the prefix through the first difference is shared. Later recorded predictions cannot establish an alternative optimizer trajectory.

Require at least 30% of the 360 comparisons to avoid invoking the visual decision for the entire schedule before timing this strategy as a standalone efficiency redesign. This threshold is a development decision. A finite trainable example must also match the eager optimizer's parameters and Adam state at every step, through both immediate and delayed decisions. Even a pass would require a timed end-to-end experiment with real candidate generation before claiming speed or dense accuracy.
