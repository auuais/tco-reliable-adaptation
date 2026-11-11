# Held-out execution status

The frozen VGGT checkpoint completed all 61 target views in thirteen sequences from four reserved scenes. Both three-view Stairs windows are included. Inference inputs contain RGB, supplied target poses and sparse depth from two neighboring source frames. Target depth is reserved for the scoring process.

All 130 point maps are sealed. Full maps and captured model heads are checksum-verified on two physical disks. The public compact depth heads, camera arrays and sparse samples regenerate all range updates. The public replay compares fixed spatial samples against nine generated map types per sequence and checks the archived identity of the ordinary frozen map. Run `python scripts/replay_heldout_sparse_depth.py --predictions-only` while dense scoring is in progress. After results publication, omit that flag to verify the complete aggregation.

The dense evaluator runs locally in container tco-heldout-sparse-score01. Its immutable predictions are recorded in predictions.json, and completed per-map checks are resumable using the matching local geometry archives. Until all sequences complete, aggregate held-out scores remain pending. The two grids reuse all targets. Results will be averaged within scenes and equally across the four scenes.
