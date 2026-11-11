# Local affine redesign result

The twelve predeclared reconstructions completed on both development sampling grids. The local affine depth-aware update scored 0.877136 and 0.858502 on the registered surface metric, versus 0.873839 and 0.856790 for the earlier local-constant depth-aware update. Lower values are better. The redesign increased error by 0.38% and 0.20%. The previous method remained the strongest comparator.

The spatial affine control scored 0.930738 and 0.914853, modestly better than the earlier spatial interpolation controls but insufficient to explain an overall advantage against the previous depth-aware candidate. The gain, per-scene safety and released-reference checks failed on both grids. Component safety and the comparison to historical native VGGT passed. This is a negative ablation, and the method does not advance.

All fitting and evaluation used the original Heads, Chess and Fire development scenes. No held-out or TUM target labels were used for this redesign. No rental was used. Full predictions are verified on two disks. Public replay regenerates the corrections from the earlier published depth heads and range packets using `python scripts/replay_local_affine_sparse_depth.py`. The augmented least-squares tests independently verify the local affine solver.

The next development check will test scale-and-shift calibration against the same sparse measurements, including the calibrated result without any local correction as a strong control. A global metric scale alone cannot represent an additive axial-depth bias. This is a classical calibration baseline and requires measurement rather than a novelty claim. All later evaluations retain the completed experiments as observed evidence.
