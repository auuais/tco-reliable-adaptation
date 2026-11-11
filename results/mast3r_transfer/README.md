# MASt3R transfer result

Ordinary MASt3R with standard sparse global alignment remained the strongest aggregate comparator on both grids. Lower registered surface scores are better. Each scene is normalized by its ordinary MASt3R score and the three scenes receive equal weight.

| Method | Centered grid | Shifted grid |
|---|---:|---:|
| Ordinary MASt3R | 1.00000 | 1.00000 |
| Supplied poses and unchanged depth | 1.21995 | 1.23228 |
| Per-view median | 1.11380 | 1.11440 |
| Spatial interpolation | 1.06384 | 1.03929 |
| Depth-aware interpolation | 1.01222 | 1.00375 |

The depth-aware update improved on spatial interpolation by 4.85% and 3.42% but missed the primary transfer criterion. Relative to ordinary MASt3R, heads gained 10.82% and 13.08%, chess gained 7.06% and 12.68%, and fire regressed 21.55% and 26.89%. Three of four advancement checks failed on both grids. The released-reference check passed. Finding a positive effect on VGGT does not establish backbone-independent improvement.

All inference ran on the local RTX 3080. Chess and fire cold forward, matching and alignment took 35.0 and 33.3 seconds, excluding checkpoint loading. Peak PyTorch allocation was 3.17 GiB. Heads used cached matching after the first capture exposed 26 finite behind-camera pixels in its fifth depth map. The corrected capture preserves those pixels and excludes them from logarithmic updates. Its 17.3-second cached run is unsuitable for cold runtime comparison. Rental spend was $0.

The public replay uses the three archived native depth maps and cameras. It checks checkpoint capture hashes, resamples the depth with a separate implementation, rebuilds all 27 predictions, and checks the 54 score records and summaries. Use `python scripts/replay_mast3r_transfer.py`. Full surface scoring additionally requires the original benchmark reference archives. Testing a learned sparse-depth TCO update or Test3R-style adaptation and testing realistic pose noise remain open comparisons. Continue held-out evaluation for the VGGT method that passed its development criterion.

## Implementation analysis before the requested pause

The implemented MASt3R path loads the pinned public metric checkpoint, runs complete symmetric image pairs at 512 by 384, and performs the standard 300-step coarse and 300-step fine sparse global alignment with depth optimization. The experiment then resamples axial depth and camera intrinsics consistently for the existing evaluation grid. It produced ordinary geometry and eight supplied-pose sparse-depth variants per scene, with 27 predictions and 54 reference evaluations across three scenes.

Network-weight adaptation is pending. The completed transfer changes reconstruction outputs using the same sparse inputs and supplied trajectories. It does not instantiate or optimize MASt3R LoRA adapters. The ordinary alignment already optimizes geometric variables, which must be distinguished from updating the network weights.

The transfer candidate regressed by 1.22% and 0.38% against ordinary MASt3R. Fire regressed by 21.55% and 26.89%, while Heads and Chess improved. Supplying poses with unchanged depth also worsened aggregate error by about 22%. These observations make pose/depth coupling an important debugging target. They do not isolate the cause of the Fire failure or show that learned MASt3R adaptation cannot help.

Ordinary inference, matching and alignment fit the local RTX 3080 at about 3.17 GiB peak PyTorch allocation. The two uncached five-view runs took 33.3 and 35.0 seconds, excluding checkpoint loading. These measurements do not establish training-memory requirements. A resumed implementation should first test jointly weighted camera and sparse-depth constraints within alignment against the existing standard alignment, then profile a single pairwise gradient step before selecting a LoRA configuration or rental GPU. A learned transfer needs an explicit pairwise coordinate gauge, camera/depth loss definitions, and a comparison at equal input and compute budgets.

The user requested pausing the goal after this analysis. MASt3R rental spending is zero of the USD 25 allocation. Unused-pod storage cleanup is still pending. All current experiment containers have finished, and the user confirmed the provider pods are stopped.
