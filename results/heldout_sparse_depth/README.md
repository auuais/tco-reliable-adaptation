# Held-out results

All 61 target views and 13 sequences completed. The two grids produced 130 sealed reconstructions and 260 reference evaluations. The primary score is the equal-scene mean of sequence-normalized accuracy and completeness. Lower values indicate better reconstruction.

| Method | Centered grid | Shifted grid |
|---|---:|---:|
| Frozen VGGT | 1.00000 | 1.00000 |
| Direct calibrated focal | 1.05505 | 1.05505 |
| Supplied pose, unchanged depth | 0.94121 | 0.94089 |
| Per-view measured scale | 0.82796 | 0.83769 |
| Spatial interpolation | 0.69002 | 0.70326 |
| Depth-aware interpolation | 0.66956 | 0.67730 |

The candidate improved over the strongest matched-input baseline, spatial interpolation, by 2.96% and 3.69%. Both missed the predeclared 5% advancement threshold. Candidate gains by scene were 4.25%, 2.25%, 5.34%, and 0.59% on Office, Pumpkin, RedKitchen and Stairs for the centered grid. For the shifted grid they were 4.74%, 5.51%, 7.49%, and minus 1.94%, respectively. Component safety and the released-reference checks passed on both grids. The shifted grid failed the no-scene-regression check.

These results show modest held-out benefit under supplied benchmark poses and neighboring measured depth. They do not establish the development effect size, backbone-independent transfer or performance with estimated poses. Earlier MASt3R transfer failed its aggregate criterion. The frozen method is retained as a baseline. TUM inputs are ready, with model inference pending a redesigned method. No rental was used for these experiments.

The full public replay is `python scripts/replay_heldout_sparse_depth.py`. It regenerates the depth corrections from the published model depth heads, checks fixed samples of the sealed 3D maps, verifies the score identities, and recomputes the complete scene-weighted decision. Recomputing dense metrics also requires the original public dataset and prepared local geometry archives. The full maps and model heads are checksum-verified on two physical disks. Further method development must use the original development scenes, with the completed held-out experiment identified as prior observed evidence.
