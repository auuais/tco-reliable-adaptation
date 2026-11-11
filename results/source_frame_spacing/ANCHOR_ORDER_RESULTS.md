# Changing the first image of the reconstruction

Each of the same five RGB images became the first image once. All 15 frozen forwards ran on the local RTX 3080. In all three original-order repeats the largest absolute difference from the corresponding archived heads was exactly zero for depth, intrinsics and extrinsics. Median measured forward time was 3.80 seconds and peak allocated GPU memory was 4.34 GiB. Timings exclude loading the checkpoint, staged preparation before the timed forward, and whole-surface evaluation. All numerical comparisons here reuse the three development sequences. [Capture protocol](../../docs/ANCHOR_ORDER_DEPTH.md) and [per-forward evidence](anchor_order_capture.json).

## Depth with cameras held fixed

The full median across gauge-aligned depth fields reduced mean axial squared log error by 1.53%. Its registered surface score was 0.97907 versus 0.97941 for direct focal, and its released score was 0.98343 versus 0.98430 (lower is better). On the registered reference it helped Fire by 1.39%, and increased error by 0.24% on Chess and 1.12% on Heads. The third cyclic shift scored 0.95360 on the registered reference. The best shift differed by scene. [Axial comparison](anchor_ensemble_axial.json) and [all canonical comparisons](anchor_ensemble_scores.json).

An oracle selecting one of the five fixed-camera maps per scene scored 0.93996 registered and 0.93960 released, using each reference separately to select the winning map. On the registered reference, a 5% improvement over historical frozen VGGT would require 0.91748 or lower. No whole-scene selector restricted to these five maps can meet even that requirement on these scenes. This calculation does not bound per-view or spatial fusion or changes to the cameras.
