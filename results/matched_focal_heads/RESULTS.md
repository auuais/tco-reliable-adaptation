# Changing only the frozen point-coordinate arithmetic moved the 3D score

On the local RTX 3080 we collected the original frozen camera and depth predictions for all nine RGB batches from the focal pilot. For each batch we ran the same upstream depth-to-world computation twice on the **same heads**, under the upstream mixed-precision setting and in float32. The float32 result agreed with an independent float64 calculation to within **2.73 × 10⁻⁷ model units** in every coordinate. The maximum allocated GPU memory during these local captures was 4.34 GiB. [Captured head and point-map hashes](capture.json) · [Two-drive archive](archive.json)

| Reference | Change in accuracy distance | Change in completeness distance | Change in normalized distance score | Normal consistency, native → FP32 |
|---|---:|---:|---:|---:|
| Registered RGB | −1.20% | +7.97% | **+3.39%** | 0.6842 → 0.7608 |
| Released | −0.27% | +9.16% | **+4.44%** | 0.6789 → 0.7460 |

Lower distance and higher normal consistency are favorable. Changing only coordinate arithmetic improved accuracy distance and estimated normals, while completeness distance increased. Scores are balanced over the same 12 controlled focal conditions as the GPU pilot. The two metadata-fault directions share each scene's frozen RGB input. These 12 conditions contain nine distinct RGB batches. All 36 unique arithmetic-mode and reference evaluations are recorded. [Numeric results](summary.json)

Across the nine native point maps, **9,136,965 of 9,137,520 point vectors** matched the archived RTX 3090 vectors exactly. The other 555 were enough for all nine tensor hashes to differ. Each batch had at least 99.955% exact point vectors and its 95th-percentile coordinate-vector difference was zero. The arithmetic effect above is a within-3080 comparison. Its 3.39% registered shift is comparable in magnitude to the earlier 3.92% adaptive-versus-frozen gap, which limits how precisely that gap can be attributed to adaptation alone. The original registered transfer decision stands. A redesigned method can now be evaluated against a local float32 frozen baseline computed from the exact same local heads. [Historical dense pilot](../dense_focal_pilot/RESULTS.md) · [Local redesign](../../docs/DEPTH_SCALE_REDESIGN.md)

### Replay

Use the public data and environment from the dense pilot. Rebuild its nine inference inputs and two sets of evaluation references. For a coordinate-level comparison to the historical GPU, the scorer also reads one sealed original initial point map per RGB batch. Fresh GPU captures may differ across systems. The `verify` command checks the shared summary with the project's CPU dependencies. This experiment incurred $0 in cloud fees. [Dense pilot setup](../dense_focal_pilot/REPLAY.md)

```sh
python scripts/collect_matched_focal_heads.py --checkpoint /data/checkpoints/VGGT-1B
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/score_matched_focal_heads.py evaluate --workers 3
python scripts/score_matched_focal_heads.py verify
```
