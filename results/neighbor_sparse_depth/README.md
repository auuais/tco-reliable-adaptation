# Results on three development scenes

The projected grid supplied 1,225 usable samples across 15 target views. All methods used the same two neighboring depth images per target, the same supplied trajectory, and the same frozen predictions. The candidate added depth-discontinuity weighting to spatial interpolation.

| Method | Mean absolute log depth error | Registered bidirectional surface score |
|---|---:|---:|
| Direct focal baseline | 0.04376 | 0.97941 |
| Per-view measured scale | 0.04024 | 1.06193 |
| Spatial interpolation | 0.03665 | 1.04504 |
| Depth-aware interpolation | 0.03504 | 0.98461 |
| Strongest earlier control (permuted) | | 0.94090 |

Lower values indicate better results. Metric depth used a single common scale inferred from neighboring samples per scene. The depth-aware update reduced mean absolute log depth error by 19.9% against direct and 4.4% against spatial interpolation. The bidirectional surface score improved by 5.8% against spatial interpolation, but regressed by 0.5% against direct and 4.6% against the strongest historical local control. Relative to that control, heads improved 9.6%, chess regressed 14.9%, and fire regressed 6.5%. All five registered advancement checks failed. No rental was used.

Run the public CPU replay with `python scripts/replay_neighbor_sparse_depth.py verify`. For full local reconstruction use the parent frozen-heads and canonical reference archives, then run `neighbor_sparse_depth.py prepare --assets <7scenes-dir>`, `predict`, and `score` under scripts. Run `scripts/audit_neighbor_sparse_depth.py` for the separate post-seal metric-depth audit. The replay regenerates corrections from the public frozen heads and compares sampled 3D coordinates from all nine sealed maps. Original dense score computation requires the public dataset and prepared reference archives.

The strongest remaining question is whether using the supplied trajectory consistently for both anchor projection and multi-view fusion explains the gap between single-view depth and multi-view geometry. Test this with the same known poses in every competing method, including a fixed-pose baseline with unchanged depth. Any gain caused by replacing the pose must be credited to the supplied pose. Results on these three reused scenes alone cannot establish generalization.
