# Can per-view rigid corrections recover the surface?

On the original three development scenes, we first measured how much corresponding-point error can be removed by fitting reference-based rigid transforms independently for the five views. Each fit used one half of a 32-pixel checkerboard and scored the other half. The halves share an image and a scene. This diagnostic uses reference geometry and cannot be deployed. [Protocol](../../docs/PER_VIEW_ALIGNMENT.md) and [moments and per-scene results](per_view_alignment.json).

For direct-focal maps, per-view translation reduced registered corresponding-point MSE by 3.81% and per-view rigid alignment reduced it by 6.50%, relative to a single scene similarity. On the released reference the reductions were 5.85% and 8.91%. Rigid fits reduced registered error in each scene (Heads 11.16%, Chess 4.40%, Fire 3.94%). Coherent shape biases as well as camera errors can contribute.

We next fitted bounded per-view transforms on RGB feature matches and accepted proposals using a second fold with disjoint feature tracks. Each scene passed the 1% match-cost requirement. Neither fold used reference geometry. Positive percentages below mean error decreased against direct focal. [Protocol](../../docs/RGB_RIGID_VIEW_FIT.md) and [transforms and disjoint-match costs](rgb_rigid_view_predictions.json).

| Scene | Unused RGB match cost reduction | Registered surface gain | Released surface gain |
|---|---:|---:|---:|
| Heads | 12.53% | -5.36% | -5.96% |
| Chess | 2.71% | +2.52% | +1.46% |
| Fire | 17.51% | -1.67% | -3.23% |

The registered surface score rose from 0.97941 to 0.99373; the released score rose from 0.98430 to 1.00893 (lower is better). Corresponding-point MSE also showed little overall change, decreasing 0.42% on the registered reference and increasing 0.04% on the released reference. Improving sparse match agreement did not consistently improve scene geometry. [Whole-surface components](rgb_rigid_view_scores.json) and [corresponding-point comparison](rgb_rigid_paired.json).

The RGB candidate failed the advancement criteria. Before further RGB pose fitting, test how a reference-assisted rigid correction itself changes the canonical surface score with the scene scale fixed. This distinguishes limited surface headroom for these corrections from a poor RGB estimator. The $25 transfer allocation remains reserved.

## Surface scores after reference-assisted rigid alignment

Fitting the five rigid transforms to registered ground-truth correspondences on alternating image blocks produced a registered surface score of 1.07739 (direct focal 0.97941) and a released score of 1.02739 (direct 0.98430). On the registered reference, surface error decreased 7.09% on Heads and increased 11.71% on Chess and 24.44% on Fire. On the released reference the respective changes were a 7.68% decrease, a 6.93% increase and a 13.10% increase. Even these reference-assisted fits failed to improve the whole-surface score in two scenes. They minimize corresponding-point error, so they provide no upper bound on what a surface-distance optimizer could achieve. [Control protocol](../../docs/RIGID_REFERENCE_SURFACE.md) and [reference-assisted scores](rigid_reference_scores.json).

These results deprioritize further tuning of sparse rigid correspondences. A next local test is to measure how much one frozen predicted view can independently support the geometry of another, including overlap and directional surface distance, before using cross-view agreement as an adaptation objective. No fresh scene has been consulted.
