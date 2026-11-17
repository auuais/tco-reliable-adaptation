# Does MASt3R discard recoverable surface detail?

**The two fitted shape directions failed the held-source test.** Fire favored the real correction over both controls in both directions. Chess favored it only when fitting the earlier source and Heads only when fitting the later source. The rule required two of the same three scenes, together with improvement in both aggregate directions. The shape method stops at the sparse screen. Additional RunPod compute was USD 0.

## What changed

The adapted MASt3R pipeline predicts the target from four partner images and blends their geometry. We extracted two image-wide directions of disagreement that could change the surface within each 8 by 8 cell while preserving its anchor depth and removing the existing relief direction. We kept the fully adapted matching and confidence bundle and the training-side global alignment. The control shuffled the same two directions across image cells and matched their norms. The shape fit and the controls each received the same observations, gauge and alignment. [Registered design](../../docs/MAST3R_PARTNER_SHAPE.md)

Three known 7-Scenes windows provided five target images each. For each window we trained two fresh D adapters (240 updates in total). One received sparse depth only from the earlier neighboring source frame, the other from the later neighbor. We refit the metric gauge, fresh pairwise inference and matching, and the 300 coarse plus 300 fine steps independently for each fold. All 30 full view predictions for each of the three arms were sealed before scoring 1,225 held-source measurements. The two temporal directions reuse the target images. The unit of comparison is the scene.

| Fit → evaluate | Adapted MASt3R | Real shape | Shuffled shape | Training-only depth bias |
|---|---:|---:|---:|---:|
| Earlier → later | 7.012 cm | 7.007 cm | 7.024 cm | **6.629 cm** |
| Later → earlier | 6.947 cm | 6.949 cm | 6.953 cm | **6.644 cm** |

Entries are mean absolute axial depth errors, averaged equally by view and scene. Real shape reduced error by 0.072% in the first direction and increased it by 0.024% in the reverse. The two retained modes explained 86.0% of the partner disagreement energy on average. All 60 real/shuffled two-parameter fits were numerically identifiable under the registered rule. No held-source prediction was nonpositive. Across all native raster pixels, the zero, real and shuffled arms had 42, 44 and 42 nonpositive depth predictions respectively. No scored measurement was removed based on predicted depth.

An optimistic diagnostic let each view choose its own two coefficients using the held depths themselves. Even this oracle could reduce these fixed-field errors by only 1.70% and 1.37%. This is a bound for the two selected directions with frozen alignment and the registered coefficient range on these sampled pixels. It gives no bound for local bases, a different optimizer or unseen scenes. [Primary scores](scores.json) · [Per-view headroom](headroom.json)

A separate one-parameter control used the median signed residual of the training ranges as a per-view offset. Its held-source gains were 5.46% and 4.37%, with the same improvement direction in all six scene/fold cases. The additive rule is classical calibration. A neighboring sensor measurement can share registration biases with the input, making a dense evaluation necessary. [Bias fits and held-source scores](bias.json) · [Separate dense comparison protocol](../../docs/MAST3R_VIEW_BIAS_SURFACE.md)

## What transferred to dense 3D

We sealed 18 reconstructions from the six fitted adapters and evaluated all 36 surface/reference associations. The three arms used identical cameras and training-side information. The additive arm moved each pixel along its camera ray by its fitted axial-depth offset. The scale control multiplied each view's depths by a bounded median measured/predicted ratio. The registered surface score equally weights accuracy and completeness relative to the native arm. The metric depth column measures dense axial MAE in centimeters before similarity alignment.

| Fit side | Method | Dense axial MAE | Registered surface score | Released surface score |
|---|---|---:|---:|---:|
| Earlier | Native adapted | 7.009 cm | 1.000000 | 1.000000 |
| Earlier | Additive offset | 6.575 cm | 0.986738 | 0.983060 |
| Earlier | Per-view scale | **6.546 cm** | 0.987634 | 0.983401 |
| Later | Native adapted | 7.049 cm | 1.000000 | 1.000000 |
| Later | Additive offset | 6.738 cm | 0.993173 | 0.991633 |
| Later | Per-view scale | **6.715 cm** | 0.991928 | 0.991943 |

Per-view scaling reduced dense metric depth error on every scene in both folds, with equally weighted reductions of **6.60% and 4.74%**. Its registered surface gains were **1.24% and 0.81%**. Heads' surface score increased about 0.63% and 2.34% under scaling. Against that scale control, the additive method improved registered surface score by 0.091% in the earlier fold and worsened it by 0.126% in the later fold. Neither fold qualified for a wider additive-method study. Each dense arm had 44 nonpositive resampled output depths across the six folds. The original common reference mask was applied to every arm. Independent interpolation and unprojection verified all 18 complete point maps. [Surface scores and per-scene checks](surface/scores.json) · [Dense axial errors by view](surface/axial.json) · [Both reference associations](surface/observations.json)

**Spending decision.** Keep the USD 25 reserved. These are six folds on three previously inspected windows, with sparse readings and supplied poses. Per-view scale is now a measured baseline for MASt3R. A next accuracy hypothesis is to couple its five scale estimates after native alignment. With camera centers fixed, scale a matched point from view i as C_i + s_i (X_i − C_i). Fit the five scalars from sparse depth ratios and penalize the world-space separation between corresponding points from different views. At zero coupling the median-ratio objective recovers the measured per-view control. First compare zero coupling, one shared scale, and the coupled fit using held-out matches and dense surfaces on the 3080. Evaluate whether the improvement transfers to a new public scene. This proposal has no experimental result yet.

## Reproduce the measurements

From the repository root, a CPU checkout with NumPy and SciPy can refit all coefficients, check source identity and overlaps, reconstruct the held predictions and recompute the decisions.

```sh
python scripts/score_partner_shape.py verify
python scripts/audit_partner_headroom.py --verify
python scripts/partner_bias_control.py verify
python scripts/partner_bias_surface.py verify
python scripts/replay_partner_surface_points.py
```

To rerun inference, use the parent study's hashed RGB packets and environment, the original public 7-Scenes data and the pinned MASt3R checkpoint. The local container was `tco-mast3r:pilot`, with the repository at `/workspace`, the checkpoint directory at `/weights` and the original 7-Scenes directory at `/data`. Use a fresh checkout or separate output location for reruns. Training intentionally protects existing run directories. Use a fresh process for each training fold.

```sh
python scripts/prepare_partner_sources.py /data
for scene in heads chess fire; do
  for side in earlier later; do
    python scripts/train_partner_shape.py "$scene" "$side" --weights /weights
    python scripts/predict_partner_shape.py --case "$scene" "$side"
  done
done
python scripts/predict_partner_shape.py
python scripts/score_partner_shape.py extract
python scripts/score_partner_shape.py score
python scripts/partner_bias_control.py score
python scripts/audit_partner_headroom.py
python scripts/partner_bias_surface.py predict
python scripts/partner_bias_surface.py score
python scripts/replay_partner_surface_points.py --export
```

The inputs verified 75 source file identities. Independently recombining the two projected packets reproduced all 15 historical merged packets exactly. Each GPU run consumed 20 directed image pairs and produced 300 finite prior-gradient records. Peak training allocation was 4.58 GiB. GPU fitting time summed to 315 seconds, excluding model loading, forward/matching, alignment and validation. Cached native depth reconstructed from the anchor formula agreed within 6.3e-7 in model depth units.

The initial Heads/later prediction attempt encountered an L-BFGS-B line-search termination. We preserved the failure, kept the objective and bounds, and checked the box-constrained first-order condition directly. All previously sealed Heads/earlier coefficients replayed. During cross-platform CPU verification, three recomputed fitting-side MAEs differed by at most 8.8e-11 meters. CPU replay checks fitting-side MAEs within 1e-8 meters and requires the original held metrics, input identities and decisions to agree exactly. [Failure record](attempts/heads_later_solver.txt) · [Input identities](inputs.json) · [Prediction seal](seal.json)
