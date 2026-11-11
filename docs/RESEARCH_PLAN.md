# Research plan from reproduction to a publishable method

Surface-distance fitting selected zero corrections on every development fold. Edge-aware smoothing improved axial depth and partly recovered normal consistency, but failed the whole-surface criterion on both references. Reference-based rigid alignment on complementary image blocks reduced corresponding-point error by 6.50% on registered direct-focal geometry and 8.91% on released geometry. The RGB-only estimator improved unused feature matches in all three scenes, but whole-surface errors increased on both references. Test canonical surface headroom under reference-assisted rigid corrections before fitting further RGB estimators. [RGB rigid evaluation](../results/source_frame_spacing/RGB_RIGID_VIEW_RESULTS.md) [Per-view evidence](../results/source_frame_spacing/per_view_alignment.json) Paid transfer remains on hold. [Surface results](../results/source_frame_spacing/WEIGHTED_SURFACE_RESULTS.md)

**Current stage.** Repeatability and controlled calibration are complete. The camera and depth redesigns have yet to meet the registered whole-surface criterion. Cross-fitted RGB reliability weights reduced axial error by 3.35% against a fixed quarter step, but increased registered surface error by 9.25% against the strongest local comparator and by 10.66% against historical frozen VGGT. Shuffling the adjustments among eligible pixels worsened depth and normal consistency while improving nearest-neighbor completeness. Check alignment and metric sensitivity before using the composite to select a redesign. Keep paid MASt3R, independent scene and cross-dataset evaluation allocations reserved. [Surface results and controls](../results/source_frame_spacing/WEIGHTED_SURFACE_RESULTS.md) · [Calibration and camera evidence](../results/source_frame_spacing/RESULTS.md) · [TCO reproduction](../results/dense_focal_pilot/RESULTS.md)

Date: 2026-11-09. Research capacity: 40 hours/week through early 2027. Public datasets only. The course midterm is November 10 and the final November 24.

**Status update, 2026-11-11:** baseline, component-ablation and the 40-job controlled calibration screen are complete. [V6 results](../results/repeat_calibration_v6/RESULTS.md) fail the calibration advancement gate and confirm same-seed execution variation despite matching captured initialization/RNG. All 40 alternate-reference CPU evaluations are also complete and do not change that decision. The fixed-match proxy has zero eligible coverage. [The current decision](DECISION_AFTER_REPEAT_CALIBRATION_V6.md) withholds later spending and prioritizes a local pose-precision engineering comparison before a redesigned method. The stages below remain hypotheses and gates, not achieved claims.

## Central question

Can a reconstruction model decide which external constraints are useful, take a bounded adaptation step, and stop before further optimization damages geometry?

The intended contribution is reliable, compute-budgeted use of imperfect priors. It must improve the accuracy/latency tradeoff and reduce harmful adaptation beyond tuned simple baselines. Transferring TCO to another backbone, decreasing its learning rate, or adding a generic confidence gate alone is unlikely to support a strong novelty claim. A top conference is an aspiration contingent on evidence; no outcome is guaranteed.

Our exploratory evidence supports investigating this question: clean adaptation sometimes harms Heads, severe synthetic priors cause substantial degradation, and reducing the number of steps reduces runtime. It does not yet establish that a controller can recognize harmful updates without reference labels.

## First executable study

`baselines_v4` freezes 39 full-resolution jobs before execution: 3 new zero-step controls plus 3 development sequences x 3 optimizer seeds x 4 conditions. Conditions are released TCO (40 steps, LR 0.001), LR 0.0003, LR 0.0001, and visual-only adaptation (LR 0.0003, pose/intrinsics weights zero). The visual objective keeps its released weight 0.2. Seeds are 42, 123 and 2026; they do not resolve the separately observed execution nondeterminism.

Measure accuracy, completeness, normal consistency, runtime and peak memory for every run. The prespecified combined error is the mean of accuracy/base-accuracy and completeness/base-completeness, equally averaged over scenes. Report the individual metrics and worst cases alongside this score. Never select the best seed or optimize against the final test set.

**Gate 1:** if lower LR removes most harm and preserves gains, use it as the principal baseline. If visual-only matches prior-assisted adaptation, investigate why the camera constraints add little before pursuing prior selection. If conclusions depend on a single seed, allocate repeats before developing a complex policy. This screen is exploratory and clean-prior-only; it cannot establish robustness to noisy priors.

## Mathematical working hypothesis

Let frozen backbone weights be theta, trainable LoRA parameters a, predicted geometry G(a), visual loss L_v, and camera-prior groups g with residuals r_g(a). The current objective approximately has the form

`L(a) = L_v(a) + sum_g w_g L_g(a)`.

Fixed weights presume useful priors. An incorrect prior can reduce its own residual while worsening the reconstruction. A lower objective therefore does not certify better geometry. At first order, a proposed prior update d_g changes an independent visual validation loss by approximately `grad(V)^T d_g`. Opposing gradients can identify a conflict, but agreement is not proof of correctness because both objectives may share a bias.

The proposed study combines three separately ablated components:

1. **Scale-aware robust constraints.** Use `rho(r_g^T Sigma_g^-1 r_g)` with explicit rotation, translation and focal units. Start with bounded, externally specified uncertainty from the perturbation model. Compare Huber/Cauchy robust loss and simple residual weighting before learning uncertainty. Raw model confidence is not automatically a calibrated covariance. Do not let an unconstrained uncertainty predictor avoid the objective by inflating every variance.
2. **Constraint usefulness probes.** From exactly the same model, optimizer and RNG snapshot, compare an update with a prior group against an update without that group. Score both on fixed, disjoint visual tracks or pairs, using a robust reprojection/cycle error V and a coverage penalty. Define `u_g = V(a_without_g) - V(a_with_g)`. This contrasts the contribution of that constraint with the same visual update; comparing only with the pre-update model would confound the two. A positive u_g is a hypothesis of usefulness, not an oracle. All probes are charged to compute.
3. **Bounded acceptance and stopping.** Accept a candidate only when validation improves enough relative to numerical variation and geometry displacement stays below a trust bound. A practical constraint is `mean(||G(a+d)-G(a)||^2 / s^2) <= epsilon^2`, with fixed coordinate/gauge convention and robust scene scale s. If using a Gauss-Newton approximation, compute Jacobian products or low-dimensional projections; never materialize a dense matrix over millions of LoRA parameters. Rollback restores adapters, Adam moments, counters and all RNG states. Early stopping is tested against tuned fixed step budgets at matched wall-clock cost.

### Proxy validity before a controller

Cache correspondences from the unadapted model or an independent matcher. Freeze track identity, train/validation partition, minimum coverage and occlusion policy before adaptation. Otherwise the model could appear to improve by discarding difficult points. Do not use reference depth, reference poses hidden from the chosen prior source, or reconstruction metrics in online decisions.

Use reference reconstruction error only offline to label whether each probe actually helped. On development scenes, report rank correlation, harmful-update detection and utility-versus-coverage. Compare against loss decrease, native confidence, residual magnitude, gradient conflict and random selection. If the proxy fails on low parallax, repeated texture or severe corruption, report that failure and stop the learned-controller branch. Agreement among biased visual checks is not a correctness guarantee.

## Data and evaluation design

Already inspected development data: Heads, Chess and Fire sequences. Reserve Office, Pumpkin, RedKitchen and Stairs for subsequent scene-level evaluation; do not tune on their labels. Baseline-only full-benchmark reproduction must be clearly separated from method selection. The pinned sparse protocol has 18 sequences; respect its differing frame counts, including three-frame Stairs samples.

Freeze separate optimizer and corruption seed lists. Test clean, moderate and severe camera corruption, plus isolated rotation, translation and focal errors. Synthetic perturbations are controlled diagnostics, not evidence that real estimators make the same errors. Add a second public benchmark such as ETH3D and independently estimated camera priors before making practical robustness claims. Download/preprocess on the local machine before rental time.

Report equal-scene means, all per-sequence outcomes, normal consistency, harmful-update rate, catastrophic failures, runtime and VRAM. Count matching, probes, rejected steps and preprocessing used at inference. Include both equal-step and equal-wall-clock comparisons. Keep the current Sim(3)+ICP protocol for comparability, but add pose/reprojection measures with explicit gauge handling. Absolute scale claims require separate unaligned or SE(3)-aligned metrics and an observable metric source.

Use several seeds and numerical execution repeats where effects approach the measured variability. Uncertainty intervals must resample independent scenes/sequences, not thousands of correlated pixels. Four held-out scenes are still small: cross-dataset evidence matters. Record full failures and incomplete batches; never silently drop them from denominators.

## Baselines and novelty audit

Required engineering baselines: unadapted VGGT, released TCO, tuned LR, step counts 0/5/10/20/40, visual-only, global prior weights, robust loss, residual/confidence gate, and simple early stopping. Match the method's inference inputs and total compute budget wherever possible.

Relevant primary work, checked November 9, 2026:

- [TCO](https://arxiv.org/abs/2604.03878) already studies VGGT and Pi3. Pi3 transfer alone is not our contribution.
- [Test3R](https://arxiv.org/abs/2506.13750) uses cross-pair geometric consistency for test-time learning. Backbone transfer and test-time consistency require comparison with this work.
- [Pow3R](https://github.com/naver/pow3r) provides prior-aware reconstruction; it is relevant when matching camera/depth inputs.
- [Trust3R](https://arxiv.org/abs/2605.19539) estimates evidential pointmap uncertainty and uses gated mean refinement. Uncertainty weighting alone is already an occupied direction.
- [GeoCond](https://arxiv.org/abs/2609.18465) uses conditioning and stability signals for uncertainty and refinement gating. A generic reliability gate is insufficient novelty.

Our tentative differentiator is testing the marginal benefit of external constraint groups under a fixed online budget, with rollback and transferable geometry validation. This distinction is a research hypothesis requiring a full related-work comparison, not a confirmed first-of-its-kind claim. Audit adjacent work again before submission.

## MASt3R and optional sensor extension

Choose [MASt3R](https://github.com/naver/mast3r) as the first transfer experiment because its descriptors can support fixed correspondence sets and its pairwise geometry differs materially from VGGT. It is a useful second backbone, not automatically a newer state of the art. First reproduce its ordinary inference/global alignment; then map camera constraints to pairwise pointmaps with explicit scale/gauge conventions. Compare unadapted MASt3R, its normal alignment, applicable Test3R adaptation, naive TCO transfer and the proposed reliability rule. Cache matching before online decisions and count its cost.

DUSt3R is a simpler compatibility/ablation fallback. Defer Fast3R until the method has passed VGGT and MASt3R gates: multi-view scaling adds engineering and comparison costs without proving novelty.

IMU is a later branch, not the midterm dependency. Start with orientation constraints on a public synchronized dataset such as [TUM VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset), verifying timestamp, extrinsic and bias conventions. Metric scale from inertial data requires sufficiently excited motion and an appropriate state/measurement model; a rotation prior alone does not determine scale. Compare against existing visual-inertial systems and [VI3](https://arxiv.org/abs/2609.03824) before making a new claim. Prefer solving robustness first to adding another sensor prematurely.

## Milestones and decision rules

| Period | Deliverable | Decision |
|---|---|---|
| Nov 9-10 | Versioned partial reproduction, baseline screen and honest 5-minute midterm evidence | No novelty claim in course reproduction |
| Nov 11-17 | Full baseline accounting, controlled-noise sweep, proxy validity tests | Continue only if a visual proxy discriminates harmful updates beyond simple residual/confidence rules |
| Nov 18-24 | Minimal robust/bounded update prototype and ablations for final | Report actual gains and failures; keep course work reproducible |
| Late Nov-Dec | Held-out scenes, estimated priors, second dataset, MASt3R transfer | Require benefit beyond tuned TCO at comparable compute on more than development scenes |
| Jan-early 2027 | Independent reruns, related-work audit, complete tables and manuscript | Choose a suitable venue based on demonstrated contribution and current deadlines |

Internal advancement targets, not promised results: at least 5% lower combined error than the strongest tuned baseline on aggregate held-out data without a >2% aggregate degradation in either component, or at least 30% lower measured latency at comparable accuracy. Seek a substantial reduction in harmful adaptation, with no hidden collapse in low-overlap subsets. These thresholds are resource decisions, not statistical proof or acceptance criteria. If simple baselines explain the benefit, publish a rigorous limited-scope study or change the research question rather than inflate novelty.
