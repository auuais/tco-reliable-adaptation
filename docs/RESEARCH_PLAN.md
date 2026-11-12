# Research plan from reproduction to a publishable method

**Latest implementation, November 12.** The local MASt3R joint-prior alignment factorial completed 21 runs and 42 reference evaluations. Joint constraints improved over ordinary MASt3R by 7.63% and 8.66%, but only by 1.43% and 1.29% over depth-only alignment. Both grids missed the registered gain criterion, and shifted Chess regressed slightly against depth-only. All baseline tensors repeated exactly. Rental spending was zero and later allocations remain reserved. Network-weight adaptation is pending. [Results](../results/mast3r_joint_priors/README.md) · [Next implementation](MAST3R_NEXT_IMPLEMENTATION.md)

Surface-distance fitting selected zero corrections on every development fold. Edge-aware smoothing improved axial depth and partly recovered normal consistency, but failed the whole-surface criterion on both references. Reference-based rigid alignment on complementary image blocks reduced corresponding-point error by 6.50% on registered direct-focal geometry and 8.91% on released geometry. The RGB-only estimator improved unused feature matches in all three scenes, but whole-surface errors increased on both references. Reference-assisted rigid correction also worsened whole-surface scores in two scenes. Cross-view agreement accepted four harmful updates among five on the archived TCO runs. A two-plane correction failed the whole-surface criterion. Changing the reference image produced measurable depth variation, but averaging depth and pose failed the transfer criterion. Disagreement failed to identify useful RGB depth corrections. The strongest camera-prior control coupled a fixed half-weight pose update to tangent-plane depth adjustments. It improved completeness in the aggregate but worsened accuracy and normals versus the no-prior model. Matched-pixel evaluation found that the tangent update reduced axial error but worsened normals. Mesh rendering and range limiting did not meet the surface criterion. Fresh noise directions reproduced the composite gain over the no-prior model, yet matched reversed moves won on two of four draws. Neighborhood normals reduced excess normal error by 44% and missed the paired screen. Joint RGB reprojection improved feature agreement but worsened both canonical surface scores; separating its pose and depth outputs did not rescue it. [Reprojection results](../results/source_frame_spacing/JOINT_REPROJECTION_RESULTS.md) [Noise and direction controls](../results/source_frame_spacing/CAMERA_NOISE_RESULTS.md) [Transport results](../results/source_frame_spacing/CAMERA_TRANSPORT_RESULTS.md) [Camera-prior results](../results/source_frame_spacing/ANCHOR_PRIOR_RESULTS.md) [Anchor-order results](../results/source_frame_spacing/ANCHOR_ORDER_RESULTS.md) [Cross-view results](../results/source_frame_spacing/CROSS_VIEW_RESULTS.md) [RGB rigid evaluation](../results/source_frame_spacing/RGB_RIGID_VIEW_RESULTS.md) [Per-view evidence](../results/source_frame_spacing/per_view_alignment.json) Paid transfer remains on hold. [Surface results](../results/source_frame_spacing/WEIGHTED_SURFACE_RESULTS.md)

**Current stage.** Held-out evaluation completed all 13 reserved sequences. The sparse-depth correction improved by 2.96% and 3.69% over spatial interpolation on the centered and shifted grids, missing the registered 5% threshold. Shifted Stairs regressed 1.94%. MASt3R transfer also failed its aggregate criterion. Preserve these results as baselines and redesign on the original development scenes before advancing later experiments. TUM input preparation is complete. The subsequent local affine development redesign increased error by 0.38% and 0.20%. The subsequent robust scale-and-shift development test improved by 3.32% and 3.78% against the prior strongest control, with all scenes improving and component checks passing. Both grids still missed the 5% gain criterion. The 5% internal spending target had also been applied to zero-rental confirmation. Correct that scope prospectively: evaluate the redesigned calibrated method on the untouched TUM targets using local hardware, report all original failed decisions, and reserve later rental allocations. TUM confirmation completed all 48 reconstructions: the calibrated candidate was 0.20% and 0.01% worse than the strongest controls. All TUM advancement checks failed. Fresh inference reproduced every tensor exactly. Independent distance checks passed in all nine fixed cases, while native normal checks disagreed in all three sequences and require a precision investigation. [TUM outcome and verification](../results/tum_sparse_depth/README.md) [Calibration tables](../results/scale_shift_sparse_depth/README.md) [Affine ablation](../results/local_affine_sparse_depth/README.md) [Held-out tables](../results/heldout_sparse_depth/README.md) | [MASt3R tables](../results/mast3r_transfer/README.md) | [TUM inputs](../results/tum_sparse_depth/README.md)

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

## Sparse-depth comparison update

The successful development configuration corrects frozen depth maps from neighboring measured ranges with supplied poses. Publication comparisons must address existing sparse-depth methods with equal measurements and an explicit pose convention.

CAPA adapts frozen 3D foundation models using LoRA or visual prompt tuning with sparse geometric observations and sequence-level parameter sharing. Its experiments explicitly include VGGT. Test-time sparse-depth adaptation and temporal parameter sharing therefore have established prior work. Compare its accuracy, memory and runtime at the same measurement density before advancing an efficiency claim. [CAPA paper](https://arxiv.org/html/2602.14751v1)

Pow3R accepts camera intrinsics, relative poses and sparse or dense depth alongside RGB. Its official CVPR 2025 checkpoint and inference code provide a relevant conditioned-model comparator. Distinguish model pretraining, test-time optimization and postprocessing costs when comparing its results. [Pow3R implementation](https://github.com/naver/pow3r)

Chen and colleagues propagate sparse depth using foundation-model structural cues through 2D and 3D spaces, with an additional learned correction module. Their paper reports evaluation on sixteen out-of-distribution datasets. Depth-guided propagation itself therefore requires a specific comparison rather than a broad novelty claim. [Propagation paper](https://arxiv.org/html/2508.04984v1)

The current result remains evidence for a fixed, inexpensive correction on VGGT under accurate supplied poses. A potential contribution is a demonstrated accuracy and cost advantage with calibrated uncertainty about pose and sampling failures. This is an inference about the research direction. It requires the held-out and second-dataset experiments and the relevant strong baselines. No new benchmark result follows from this literature check.
