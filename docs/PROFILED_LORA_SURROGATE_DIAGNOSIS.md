# Conditional diagnosis of surrogate-to-native disagreement

D0 and D1 are complete. The verified D1 report records query-loss reductions on both inference paths and attenuation on the training path. The chosen follow-up is calibration compensation regularization with a trace-matched damping comparator. The preparation requirements below are retained for reproduction. [D1 report](../results/lora_profiled_d1_v1/README.md) · [Redesign](PROFILED_COMPENSATION_REDESIGN.md)

The M2 inventory, global seal and registered scoring are now complete with no eligible nominee. This guide supplies the mathematics for the bounded failure diagnosis; it is not evidence that a cause has been observed. D0 is now complete with an exact replay; start the selected D1 work package below and read the [current handoff](CALIBRATION_AWARE_LORA_HANDOFF.md). The broader diagnostic options below are conditional D1 choices, not permission to load additional tensors or launch them all. Preserve the completed screen and its frozen sources. After its verdict, use this guide with the [mathematical plan](FROZEN_LORA_ACCURACY_PLAN.md), [goal contract](CALIBRATION_AWARE_LORA_GOAL.md) and [failure policy](PROFILED_LORA_TERMINAL_FAILURE_POLICY.md). All diagnostic measurements below use model outputs, recorded correspondence identities and the permitted adaptation packet only; support **and** query are training observations.

**Selected D1 work package after verified D0**

D0 found a nonzero P3 raw-depth change with 68.47% mean per-pair non-scalar energy. It does not establish whether that change is useful or survives fusion/calibration. The [prospective D1 design](../manifests/lora_profiled_d1_design_v1.json) therefore selects one paired inference/calibration-response probe, emphasizing preservation of learned shape rather than assuming scale absorption. D1 has not accessed production arrays or run a model. The [combined helper preparation](../results/lora_profiled_d1_v1/preparation01.json) passes84 pinned CPU tests:32 math,16 artifact-boundary,16 sensor-bridge and20 capture/preprocessing fixtures. Actual D1 production acceptance remains pending.

Use the implemented helpers described in the [handoff](CALIBRATION_AWARE_LORA_HANDOFF.md), and complete the remaining controller/registration layers without changing the design:

| Component | Required behavior |
|---|---|
| Strict D1 inputs and byte registry | Exact member allowlists, checkpoint/config, parent-seal links, selected teacher/adapter/cache hashes, unchanged source and runtime |
| Model capture | One pristine model; five encoder-feature computations; 20 pristine teacher pairs; then seed42/install LoRA/load saved P3 factors; 20 final pairs on the same features; no optimizer/backward |
| Common fusion | Same pinned canonical_depth, ascending partner order, FP32 XYZ/confidence on the same CUDA device, autocast disabled, pristine BF16 confidence for all four states; only then promote depth to float64 |
| Sensor-only P3 preparation | Both spatial folds built once from B0, graph weight zero, fixed B0 gauges/normalizers/Huber factors/ridge; prove equivalence without unused graph tensors |
| Matrix bridge | Exact A,b from assemble_system and explicit Q,d from query_residual's formula; verify Q@s-d agrees with the existing query-residual function |
| Controller and reports | Prospective executable registration and numerical limits; actual call/state/memory/time checks; teacher acceptance before further comparisons; immutable failures; CPU replay and backups |

The production sequence mirrors the original runner. Construct the **unwrapped pristine model**, extract its five frozen BF16 encoder features, and collect the 20 BF16-decoder/FP32-head teacher pairs. Reproduce the archived fused teacher depth/confidence and both fold gauges, normalizers, ridge and support/query counts. Stop on mismatch. Only afterward install the 144 LoRA modules using Torch seed42, verify the original initial adapter digest, freeze all parameters again, load the saved final P3 factors, verify their digest, and collect the second 20 pairs using the same feature objects. Keep original tensor references and verify their digest before/after; no optimizer is constructed and no network backward call is allowed.

Historical per-partner teacher confidence was not saved. Its reconstruction is checked through pinned deterministic source/runtime and the archived fused outputs; do not claim a historical per-partner byte match. Freeze this reconstructed confidence for all B0/B1/F0/F1 fusions. Read the existing FP32 P0/P3 raw first-camera XYZ caches after teacher acceptance; no new FP32 decoder forward is needed. D0's depth-only loader cannot supply exact XYZ fusion and must not be silently reused for this purpose.

The four states are pristine/final under the original training path (B0/B1) and saved fresh FP32 inference (F0/F1). Primary adaptation comparisons are B0→B1 and F0→F1. Secondary inference shifts are B0→F0 and B1→F1. Report per-partner and per-view fused changes on D0's same fixed 768-pixel grid and separate adaptation coordinates. Keep both spatial folds separate; do not select a favorable fold, confidence variant or pair.

Allowed archive members are explicit:

- Selected RGB input archive: `images` only.
- Earlier packet: `heads_25`, `heads_225`, `heads_425`, `heads_625`, `heads_825`, selected by registered `__rows`; column2 is axial depth and source frame must equal target−4.
- Split archive: only the corresponding `heads_earlier_FRAME__rows`, `__xy`, `__fold` members.
- Supplied pose archive: `heads_clean` only. Preserve the original `intrinsic()` function, including its 1.12 focal multiplier and pixel-center resizing.
- Selected P3 teacher NPZ: `depth`, `confidence`; selected P3 adapter: the 288 established factor members.
- P0/P3 raw forward caches: all 20 ordered pairs each; first-camera XYZ only for fusion.

Do not call `prepare_training()` wholesale: it also opens unused matching caches and prepares P1-only fields. P3's graph weight is zero. Use `partition` and `prepare_calibration(..., matches=[], ...)`, then test equality of sensor A,b,ridge and queries with the full preparation at graph weight zero. Match archived teacher sensor fields; do not claim to reproduce unused graph-filtering metadata.

For each fold and each inference path, refit s0/s1 using exactly the same frozen preparation. The finite decomposition is exact:

```text
s_k = solve(A_k^T A_k + lambda I, A_k^T b + lambda*1)
e_k = Q_k s_k - d
t = (Q1-Q0) s0
c = Q1 (s1-s0)
e1-e0 = t+c
```

Report all vectors, losses, norms, cosine, survival ratio, identity residuals and conditioning. Zero denominators are undefined. Opposing compensation is attenuation only if the surviving norm is smaller beyond the prospectively fixed arithmetic floor; overshoot can amplify the change. The direct term is a frozen-s0 counterfactual, not a forward-identical detached-gradient arm.

At final B1, use the output-space direction v=z_B1−z_B0 and the exact derivative below. Refit scales at every float64 perturbation z_B1±h*v. The pure math module retains all three fixed h values (0.01,0.003,0.001); its prospective numerical acceptance requires at least two passing, including the smallest, with residual-vector, loss and stationarity checks. This remains an output-space derivative, not a network parameter Jacobian. Fixed loss/norm resolution floors classify arithmetic changes only; they do not establish significance or accuracy.

Interpretation must follow both folds. If no resolved BF16 adaptation benefit exists in both, the premise that useful training geometry was lost downstream is unsupported. Benefit in BF16 that disappears/reverses in FP32 with common fusion supports a mismatch at this defined boundary. Surviving benefit in both paths leaves overfitting, matching and native alignment unresolved. Compensation along the observed direction does not itself prove it is harmful. A mixed fold result stays mixed. No outcome alone releases redesigned training or rentals.

Resource limits are 40 new decoder calls, five encoder-feature computations, no network backward/optimizer, one local job, at most60 minutes model work plus30 minutes CPU analysis. The design fixes 8192MiB allocated/9216MiB reserved CUDA ceilings and an8GiB host container cap. Freeze exact enforcement, source/runtime identities, teacher-comparison tolerances and commands in a separate executable registration before new access. Preserve the current-config/historical-M2-config distinction. A failed engineering check stops interpretation; do not quietly expand the probe.

**The exact derivative we need to diagnose**

For one fixed spatial fold in P2–P5, let A(phi) be the weighted support-plus-graph matrix, Q(phi) the weighted query matrix, b and d the fixed targets, and s the five fitted scales. Teacher gains, row weights, cameras, graph, normalizers and lambda are frozen. P2/P3 have graph weight zero; P5/P4 use 0.1. Write

\[
F=\tfrac12\|As-b\|^2+\tfrac\lambda2\|s-\mathbf1\|^2,
\quad H=A^TA+\lambda I,\quad
s=H^{-1}(A^Tb+\lambda\mathbf1),\quad r=As-b,
\quad e=Qs-d,\quad J=\tfrac12\|e\|^2.
\]

For a direction u in LoRA parameters, put dot A=D_phi A[u], dot Q=D_phi Q[u], t_S=dot A s and t_Q=dot Q s. Differentiating stationarity gives the **exact** local response

\[
\dot s=-H^{-1}(A^Tt_S+\dot A^Tr),\qquad
\dot e=t_Q-QH^{-1}A^Tt_S-QH^{-1}\dot A^Tr,
\qquad D_\phi J[u]=e^T\dot e.
\]

The detached comparator keeps only e^T t_Q at the **same** model state. Comparing derivatives at the final, different P3/P2 or P4/P5 states would confound the derivative rule with their training trajectories. A useful adjoint implementation is v=H^{-1}Q^T e, giving partial J/partial Q=e s^T and partial J/partial A=-(Av)s^T-rv^T, followed by each matrix's pullback to LoRA. All solves are five-dimensional; these expressions assume the registered weights and lambda do not change with phi.

Define P_lambda=I-AH^{-1}A^T. Then dot r=P_lambda t_S-AH^{-1}dot A^T r. Along a left singular direction of A with singular value sigma, P_lambda has eigenvalue lambda/(sigma^2+lambda); on the orthogonal complement it has eigenvalue one. With positive ridge it is symmetric positive semidefinite, generally **not** an orthogonal projector. Omitting the residual term dot A^T r is an approximation, not the derivative implemented by our solver.

For the linearized **inner** residual, with fixed-s sensitivity J_S, eliminating a nuisance increment yields the Gauss–Newton Schur matrix J_S^T P_lambda J_S. This is neither the exact Hessian nor the outer predictive curvature. The latter's Gauss–Newton matrix is J_eff,Q^T J_eff,Q, where J_eff,Q u=dot e above. Because Q and S differ and the outer loss omits the inner ridge, the envelope theorem cannot remove dot s. Only an outer objective identical to the minimized inner objective, including ridge and weights, has that cancellation.

There is a revealing limit: if lambda=0 and A has full column rank, a per-view multiplicative depth change A'=AD, Q'=QD, with D positive diagonal, is exactly absorbed by s'=D^{-1}s. Both calibrated support and query predictions stay unchanged. A fixed match graph does not remove this column-rescaling freedom. Positive ridge breaks exact invariance through the prior on s, but it can leave strong attenuation. This is a nuisance reparameterization argument, not evidence that the observed LoRA updates actually follow those directions.

**First separate three explanations**

| Explanation | Measurements from allowed information | Interpretation and next conditional test |
|---|---|---|
| A. Tiny effective neural change | Actual delta W per adapted layer; fixed-input point/depth changes; active gradients, clipping and optimizer updates | Nonzero factor updates alone do not establish a meaningful geometry update. Check numerical resolution and functional change before changing the objective. |
| B. Calibration absorbs the change | Direct t_Q, compensating Q dot s, their cosine, surviving dot e; support spectrum/ridge and residual-response term | Large raw change plus opposite, nearly cancelling calibration response supports absorption along the measured direction. A low training loss alone does not. |
| C. Surrogate improvement disappears downstream | Same-observation residuals and output changes at each precisely isolated inference/fusion/matching/alignment stage | A lower surrogate training residual can be lost by precision, confidence/matching changes or native nuisance optimization. It is not yet an accuracy improvement. |

For A, the actual layer update is delta W=(alpha/r)B_L A_L relative to the frozen backbone, not the concatenated factor norm; A_L/B_L denote LoRA factors, distinct from the calibration matrix A. A step changes it by (alpha/r)(delta B_L A_L+B_L delta A_L+delta B_L delta A_L). Report layerwise and aggregate Frobenius changes relative to original W and their effect on fixed outputs. The factor transformation A_L→cA_L, B_L→B_L/c leaves the function unchanged; `adapter_update_l2` is therefore not a functional distance. Small raw gradients also do not imply a small Adam step. At zero B_L, zero A_L-factor gradient is expected; it does not establish a disconnected adapter.

Measure RMS/quantiles of axial-depth change on the fixed adaptation pixels and a prospectively fixed image grid, using the same teacher gains/normalizers, feature precision and frozen confidence weights on both sides. Compare with a same-state repeat and the registered arithmetic's resolution. Keep invalid outputs visible; do not manufacture small changes by removing difficult pixels. First verify original-state hashes, 20-pair accumulation, eight updates, finite active gradients and clipping statistics from existing records. No learning-rate/rank/step sweep is justified by those records alone.

For B, report per fold: ||t_Q||, ||Q dot s||, their cosine, ||dot e||/(||t_Q||+epsilon), and e^T dot e versus e^T t_Q. The survival ratio can exceed one: calibration may amplify rather than absorb a change. Also report ||QH^{-1}dot A^Tr||, the stationarity residual ||A^Tr+lambda(s-1)||, and the sensor-only/combined singular spectra, lambda and condition number of H. Numerical invertibility from ridge is not data identifiability; a large stationarity error is a solver problem before it is an absorption diagnosis. Use fixed epsilon conventions declared before the diagnostic, and report a zero-direct-change case as undefined rather than a spectacular ratio.

Validate the response with a central difference that refits s at both perturbations, with frozen masks/weights and several declared step sizes. An output-space depth direction can be tested cheaply because A and Q are linear in depth under this fixed preparation; label it an output-space diagnostic. A final-minus-initial depth secant is **not** a LoRA Jacobian direction. A claim about a parameter update requires the actual network directional derivative at a specified state, or a corresponding finite difference in the LoRA factors. Do not confuse the inner Schur approximation with that test.

**Locate downstream loss of the surrogate change**

Training uses BF16 cached encoder features/decoder, FP32 heads, pretrained confidence fusion, fixed pretrained matches and supplied K/cameras. The endpoint uses fresh FP32 inference, adapted confidences/descriptors/matches, native 300/300 KDP optimization of anchors/poses/focals, then a fresh all-observation five-scale refinement in native cameras. Native within-cell depth ratios (`canon2`) and both point branches contain structure beyond the surrogate's fused self-camera axial depth. These operations are different; the current gradient bridge does not claim to differentiate them.

After M2, predeclare one representative scene/source and its matched full/detached arms using a reproducible rule, not a search for the largest improvement. Replay only what existing artifacts cannot answer, locally, without adapter updates. Record all added forward/alignment cost. The following sequence separates mechanisms; changing several elements at once does not identify one cause:

1. Re-evaluate both spatial objectives at the initial and saved final adapters using exactly the training precision, teacher state and frozen fusion/graph. Then change only fresh inference precision/features to the endpoint path. This isolates an arithmetic/inference mismatch before blaming geometry. Recorded step losses are pre-update; compare even steps with even and odd with odd. Alternating folds make adjacent losses different objectives.
2. With identical current point maps, compare fixed pretrained versus current predicted confidence fusion. Record per-view depth changes and allowed-observation residual changes. The fixed weighted average can cancel partner-specific corrections; report partner disagreement as well as fused change. Changing confidence can alter the endpoint even though training cannot optimize its confidence weights.
3. Inspect correspondence overlap, per-edge counts and connectivity before/after adaptation. If matching is implicated, register a paired native replay with current candidate geometry but explicitly fixed pretrained correspondence IDs, weights and view-tree policy, versus the fresh-matching path. Do not substitute entire old forward caches: that would also replace the candidate geometry. A correspondence change alone is not evidence that matching became better or worse.
4. Compare native versus final-refined changes in K, relative rotations, dimensionless center geometry, anchor depths, g0 and the five relative scales. Use adaptation-derived metric gauges for depths and translations; raw native units and absolute world origins cannot be compared directly. Compute fixed-observation diagnostic residuals with declared common weighting, while keeping the native objective's own trace separate. If native nuisance adjustment reverses a surrogate gain, it supports endpoint mismatch, not an accuracy verdict.

No stage may use the opposite-source packet to fit a gauge, choose matches, tune a weight or stop a replay. Final refinement uses all permitted observations, whereas the training solve uses only S; distinguish those fit sets in every table. A reduction on allowed observations may be overfitting. Only the already registered M2/M3 evaluations can establish useful generalization or surface accuracy.

**What is already recorded, and what is not**

Existing JSON evidence records per-step pre-update loss/scales, gradient/update norms and hashes, teacher preparation, final calibration diagnostics, matching hashes/counts and endpoint timing. Initial fused teacher depth/confidence, final adapters, forward caches and native/final predictions exist as sealed arrays, but this guide has not opened them. Per-step adapter tensors and post-update surrogate objectives are not archived; exact per-step delta W or gradient-response decompositions require a separately registered deterministic diagnostic replay. Do not reconstruct an optimizer trajectory from factor norms or digests.

**Bounded decisions after the evidence**

Use the smallest diagnosis that distinguishes A, B and C; do not automatically execute every branch. Fix a demonstrated implementation error as engineering, preserving the failed protocol. If the method is sound but ineffective, choose at most the contract's **one targeted local redesign across all failure branches**, with a measurable prediction and forward-matched control. A new objective/version must earn its own confirmation and cannot inherit the old gate.

Large measured absorption can trigger the M4 reduced-curvature diagnostic, using the correct predictive Jacobian and a classical damping control. A useful but unstable effect can trigger residual step sizing with full/half-step and matched-cost backtracking controls. Correlated covariance requires evidence of coherent source-calibration error and a diagonal comparator with identical marginals. Endpoint mismatch can motivate a bounded fixed-match refinement surrogate only after differentiability/gauge checks, with a forward-identical detached control. None of these branches is automatically justified by a negative M2 result.

Freeze the chosen hypothesis, diagnostic/ablation inventory, constants, stopping conditions and cost before running it. Do not launch a learning-rate sweep, several redesign families, or VGGT rental to rescue a failed result. The local rental allocation remains USD 0; the existing USD 25 reserve remains gated by the goal's accuracy/repeatability criteria. Report inconclusive diagnosis honestly and retain negative evidence; the mathematics above establishes identities, not novelty or an observed accuracy gain.
