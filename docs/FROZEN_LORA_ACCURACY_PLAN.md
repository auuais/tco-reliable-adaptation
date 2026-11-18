# Mathematical experiments for accurate reconstruction with frozen backbone LoRA

November 18, 2026. The primary experiment asks whether LoRA can learn geometry that remains accurate after a geometric solver responds to its predictions. Start on MASt3R, whose decoder adaptation already fits the RTX 3080. Transfer a successful objective to VGGT. The original backbone, heads and encoder stay frozen, and the learned matrices belong exclusively to LoRA. This is a proposed accuracy study. The accompanying synthetic derivative check verifies the small solver mathematics.

The recommendation follows three measured observations. Adapted coordinates produced most of our modest MASt3R gain. Per-view scaling reduced dense metric depth error by 6.60% and 4.74%, while reconstructed surface error fell by 1.24% and 0.81% and Heads regressed. Two directions explaining 86% of between-partner shape disagreement gave negligible held-source improvement. These motivate studying the response of calibration and alignment to neural updates. They leave open whether any proposed loss improves the final surface. [Crossover](../results/mast3r_geometry_matching_v1/README.md) · [Source-split experiments](../results/mast3r_partner_shape/README.md)

**What current methods contribute**

The architectural lesson is to distinguish camera calibration, metric scale and local surface shape explicitly. MapAnything factors reconstruction into rays, depths, poses and scale. DA3 predicts depth and rays. Pi3 predicts local pointmaps and camera poses with a permutation-equivariant architecture, and Pi3X adds metric and optional geometric inputs. These designs motivate a calibration layer around predictions. Their derived world-point outputs can satisfy internal geometric identities by construction, so testing consistency between two algebraically dependent outputs gives no learning signal. [MapAnything](https://arxiv.org/abs/2509.13414) · [DA3](https://arxiv.org/abs/2511.10647) · [Pi3 and Pi3X](https://github.com/yyfz/Pi3)

Crop-and-resize equivariance also has direct precedent. It can diagnose camera/scale sensitivity, but would need a further contribution as a primary method. The November 2026 Zoom In, Scale Out study examines this problem explicitly. [ICCV 2023 equivariance study](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhong_Improving_Equivariance_in_State-of-the-Art_Supervised_Depth_and_Normal_Predictors_ICCV_2023_paper.pdf) · [Zoom In, Scale Out](https://arxiv.org/abs/2611.17903)

VGGT and MASt3R serve as the target models in this project. Stronger contemporary models, including VGGT-Omega and prior-aware MapAnything/Pi3X, belong in the eventual benchmark when their checkpoints and required inputs are available. Benchmark rankings depend on task, supplied priors and evaluation alignment. [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) · [MapAnything benchmark framework](https://github.com/facebookresearch/map-anything)

| Prior work | Established mechanism | Consequence for this study |
|---|---|---|
| TTT3R | Controls the recurrent memory update using attention-derived confidence | Motivates testing how much of a LoRA update to accept |
| Test3R | Cross-pair geometric consistency with visual prompt adaptation | Compare the existing consistency objective explicitly |
| LoRA3D | Robust global alignment and confidence-calibrated pseudo-labels for LoRA | Alignment followed by distillation is a required control |
| TCO | LoRA adaptation using visual compatibility and geometric priors | Preserve an official-method comparator on compatible inputs |
| CAPA | Sparse-depth adaptation with per-iteration robust affine alignment | Re-solving scale or scale-and-shift alone has prior art |
| Efficient decoder adaptation | Frozen feature caching, decoder LoRA and affine alignment | Memory efficiency of the existing implementation is established practice |
| Free Geometry | Full/partial-view feature and relational distillation with LoRA | Plain subset consistency has close prior art |
| Self-Geometry | Fixed correspondences, multi-view/epipolar objectives and gradient disentanglement | Adding correspondence losses or gradient projection needs a stronger contribution |
| IntriAdapt | Known-intrinsics ray supervision for LoRA | Ray consistency by itself has direct prior art |

Sources are the official papers or authors' repositories. [TTT3R](https://arxiv.org/html/2509.26645v4) · [Test3R](https://arxiv.org/abs/2506.13750) · [LoRA3D](https://arxiv.org/html/2412.07746v1) · [TCO](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Learning_3D_Reconstruction_with_Priors_in_Test_Time_CVPR_2026_paper.pdf) · [CAPA](https://arxiv.org/html/2602.14751v1) · [Decoder adaptation](https://arxiv.org/abs/2603.01765) · [Free Geometry](https://arxiv.org/html/2604.14048v1) · [Self-Geometry](https://arxiv.org/html/2608.10708v2) · [IntriAdapt](https://arxiv.org/html/2610.24607v1)

The contribution under investigation is learning from disjoint sensor evidence through a graph-coupled calibration solve, together with a measured link between adapter updates and final surface improvement. Variable projection, implicit differentiation, Schur complements and trust regions are established mathematics. The novelty claim depends on the full formulation, relevant baselines and independent results.

**1. Learn through the response of calibration**

For each adapted linear layer write

\[
W_{\phi}=W_0+\frac{\alpha}{r}BA.
\]

Freeze every entry of the original state. Use the current rank 4, alpha 1 decoder attention adapter for the first MASt3R comparison. Initialize one factor randomly and the other to zero, giving exact zero-update behavior. Initializing both factors to zero kills both gradients. Reuse the existing 144 selected attention projections and record the approximately 1.03 million trainable adapter parameters. Reset adapters and optimizer independently for every arm and scene. Cached encoder features are safe because the encoder is frozen.

Let z_i(p;phi) denote predicted axial depth for target i after a fixed initial metric conversion. The raw depth is explicitly `canon[..., 2]` from the pinned MASt3R canonical fusion, using pretrained fusion confidences. Compute the conversion as g_i=median over S_i of measured depth divided by pretrained raw depth, then multiply subsequent raw depths by g_i. Freeze g_i separately for each support fold. This puts the scale prior in dimensionless units. With fixed supplied camera-to-world rotation R_i, center C_i and known K_i, define the camera ray q_i(p)=K_i^{-1}[u,v,1]^T with third component one. A small geometric layer estimates one positive relative scale per view,

\[
X_i(p;\phi,s)=C_i+s_iR_i q_i(p)z_i(p;\phi).
\]

This formulation makes depth and camera-center units explicit. If the camera centers are only known up to scale, translation scale becomes an additional nuisance variable and the system needs a metric anchor. The initial experiment uses the existing supplied metric cameras. Supplied-pose noise remains a separate stress condition.

Build a sparse observation matrix A(phi) from fitting observations S. A sensor row places z_i(p)/sigma_d in column i and measured axial depth d_i(p)/sigma_d in b. A match contributes three rows with R_i q_i(p)z_i(p)/sigma_m in column i, the negative corresponding vector in column j, and (C_j-C_i)/sigma_m in b. Normalize the sensor block equally by view and the correspondence block equally by unordered image pair. Start with frozen pretrained matches and confidence weights so the network cannot reduce the training loss by suppressing confidence. Fresh adapted matching is recomputed for final evaluation.

Specifically, multiply each sensor row and its right-hand side by sqrt(w_d/(V |S_i|)), with V=5. For an edge with M_ij matches, multiply all three rows of a match by sqrt(lambda_m c_ij w_m/(E |M_ij|)), where E counts nonempty unordered edges. The scalar robust weight is min(1,1/|r|) for a normalized initial depth residual and min(1,1/||r_xyz||) for the normalized initial 3D match residual, with weight one at zero. Form each raw match confidence as the geometric mean of the two positive pretrained fused confidences, then divide by its edge mean to obtain c_ij. Freeze these weights after support-only metric conversion. Query rows use 1/(sigma_d sqrt(V |Q_i|)) and no query-derived robust weights. Reject nonfinite or nonpositive confidence before constructing the fixed graph.

At fixed nonnegative weights the five-scale solution is

\[
s^*(\phi)=\arg\min_s\frac12\|A(\phi)s-b\|^2+
\frac\lambda2\|s-\bar s\|^2,
\qquad
H=A^TA+\lambda I,
\qquad
s^*=H^{-1}(A^Tb+\lambda\bar s).
\]

Use a float64 Cholesky solve and set the prior center to ones. Set lambda to 0.001 times trace(A_D0^T A_D0)/5 using the initial sensor-only matrix, identically across arms for each spatial fold. Freeze it through training. A ridge prior supplies numerical stability. Data singular values, per-view sensor support and graph connectivity separately describe how much the observations constrain the solution. A disconnected graph can still yield individually identifiable sensor scales. Report both facts. Negative scales invalidate the unconstrained pilot and require a separately tested positive-scale solver.

The pilot uses robust weights computed once from the pretrained support residuals, with the same weights across methods. This keeps the exact derivative unambiguous. A later IRLS version must distinguish differentiation through its iterations from an approximation that freezes its weights.

Now evaluate different sensor observations Q using this fitted geometry,

\[
\mathcal J(\phi)=\frac12\|Q(\phi)s^*(\phi)-d_Q\|^2.
\]

The derivative of the scale solution is

\[
ds^*=H^{-1}\left[dA^T(b-As^*)-A^T(dA)s^*\right].
\]

Writing e=Qs*-d_Q gives

\[
d\mathcal J=e^T\left[(dQ)s^*+Q\,ds^*\right].
\]

The first term measures the direct change in depth. The second measures the compensating response of geometric calibration. The experiment asks whether using both changes the learned surface in a useful direction.

There is an important limiting case. When the outer objective is exactly the minimized inner objective, including its ridge prior and all weights, its derivative with respect to the optimized nuisance variable vanishes at the optimum. The envelope theorem then permits ignoring ds*/dphi. An outer loss omitting the inner ridge need not have this property. A same-observation, same-objective comparison could therefore show no benefit from implicit gradients even with perfect code. Separate fitting and adaptation observations create the predictive question being tested here.

For a general smooth inner objective F_S(q,phi) and outer objective Q_Q(q,phi),

\[
\nabla_\phi\mathcal J
=Q_\phi-F_{\phi q}(F_{qq})^{-1}Q_q,
\]

at a differentiable isolated inner optimum after fixing gauge. A solve with an approximate Hessian defines an approximation that needs its own gradient check. Differentiable bundle adjustment has prior art. [BA-Net](https://arxiv.org/abs/1806.04807)

The mathematical check already passes on five synthetic cameras, 30 fitting depths, 20 distinct query depths and 30 matches. The full gradient is 3.666001177, finite differences give 3.666001196, and detaching the solve gives 7.094145105. This establishes the mechanism on a constructed example. [Replayable check](../scripts/check_lora_profiled_gradient.py) · [Numbers](../results/lora_profiled_design/math_checks.json)

**2. Separate adaptation observations from test observations**

The experiment has three roles for measurements. Earlier-source depth is the available adaptation packet. Spatially group its original sensor rays into two deterministic folds S and Q, then swap their roles on a fixed schedule. Use original sensor identifiers and native anchor-cell grouping to keep duplicate projections together. A query point influences LoRA and is therefore training data. Later-source depth is held-source development evaluation: exclude it from all fitting, weighting, per-run stopping and gauge initialization. Reveal its scores only after every registered prediction is sealed, then use the declared nomination rule to select at most one candidate. Repeat with earlier/later reversed. Because these scores select the method, they are not untouched publication-test evidence. Freeze new scenes and the final method before the subsequent generalization test.

Use a fixed seed and split whole sensor-ray groups, with at least eight fitting and eight query samples per target. If a view lacks that support, report the condition and revise the split prospectively before scoring. Keep sample counts, invalid depths and fit failures visible. Dense target depth remains evaluation-only. The two temporal folds share images and scene, so their results do not double the number of independent scenes.

For P2–P5, initial camera estimates, range gauges and masks must come from image-only inference, supplied calibration or S alone. The present D adapters trained on a whole earlier packet cannot initialize a clean S/Q experiment, because Q has already influenced them. Start each arm from the pretrained checkpoint. P1 retains the original D objective's use of the whole permitted adaptation packet; its gauges also use that packet. It remains a practical method comparator. P3/P2 and P4/P5 isolate the solver-gradient effect under identical splits. After training, all permitted adaptation observations may fit the final calibration for every arm.

**3. Run a small factorial that isolates the mechanism**

All new neural arms use identical LoRA placements, rank, full-image forward schedule, initialization, prediction refreshes, precision and optimizer budget. The five-scale solve remains a differentiable surrogate for native MASt3R alignment. Final measurements must include the usual fresh forward, fresh matching and native 300/300 alignment. Report that pipeline both before and after the same final five-scale refinement for every arm.

| Arm | Network update objective | What the comparison tests |
|---|---|---|
| P0 | Frozen network, graph calibration only | Pure geometric postprocessing |
| P1 | Existing sparse depth plus cross-pair D loss, with the same final calibration | Whether the new rule beats our current adapter |
| P2 | Fit scales with S, adapt on Q, detach fitted scales, graph weight zero | Cross-fitting without a solver gradient |
| P3 | Same as P2, differentiate scales | Effect of predictive implicit differentiation |
| P4 | Same as P3, include the fixed match graph | Added value of joint view calibration |
| P5 | Same as P4, detach fitted scales | Full forward-matched gradient ablation |

Historical D results and the existing 40-update implementation remain in the tables. They used a different schedule and cannot substitute for the matched P1 rerun.

Define each P1 macro-step as the arithmetic mean of the existing `objective` over the complete fixed `schedule(20)` triplet list in `run_mast3r_neural_transfer.py`. Preserve its depth plus 0.1 times cross-partner consistency, its initial per-pair median gauges, and its entire permitted sensor packet. Compute the 20 unique ordered pair outputs once, reuse their leaves across triplet terms, then sum their adjoints for replay. This fixes loss weighting while matching the decoder-forward budget of P2–P5.

Use eight complete gradient steps for P1–P5 initially. Each step sees all 20 directed pairs from five images. A two-pass implementation uses 20 graph-free forwards followed by 20 recomputed pair forwards and their vector-Jacobian products before one optimizer update. All five neural arms share this schedule. This is 160 differentiated pairs per case plus recomputation. Report actual decoder calls and wall time. The first screen contains 3 scenes × 2 temporal folds × 5 neural arms = 30 trained states, plus 6 frozen P0 reconstructions. A one-scene engineering check precedes it. Complete the prespecified screen if the gradient and runtime checks pass.

Concrete starting constants are in the manifest. Use Adam LR 1e-4, rank 4, alpha 1 and clipping at norm 1. Choose graph weight 0.1, normalize depth residuals by 2% of the fitting depth median, and use frozen support Huber weights with transition 1. Freeze ridge strength from the initial normalized support matrix. These are proposed engineering choices. A change after inspecting test scores creates a new study.

For edge i,j normalize world-space match residuals by 0.02 times the geometric mean of the two support-depth medians. These meter-valued constants standardize residual magnitudes. They are engineering scales and carry no calibrated covariance claim. Every arm receives the same final graph calibration at weight 0.1 after native alignment, fitted from all allowed adaptation observations with a fresh gauge and initial normalizer. Also publish the corresponding native outputs before this last operation.

The native endpoint uses the existing 300/300 KDP alignment with pose weight 0.01 and the same allowed observations and supplied calibration for every arm, including P0. Final five-scale refinement uses the resulting native optimized intrinsics and rotations. Let g_0 be the median of all valid measured/native axial-depth ratios over the permitted adaptation packet. Multiply native depths and native camera translations by this same global g_0, with centers C_i=g_0(C_i_native-C_0_native). Retain native rotations and K_i. Fit dimensionless per-view scales around one in this common metric frame and reconstruct X_i=C_i+s_i R_i K_i^{-1}p (g_0 z_i_native). Keep native matching correspondences fixed during this final solve. This explicitly avoids replacing optimized camera centers with supplied centers. The original native output and the final calibrated output require separate zero-adapter fidelity checks.

To nominate a method for the dense screen, require at least 1% lower mean held-source MAE than matched P1 in each temporal direction, an improvement on at least two of the same three scenes, and no numerical failures. The primary endpoint for this decision is native KDP followed by the common final five-scale refinement; native KDP alone is secondary. Also report its paired difference against P2/P3/P5. A candidate whose full and detached versions tie within fresh-run variation provides no evidence for the implicit-gradient mechanism. The 1% criterion limits local follow-up effort. The established 5% surface criterion governs a rental recommendation.

**4. Test physical uncertainty only after a predictive signal**

Depth samples projected from the same neighboring frame share camera and calibration errors. Treating fifty samples as fifty independent camera measurements can make an adapter learn a coherent registration error. Model their linearized residual as

\[
r=J_\eta\delta\eta+\epsilon,
\qquad
\delta\eta\sim\mathcal N(0,\Sigma_\eta),
\qquad
\epsilon\sim\mathcal N(0,D).
\]

Marginalizing the common perturbation gives

\[
C=D+J_\eta\Sigma_\eta J_\eta^T,
\quad
C^{-1}=D^{-1}-D^{-1}J_\eta
(\Sigma_\eta^{-1}+J_\eta^TD^{-1}J_\eta)^{-1}
J_\eta^TD^{-1}.
\]

Compute the Woodbury product in blocks. For a pose-only perturbation the small inverse is six-dimensional per source camera. Fit shared source noise at the source-frame level. The loss is one half r^T C^-1 r plus one half log det C. If C is frozen from initial geometry, the determinant is constant. If C is learned or recomputed as a differentiable prediction, its determinant term and derivatives matter.

Compare the surviving P arm under scalar Huber weights, diagonal covariance with identical marginal variances, and full correlated covariance. Keep the initial J_eta and covariance frozen in the first trial. Use controlled source-pose perturbations with known covariance, plus unperturbed input. A source-pose corruption must propagate through the sensor projection. The older pose-only stress packet, which keeps ranges fixed, tests a different channel.

This is a possible way to protect useful shape gradients from uncertain calibration directions. Its usefulness requires better surface accuracy under corruption and preserved clean accuracy. Classical uncertainty modeling alone cannot carry the paper. Direct comparisons with the project's earlier prior-weighting controls are required.

**5. Test an evidence-dependent LoRA step**

For a proposed adapter update u from Adam, let e_Q be the whitened query residual after the calibration solve and j be its directional derivative, including the solver response. A local quadratic model is

\[
\frac12\|e_Q+\beta j\|^2+\frac\gamma2\beta^2,
\quad
\beta^*=\operatorname{clip}_{[0,1]}
\frac{-e_Q^Tj}{j^Tj+\gamma}.
\]

The scalar controls the size of the entire LoRA parameter update. This takes TTT3R's idea of regulating adaptation strength into a measured residual objective. It is a classical damped one-dimensional step. Compare it with beta=1, beta=0.5 and ordinary backtracking at equal total forward cost. Same-objective improvement is an optimization property. Test accuracy remains an empirical outcome. At beta zero preserve the current adapter. Record the convention for Adam moments after rejected steps and hold it identical across controls.

Use this ablation only when P3/P4 has a positive but unstable effect. Otherwise it risks spending time controlling an unhelpful objective. Query observations used for the rule remain adaptation data, and each proposal refits the nuisance variables consistently.

**6. Examine which adapter changes are observable**

Linearize whitened residuals with respect to an adapter update delta_phi and nuisance calibration delta_q,

\[
r+J_\phi\delta\phi+J_q\delta q.
\]

Eliminating the nuisance update yields

\[
H_{\rm eff}=J_\phi^T J_\phi-
J_\phi^TJ_q(J_q^TJ_q+\Lambda)^{-1}J_q^TJ_\phi.
\]

This curvature describes which adapter changes survive calibration adjustment. For nonzero Lambda the corresponding residual operator is positive semidefinite and generally not idempotent. At zero regularization, pure per-view scale changes can be absorbed exactly by nuisance scales. Rank-four LoRA restricts a matrix factorization, while this observable subspace depends on the cameras and measurements. Those are distinct restrictions.

An optional diagnostic projects the update into the span of the depth gradient, match gradient and previous update. Compute only the 3 by 3 reduced curvature, damp it, and compare with ordinary damping under the same objective. Run this only if the main study demonstrates large calibration absorption or unstable gradients. It is a potential optimization improvement with a clear classical baseline.

**7. Implementation and gradient checks**

The existing native alignment has graph breaks in preprocessing and detached anchor offsets and outputs. A new loss callback alone cannot propagate into LoRA. Build the five-variable differentiable layer first and validate every derivative. A later native-alignment experiment can differentiate a fixed number of refinement iterations with fixed matches and a fixed view tree, then compare against detached refinement. Its claimed derivative must match those iterations; an approximate stationary-point formula needs convergence and gauge checks.

For MASt3R, derive each target's differentiable self-camera depth from its four ordered first-view point maps using `canon[..., 2]` from the pinned canonical fusion. The other return value, `canon2`, is a within-cell ratio and cannot directly enter a metric depth residual. Keep training confidence weights at their pretrained values for the initial factorial. Construct the initial unit conversion and spatial support from S only for the cross-fit arms. Verify the fused output against the corresponding upstream computation before applying the sensor layer. Both XYZ branches used by the loss must have an explicit coordinate convention. Avoid treating second-view points expressed in the first-camera frame as self-camera depths. This surrogate couples metric depth across views; native MASt3R alignment additionally changes per-cell anchor depths, poses and focals. Success on the surrogate requires confirmation through that native endpoint.

Use two-pass gradient accumulation for memory. First compute all required pair outputs at a fixed adapter state without retaining decoder activations. Treat those outputs as differentiable leaves, fuse, solve and calculate the loss, then obtain their adjoints. Recompute each pair at identical parameters, precision and masks, backpropagate its saved adjoints, then take one optimizer step after accumulating gradients from all 20 recomputed pairs. Compare with a retained-graph two-pair calculation. This is the chain rule with recomputation. Dropout, mutable buffers and inconsistent arithmetic can invalidate it.

| Existing component | Reuse |
|---|---|
| scripts/mast3r_network_priors.py | LoRA installation, frozen-state digest, pixel mapping |
| scripts/mast3r_network_capture.py | RGB views, frozen features, pair decoder, final capture |
| scripts/train_partner_shape.py | Source-specific restart and capture structure |
| scripts/prepare_partner_sources.py | Independently projected source observations |
| scripts/score_partner_shape.py | Final source-held evaluation and overlap handling |
| scripts/partner_bias_surface.py | Matched surface and metric-depth reporting |
| scripts/check_lora_profiled_gradient.py | Analytical solve and numerical derivative fixture |

The first production checks are zero-adapter fidelity, unchanged base-weight and buffer hashes, nonzero finite gradients to the intended LoRA factors, output-adjoint replay agreement, and central finite differences in a sampled adapter direction. At initialization the gradient of the random factor may legitimately be zero because the other factor is zero. Check the active factor. Require source/query disjointness before model execution and verify the final evaluator reads the candidate output.

For VGGT, use its pretrained image encoder and frozen task heads with LoRA in a predeclared decoder/aggregator attention subset. Global attention mixes images, so MASt3R's independent pair replay cannot be transplanted unchanged. Start with the existing measured two-view footprint, gradient checkpointing and a two-view backward test. Five-view decoder training and a native-alignment gradient may require 24–48 GB. Record the actual peak before choosing a rental.

**8. Accuracy evidence, stopping rules and budget**

Select at most one candidate after the registered screen. Run it on both sparse grids and clean/noisy pose packets with matched same-input controls and both dense references. Compare metric axial depth before alignment, surface accuracy/completeness after the existing similarity-plus-ICP protocol, normals, boundary error and worst-scene changes. Define boundary bins from evaluation reference geometry only after sealing. Include the final calibration in every baseline. Improvements caused by supplied poses or extra sensor points must be attributed to those inputs.

For rental advancement retain the 5% primary surface gain target against the strongest eligible control, no development-scene regression, at most 2% degradation in either distance component, and no aggregate regression on the alternate reference. These are internal resource criteria. Repeat the surviving case in three fresh processes and new corruption draws. Development folds provide correlated observations. For publication, freeze a new scene list before reading scores, seek at least 10 independent scenes in two public datasets, and report paired scene-level uncertainty. ScanNet++ offers sensor depth and independent scanner geometry, subject to its access process. Previously evaluated 7-Scenes and TUM windows remain observed data. [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/)

| Stage | Compute decision | Spending envelope |
|---|---|---:|
| Mathematical design and checks | CPU | USD 0 |
| MASt3R gradient checks and 30-state screen | Local 3080, measured memory and runtime | USD 0 rental |
| Matched dense evaluation and repeats | Local GPU/CPU | USD 0 rental |
| VGGT feasibility after a positive MASt3R signal | Rent only if local memory blocks the required setup | Up to USD 8 |
| One frozen transfer/confirmation batch | Same checked environment and cached downloads | Up to USD 12 |
| Failed-run/storage contingency | Included inside the remaining allocation | Up to USD 5 |

The conditional rental total is capped at the existing USD 25 reserve. Current balance and hourly rates must be checked when scheduling. Estimate cost from a measured smoke run and set a provider-side stop before the paid batch. Stage data and dependencies locally first. The prior stopped workspace is retained under the user's instruction. Respect the user's 5% weekly usage reserve. The plan commits to reporting a negative result when a candidate fails. A successful mathematical identity is insufficient to claim an accuracy improvement.
