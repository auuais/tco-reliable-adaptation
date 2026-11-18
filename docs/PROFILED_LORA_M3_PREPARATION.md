# M3 dense confirmation: preparation contract

This is a preparation and implementation review, not an experiment result. No study prediction arrays, held-source scores or dense reference arrays were opened for this review. The M2 scientific sources remain frozen. This document does not authorize GPU rental or replace the goal, handoff or evaluation rules.

The controlling documents are `CALIBRATION_AWARE_LORA_GOAL.md`, `CALIBRATION_AWARE_LORA_HANDOFF.md` and `../manifests/lora_profiled_evaluation_v2.json`. They require eight dense gate cells and a separate executable registration before dense scores. The successor choices in this document were fixed prospectively on **November 18, 2026, before M2 scores were opened**. They resolve repeat and gate details without editing the bound M2 sources or changing the method. The executable M3 manifest and its synthetic checks must encode these choices exactly before any M3 score access. The nominee is then filled in by the existing M2 rule; its identity must not be chosen using dense results.

## What is missing

The existing implementation covers the centered, clean, three-scene, two-source-direction M2 screen. It does not yet implement the full M3 factorial. Five gaps matter:

1. There are no registered **signed-only shifted packets and spatial folds** for this study. Historical shifted packets combine both temporal sources and are unsuitable.
2. The case loader and native prior-row loader hardcode centered study paths. A new registered M3 case adapter must carry grid, pose draw, packet identity and split identity through both training and final reconstruction.
3. The fixed noise draw, exact repeat inventory, component aggregation and control selection defined below still need executable registration and validation.
4. Existing dense evaluation kernels are reusable, but their historical orchestration, baseline joins and cache identities do not constitute a valid M3 scorer.
5. Metric depth and boundary diagnostics need a common, reference-only definition linked to the final endpoint and both reference conventions.

No change to the calibration equations, adapter rank, optimizer schedule or final endpoint is needed to address these gaps.

## Signed-only shifted extraction

Reuse the geometry and provenance checks from `scripts/prepare_partner_sources.py`, `scripts/neighbor_sparse_depth.py:project_samples` and `scripts/prepare_profiled_inputs.py`. Do not call `neighbor_sparse_depth.packet()` or use `results/shifted_sparse_grid/measurements.npz`: that historical routine loops over both `target-4` and `target+4`, creating an information advantage over the current one-direction study.

Prepare separate packet archives for each grid and temporal direction:

| Field | Centered | Shifted |
| --- | --- | --- |
| Original sensor raster | 640 by 480 | 640 by 480 |
| First sensor pixel, x/y | 40, 30 | 20, 15 |
| Sensor spacing, x/y | 80, 60 | 80, 60 |
| Earlier source | target frame minus 4 only | target frame minus 4 only |
| Later source | target frame plus 4 only | target frame plus 4 only |
| Target frames | 25, 225, 425, 625, 825 | identical |

Preserve the current sensor intrinsics, calibration transform, clean source-to-target projection, 518 by 392 intermediate raster, source range validity and nearest-depth collision rule. In particular:

- Verify original source file hashes against the existing source index. Explicitly reject reading a target frame's depth image. Camera poses for projecting allowed source measurements are permitted inputs.
- Read only the selected direction's source depth when constructing that direction's packet. Do not union packets across directions or grids.
- Preserve the seven original columns: target x/y, target axial depth in meters, source frame, source x/y, source depth in millimeters. Keep original rows for the native KDP callback; do not reconstruct pseudo-source rows from native pixels.
- Reject nonfinite/nonpositive measurements; retain the established upper source range of 10,000 mm. Report projection exclusions and collision counts without adapting a threshold to reconstruction quality.
- Apply `prepare_profiled_inputs.native_selection` exactly for the 518-by-392 to 512-by-384 half-pixel remap and nearest-depth deduplication.
- Build independent shifted grouped folds with the current source-ray and native 8-by-8-cell union rule. Use the existing fixed split seed with a prospectively specified case key including the grid. Preserve the current centered folds byte-for-byte; do not regenerate their assignments under a renamed key.
- Check no source-ray or native-cell intersection across folds and at least eight support and eight query observations in every view. A failed minimum is an engineering/input failure; do not search split seeds or grid offsets.
- Freeze masks and selections before adaptation. Any exclusion based on pretrained prediction validity must be shared across all arms, prospectively documented and followed by the same minimum-count checks. Do not introduce candidate-dependent filtering.

The preparation report should include the exact allowed file inventory, count of target-depth files opened (zero), packet byte and tensor hashes, split byte and tensor hashes, row identities, group statistics and source reconstruction checks. Use a new output directory; preserve the original centered artifacts.

## Pose-only stress and the native endpoint

The stress channel changes **supplied target camera priors**, not the physically projected sparse measurements. Keep sparse packet values fixed between clean and noisy conditions. Perturbing source projection would define a different sensor-corruption experiment and must not be silently combined with this one.

The existing convention is in `run_mast3r_neural_transfer.pose_packets`:

1. Convert clean world-to-camera matrices to camera-to-world matrices.
2. Leave view zero unchanged.
3. For each later view, draw a normalized Gaussian axis and right-multiply its clean rotation by a five-degree axis-angle rotation.
4. Draw an independent normalized Gaussian direction and add a translation of 0.02 times the clean camera-center diameter.
5. Invert back to world-to-camera matrices. Use the same matrices for all arms, grids, temporal directions and all three fresh-process repetitions for a given scene.

The fixed seed is `20261112 + scene_index`, with scene indices 0, 1 and 2 for `[heads, chess, fire]`. Generate and hash each scene's noisy matrix packet once, then reuse those exact matrices in repetitions 0, 1 and 2. Register the seed, scene order, NumPy/SciPy versions, matrix hashes and formula before scoring; do not regenerate a different draw for a repetition or select a draw by difficulty or apparent improvement. Clean conditions likewise reuse identical camera matrices. This isolates execution variation under identical inputs.

There is only **one fixed five-degree/2%-diameter perturbation draw per scene** in the initial M3 gate. Passing it does not establish robustness across the corruption distribution, different magnitudes or different datasets. Additional distinct draws are optional stress experiments requiring separate prospective registration; they do not replace or count as the three same-input repeatability runs.

Pass the same selected camera packet to all relevant paths:

- The fixed cameras in P2-P5's profiled training system.
- P1's relative-camera targets, computed from the same supplied `w2c` and `c2w`.
- The native KDP `CalibratedPriors` callback for P0 and every trained arm.

This is compatible with the M2 common endpoint. `check_profiled_endpoints.capture_endpoint` already passes `case['w2c']` to `CalibratedPriors`. `mast3r_joint_prior_loss.JointPriors` uses supplied relative rotations and translations as fine-alignment loss targets with pose weight 0.01; it does **not** fix the returned native cameras to those priors. Native K, rotations and translations are optimized. The common final refiner uses those optimized outputs, never substitutes supplied centers, and applies the same all-allowed-observation operation in every arm.

The M3 adapter therefore changes input selection, not which method receives priors. Do not give a clean prior to one arm and a noisy prior to another; do not suppress KDP priors for P0; do not make the common final solve use supplied clean centers to cancel the stress.

## Minimal implementation path

Keep current bound modules unchanged while M2 runs. Add separately registered M3 modules and manifests, then compose the existing pure functions.

| New component | Reuse | Required difference |
| --- | --- | --- |
| Signed-grid preparer | `prepare_partner_sources`, `project_samples`, `native_selection`, `grouped_folds`, `verify_grouping` | Explicit grid/side/output arguments, fixed seeds, strict read inventory |
| M3 case loader | `profiled_network.load_case` schema | Resolve registered packet/split/pose paths; include original selected packet rows and all identities |
| Endpoint adapter | `check_profiled_endpoints.capture_endpoint`, `load_fresh_graph`, `profiled_final_calibration.refine_native` | Replace only hardcoded original-prior-row lookup; maintain identical callback, fresh caches and 300/300 iterations |
| Case runner/registry | `run_profiled_lora`, `register_profiled_execution` conventions | Registered grid, condition and draw in case ID; same numerical settings and failure semantics |
| Dense sealing/scoring wrapper | `seal_profiled_predictions` provenance pattern, dense kernels below | Full conditional inventory, both references, current controls, metric/normal/boundary diagnostics |

Before launching new model states, use CPU fixtures to verify case permutations, shared inputs across arms, clean/noisy propagation to all three camera consumers, native row equivalence, output gauge conversion, control selection and failure gates. Confirm the original centered/clean case adapter reproduces existing identities and values exactly. If any computation rather than input plumbing changes, determine which M1 checks need a new registered run; do not assume engineering acceptance transfers automatically.

## Endpoint conversion and dense kernels

The main endpoint is the saved **final-refined** artifact. Its `depth` is already `relative_scale * g0 * native_depth`, and its camera centers already equal `g0 * (native_center - native_center_0)`. Use its depth, native intrinsics and saved metric camera-to-world matrices directly. No extra gauge or per-view scale is applied during point construction.

For the secondary native-only endpoint, use `g0 * native_depth` and the same global metric camera-center conversion, without the five fitted relative scales. Check the saved main/native relationship, positive finite g0/scales, finite fields, exact raster shape and endpoint hashes as in the current scorer. Report both endpoints; never select between them using accuracy.

The established conversion path is:

1. `score_mast3r_transfer.resample`: bilinear depth resize from 384-by-512 to 392-by-518, with focal scaling and the matching half-pixel principal-point transform.
2. `fixed_camera_depth_scale.point_map`: pass resized depth, resized K, the inverse of the saved camera-to-world matrices and unit scales. It expects world-to-camera matrices.
3. Independently check `X_world = R_c2w * (z * inverse(K) * [u,v,1]) + C`. `replay_partner_surface_points.py` and `replay_mast3r_transfer` provide an independent resize/point reconstruction pattern, but their historical study wiring must be replaced.

`run_repeat_calibration.evaluate` is the existing surface kernel: a single scene-wide proper similarity using fixed reference pixels, then rigid point-to-point ICP with threshold 0.1 m, followed by accuracy/completeness and normal consistency. Pin this source, its imported metric/Umeyama functions, Open3D/NumPy/PyTorch versions, thread counts, random seeds and ICP/default normal settings. The current `score_depth_scale_study.evaluate_points` handles verified reference files and both reference labels, but its cache identity includes only prediction/reference identity. A new M3 wrapper must bind evaluator/runtime identity too, use a new cache root and retain alignment parameters. Historical `checked.json` files and joined historical baselines cannot substitute for fresh same-input evaluation.

Hash and seal converted pointmaps, endpoints, allowed inputs, reference archive identities and evaluation code before opening dense reference arrays. The reference archives and RGB identity metadata are indexed by `manifests/dense_focal_inputs.json` for each `scene_base`. Use `registered_rgb` as primary and `released` as alternate with the archived fixed validity mask. Do not use prediction confidence, positivity or error to shrink that mask.

### Metric depth, normals and boundary bins

Similarity plus ICP removes global gauge error, so a surface gain alone does not establish metric accuracy. Separately report axial depth in meters with no evaluation-fitted scale or per-view alignment. Convert reference points to reference axial depth using the reference's own pinned camera convention. `partner_bias_surface.py` illustrates the registered-RGB conversion, but its camera metadata file and reference convention must be explicitly bound; do not assume the released reference shares that convention. If only registered-RGB axial depth is defined, label that diagnostic accordingly rather than assigning an undocumented alternate-camera transform.

Reuse the reference-only region definition in `depth_spatial_support.partition`:

- `missing_neighbor`: valid pixels within two pixels of an invalid depth inside the raster.
- `jump_neighbor`: remaining valid pixels within two pixels of a horizontal/vertical valid-neighbor reference 3D separation greater than 0.05 m.
- `interior`: remaining valid pixels. Image borders alone are not missing-depth boundaries.

Compute the three masks separately for each reference with the same fixed validity mask. Report counts and errors for every scene/view/bin; check that counts partition the mask exactly. Reuse the **partition function**, not the historical `residuals`/`collect` pipeline, which fits an independent similarity per view and would change the M3 endpoint. The boundary diagnostics are axial absolute error for the registered-RGB reference and paired-pixel 3D residuals after the same single global similarity-plus-ICP used by the surface kernel. Label paired-pixel residuals distinctly from nearest-surface accuracy/completeness. Do not introduce extra transforms per region or view. Report the existing normal-consistency metric separately; it is not part of the scalar distance gate.

Reject nonfinite artifact fields. On each fixed reference mask, count nonpositive main-endpoint metric depths and disqualify any candidate with even one such pixel in any repetition. Retain all fixed-mask points in the auditable metrics; do not filter the invalid pixels out or substitute a smaller mask. Report the corresponding native-only counts and every control's invalid counts too. The historical surface kernel alone does not enforce positive depth. Boundary bins with zero valid points receive an explicit empty-bin status, not a fabricated zero error.

## Controls, repeats and the executable inventory

The mandatory eligible control pool is P0/P1/P2/P5. The nominee is one of P3/P4 chosen exclusively by the registered M2 rule. Both possible nominees are covered by this plan; do not rerun or pick the alternate nominee after seeing dense scores. Its matched detached comparator is already in the pool: P2 for P3, P5 for P4.

The complete minimum model-state inventory is:

`3 scenes * 2 directions * 2 grids * 2 supplied-pose conditions * 5 arms * 3 repetitions = 360 states`.

There are **three total** fresh-process repetitions, numbered 0, 1 and 2. Each begins from pretrained original weights and the registered zero-function LoRA initialization, with optimizer seed 42, the same allowed observations/folds, the same supplied camera matrices and the same runtime settings. Both clean and noisy repeats therefore measure same-input execution variation. Reuse the exact noise packet hashes specified above across all repetitions. There is no fourth mandatory repetition and no changed-corruption draw in this repeatability inventory.

The 30 centered/clean M2 states belonging to P0/P1/P2/P5 and the nominated candidate can supply repetition zero **only if** their original isolated process, complete artifact hashes, inputs, strict runtime and final endpoint match the M3 contract exactly. Bind their original receipts; do not copy historical unmatched predictions into the inventory. With that allowed reuse, 330 new states remain. All controls are paired by scene, side, grid, pose draw and repetition. The full factorial is a substantial local workload despite zero rental cost; measure case wall time and retained bytes before batching.

For a cell and repetition r, let `A[a,s,r]` and `C[a,s,r]` be the mean surface distances. Define:

`S[a,s,r] = 0.5 * (A[a,s,r]/A[P0,s,r] + C[a,s,r]/C[P0,s,r])`

`S[a,r] = mean_s S[a,s,r]`.

Use equal scene weights, never pooled pixels or an average of per-point percentage gains. All P0 denominators are from that same cell and repetition; reject zero/nonfinite denominators. For each cell, select control `B = argmin_a S[a,0]` over P0/P1/P2/P5 using the primary reference in repetition zero; exact ties prefer the smaller numeric arm ID. Freeze this selected control's identity for that cell across all repetitions, scenes, distance components and alternate-reference checks. Do not reselect a control from repetition 1 or 2, or separately for each metric. Report every control's score as well as the selected identity.

For the two component safety tests, preserve the existing normalized-component convention: compare `mean_s A[a,s,r]/A[P0,s,r]` and `mean_s C[a,s,r]/C[P0,s,r]` against the selected control B's corresponding quantities in the same repetition. Also report unnormalized meter-valued scene means. For the alternate reference, compute its own same-condition, same-repetition P0 normalization, retaining control B selected from primary repetition zero. Report the worst-scene ratio, individual scenes and both directions rather than hiding a failed cell in a pooled average.

Let F denote the nominated full-gradient arm. Each of the eight cells must satisfy all of the following, without pooling cells or selecting a repetition:

- Repetition zero primary gain `1 - S[F,0]/S[B,0] >= 0.05`.
- Primary gain `1 - S[F,r]/S[B,r] > 0` in **every** repetition r = 0, 1, 2. Repetitions 1 and 2 need positive gain, not a second 5% threshold.
- No scene regression in any repetition: `S[F,s,r] <= S[B,s,r]` for all three scenes and all repetitions.
- Neither normalized aggregate distance component exceeds 1.02 times B's corresponding component in any repetition.
- The alternate-reference aggregate score does not exceed B's alternate score in any repetition.
- All required artifacts, source/input identities, denominators, finite checks and candidate fixed-mask depth-positivity checks pass in every repetition. A numerical or validity failure cannot be removed by excluding pixels, dropping a run or selecting another attempt by its accuracy.

The full-versus-detached contrast is a separate mandatory mechanism gate, even if B is a different control. Define D = P2 when F = P3, and D = P5 when F = P4. For each cell, using primary normalized aggregate scores, require:

`S[D,r] - S[F,r] > 0` for r = 0, 1, 2;

`execution_floor = max_r abs(S[F,r] - S[F,0]) + max_r abs(S[D,r] - S[D,0])`;

`S[D,0] - S[F,0] > execution_floor + 1e-6`.

The 1e-6 margin is in **absolute normalized S units**, not meters or relative percent. These strict inequalities are fixed before scores. Report each actual contrast, the two floor terms, their sum and the resulting margin. Equal repeated scores give a zero empirical floor but still require improvement greater than 1e-6. This establishes a measured effect beyond observed same-input execution variation; it does not establish scene-level statistical significance or prove a general mechanism. The three repeats use the same three scenes and are not additional independent scene observations.

## Economical conditional execution and release gate

Register the entire inventory and stopping order up front. A faithful economical order is:

1. Verify M2's global prediction seal and apply its unchanged nomination rule. If no candidate qualifies, follow the existing diagnosis/one-redesign branch; do not execute M3.
2. For a nominee, seal and score centered/clean repetition zero in both temporal directions using the existing matched endpoints. A failed mandatory gate stops progression to additional conditions and invokes the registered diagnosis rule.
3. Run and seal the remaining repetition-zero conditions in a predetermined order, with all five arms and three scenes complete before opening a cell's scores. Stop only by the preregistered failure rule; list unexecuted cells as conditional, not passed.
4. If all eight primary cells pass, execute the two additional same-input paired repetitions and the specified mechanism confirmation. Preserve each cell's repetition-zero control identity. A failure remains a failure; do not select the best repetition, substitute a new draw or average away a failed direction/grid/pose condition.
5. Only a complete passing inventory can release the conditional transfer/generalization allocation. Local success is development evidence; the separate held-out/cross-dataset stage is still needed for a publication claim.

Before the first M3 score, the executable registration must contain all input/reference hashes, exact case IDs and conditional dependencies, nominee selection linkage, control pool, the single fixed noise draw, three total same-input repetitions, repetition-zero control selection, all thresholds and the mechanism formula above, invalid policy, component and boundary definitions, camera conventions, code/runtime hashes, output seal schema, command lines, source-freezing policy and failure/retry rules. Synthetic fixtures must test that a changed control identity or noise packet is rejected; a repeat with a scene/component/alternate/validity failure cannot pass; exactly 5% repetition-zero gain passes the gain threshold; zero repeat gain fails; and a mechanism contrast at or below `execution_floor + 1e-6` fails. Store full raw evidence on the existing C and E disks with verified checksums; commit only permitted compact evidence and code.

This document does not generate packets or run a scorer. Its repeat, control, threshold and validity choices are settled prospectively; the next implementation must encode and test them without reopening those choices after score access. The method, M2 nomination rule and rental caps remain unchanged.
