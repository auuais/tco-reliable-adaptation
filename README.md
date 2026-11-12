# Reliable test-time adaptation for 3D reconstruction

An independent research workspace for reproducing [TCO](https://github.com/cvlab-stonybrook/TCO) and studying when camera priors help or harm test-time adaptation. This is an exploratory study, not an official implementation or a claim of a new state of the art.

## Evidence so far

**Next research goal, registered November 13.** A local 48-alignment experiment will separate the effects of changing MASt3R's 3D coordinates and changing its correspondences and confidence weights. The first 24 alignments check archived controls. The other 24 cross the two sources on the same images. The USD 25 rental allocation stays reserved. [Execution handoff](docs/MAST3R_GEOMETRY_MATCHING_GOAL.md) · [Registered run list](manifests/mast3r_geometry_matching_v1.json)

**MASt3R neural adaptation, November 13.** All 24 adapter states, 960 updates, 105 reconstructions and 210 reference evaluations completed on the local RTX 3080 at about 4.58 GiB peak training allocation. Joint camera-supervised candidates were 0.91%/1.09% worse than the strongest controls with clean poses, and 0.05%/0.03% worse with noisy poses. All four advancement decisions fail. A fixed fresh-process repeat also failed numerical exactness/tolerance, with small measured surface differences. The goal pauses after this analysis at the user's request; later rental funds stay reserved. [Complete results](results/mast3r_neural_transfer/README.md) · [Repeat](results/mast3r_neural_repeat/README.md) · [Resume handoff](docs/MAST3R_ANALYSIS_HANDOFF.md)

**Neural-adapter feasibility, November 12.** One full-resolution MASt3R pairwise update completed on the RTX 3080 at 3.66 GiB peak allocation, with exact zero-update fidelity, finite adapter gradients and unchanged original-state hashes. This is an engineering result, not measured reconstruction improvement or complete TCO transfer. The initial input-setup failure is preserved. [Probe evidence](results/mast3r_adapter_probe/README.md)

**Latest MASt3R experiment, November 12.** Joint camera/depth constraints inside standard alignment reduced development reconstruction error by 7.63% and 8.66% versus ordinary MASt3R. Depth-only alignment supplied most of the gain; the joint increment was 1.43% and 1.29%, missing the registered 5% criterion. All 21 alignments and 42 evaluations completed locally. The subsequent neural study above evaluates weight updates; rental funds remain reserved. [Results and replay](results/mast3r_joint_priors/README.md) · [Next implementation](docs/MAST3R_NEXT_IMPLEMENTATION.md)

**Earlier evaluation, November 12.** The calibrated sparse-depth correction gained 3.32% and 3.78% on the original development scenes, then failed TUM confirmation. Its surface error was 0.20% and 0.01% higher than the strongest same-input controls. All 48 TUM outputs were sealed before target-depth scoring. The earlier held-out study also missed its threshold, and the earlier MASt3R transfer prototype did not improve aggregate error. Later rental allocations remain reserved. [TUM results and replay](results/tum_sparse_depth/README.md) · [Calibration development](results/scale_shift_sparse_depth/README.md) · [Held-out results](results/heldout_sparse_depth/README.md) · [MASt3R transfer](results/mast3r_transfer/README.md)

Independent distance checks passed in all nine fixed TUM cases, and a fresh process reproduced all twelve model-output tensors exactly. Three native normal-consistency checks failed. Follow-up tests linked the discrepancy to alignment-sensitive normal neighborhoods in quantized native geometry, with both neighbor backends agreeing on identical aligned clouds. All failed checks are retained. [Verification records](results/tum_sparse_depth/independent_surface.json) · [Neighborhood diagnosis](results/tum_sparse_depth/normal_neighborhoods.json)

In the 72-run dense pilot, adaptive focal selection reduced affected-camera log-focal error by 95.5% against the better unconditional focal policy. It improved registered 3D error by 6.38% against TCO with unchanged metadata. Frozen VGGT still had the lowest aggregate error, with adaptive TCO 3.92% higher. Both seeds missed the transfer criterion. MASt3R and held-out allocations remain reserved. [Dense reconstruction results](results/dense_focal_pilot/RESULTS.md) · [All GPU traces](results/dense_focal_pilot/gpu_traces.zip) · [Reproduction guide](results/dense_focal_pilot/REPLAY.md)

A focal check using the other four cameras to estimate VGGT's shared focal bias repeated on 15 different images from the same three development scenes. It corrected 30/30 distinct metadata faults and preserved 30/30 genuine digital zooms, matching the earlier design windows. Adding a two-fold geometric requirement vetoed true corrections and missed two replication criteria. The model-only result advances to a full TCO dense reconstruction pilot. Transfer and held-out performance remain open. [New-window results](results/relative_focal_windows/RESULTS.md) · [Design windows](results/relative_focal_confirmation/RESULTS.md) · [Dense pilot](docs/DENSE_FOCAL_PILOT.md)

Fresh image crops exposed a failure of the focal-repair rule under pose uncertainty. On three development windows it cut mean focal-gradient error by 66.9% against the stronger unconditional baseline but edited 9/60 valid zooms, above the 3/60 ceiling. Frozen VGGT responded in the correct focal direction on all 30 paired zooms. Transfer spending remains on hold. Separately, all 360 focal ambiguities in archived full-model trajectories required a decision within six updates, so delaying visual checks offered no full-run matcher saving there. [Fresh-image experiment](results/focal_zoom_gradient/RESULTS.md) · [CVPR visual comparison](results/visual_focal_baseline/RESULTS.md) · [Optimizer trace replay](results/focal_gradient_deferral/RESULTS.md)

On a new five-frame window in each of three development sequences, all 12 full-model TCO calibration runs completed on one RTX 3090. Correcting only the supplied intrinsics reduced normalized reconstruction error relative to original TCO by 6.38% on the released reference and 12.57% on the registered RGB reference. On Fire the released-reference score worsened by 7.43%. The preregistered joint pose-and-intrinsics correction missed its advancement criterion. Calibration-stage spending is estimated at $0.95–$0.99 of $3. [Full model results](results/canonical_sensor_factorial/RESULTS.md) · [Reproduce the experiment](results/canonical_sensor_factorial/REPLAY.md)

A local focal-prior test kept the supplied camera matrices identical between synthetic metadata faults and legitimate image-coordinate zooms. RGB tracks corrected 201 of 270 faults and incorrectly edited 4 of 270 valid zooms. This development result justifies testing the effect on TCO gradients under pose uncertainty. The $25 MASt3R allocation remains reserved. [Matched focal controls](results/focal_metadata_control/RESULTS.md)

**Decision recorded November 11.** The USD 25 MASt3R allocation remains unspent. On three development scenes the learned camera-frame correction missed the advancement criterion. Unchanged TCO itself showed 6.56% lower normalized reconstruction error than frozen VGGT under the released reference, 0.91% higher under registered RGB and 1.49% lower under RGB SfM. All nine paired substitutions of observed identical-seed reruns reduced TCO's apparent advantage when the reference changed from released to registered RGB. [108 dense evaluations](results/sensor_extrinsic_v15/RESULTS.md) · [Rerun sensitivity](results/reference_repeat_sensitivity/RESULTS.md)

Across nine disjoint five-frame windows in the same sequences, a fixed camera-frame and intrinsics correction reduced registered-depth reprojection error from a mean median of 13.03 px to 5.29 px. A CPU replay from the original Microsoft ZIPs matches all 45 camera-condition scores within 0.000026 px. [Camera consistency](results/sensor_windows_v16/RESULTS.md) · [Public-data replay](results/sensor_public_replay/README.md)

Using just the RGB matches and supplied cameras, the highest epipolar-error camera identified the deliberately corrupted prior in 75.6% of rotation cases, 79.3% of translation cases and 71.1% of focal cases under the calibrated convention. The stricter alert-and-localize rates were 66.7%, 72.6% and 45.9%, with one alert on nine clean windows. On a separate 40-update Heads timing comparison, staged execution reduced peak GPU allocation from 8.63 to 6.71 GiB and ran faster, but its reconstruction had larger distance errors under both references. [Single-camera fault localization](results/single_camera_fault_localization/RESULTS.md) · [Matched RTX 3080 comparison](results/staged_heads_bridge/RESULTS.md)

Fitting a correction for the flagged camera reduced mean virtual projection error by 5.3%, missed the 20% target and damaged a clean Heads window in both track folds. Every alerted proposal on corrupted inputs also passed validation on disjoint RGB tracks. Agreement on those tracks did not ensure accurate camera geometry. [Selective camera repair](results/single_camera_repair/RESULTS.md)

At the camera-head output, dropping the flagged camera increased mean gradient error by 90.2%. A partial weight fitted on the other two development scenes increased it by 27.1%. Even choosing the best weight separately with the clean gradient could improve it by only 18.6% with this detector. The bound applies to these 180 synthetic faults and this family of camera losses. [Recorded gradients](results/masked_camera_gradients/RESULTS.md) · [Soft-weight bounds](results/soft_camera_gradient_bounds/RESULTS.md)

Earlier mechanisms include an RGB-track selector that lost under clean and perturbed priors, and a crossed camera/depth experiment that identified a large scene-specific interaction. [Camera selection](results/track_camera_selection_v13/RESULTS.md) · [Depth and camera swaps](results/camera_depth_swap_v12/RESULTS.md)

The [40-job repeatability/calibration study](results/repeat_calibration_v6/RESULTS.md) completed under its USD 3 cap (approximately USD 0.72; GPU stopped). Axis-aware nominal resize improved the aggregate score by only 2.1%, below the frozen gate and observed execution variation. The 525-focal variant worsened it. Identical captured initialization and RNG still led to different adapted weights. The fixed sparse-match proxy abstained on every adapted run. [Later spending is withheld pending redesign](docs/DECISION_AFTER_REPEAT_CALIBRATION_V6.md); no MASt3R or held-out success is claimed.

A [matched-head coordinate precision control](results/pointmap_precision_v9/RESULTS.md) found that our zero-step and adapted output paths use different unprojection precision. On three development scenes this alone moved normal consistency by 0.059–0.084 and changed distance errors in opposite directions. Interpret historical zero-relative gains with this caveat; the calibration gate still fails.

The initial five-sequence 7-Scenes subset showed mean aligned reconstruction accuracy error decreasing from **1.770 to 1.415 cm**, and completeness error from **2.056 to 1.629 cm**, after 40 TCO steps. This is partial reproduction on selected sequences, not reproduction of the full published benchmark.

Follow-up execution repeats exposed substantial variation: on Heads, clean 40-step adaptation harmed accuracy or completeness in 2 of 3 executions. Severe synthetic camera-prior corruption harmed all executions on the two inspected sequences. Three execution repeats with the same seed are not three independent optimization seeds. These observations motivate stronger baselines and reliability experiments; they do not establish a novel method.

![Repeated exploratory screen](results/diagnostics_v3_repeats/repeated_screen.png)

Aggregate results are in [results](results/). See [reproduction boundaries](docs/REPRODUCTION.md), [third-party sources](docs/THIRD_PARTY.md), and frozen manifests. Historical runners are retained without methodological edits.

The [research plan](docs/RESEARCH_PLAN.md), [USD 200 budget](docs/BUDGET.md), and [implementation handoff](docs/IMPLEMENTATION_HANDOFF.md) define the stages. The first new study was a precommitted 39-job learning-rate and visual-only baseline screen.

That [39-run screen is complete](results/baselines_v4/RESULTS.md). LR 0.0003 reduced the average combined distance-error score by 2.6% relative to the released setting, with a normal-consistency tradeoff. Visual-only adaptation helped Fire but harmed Heads in that batch. The [decision from that screen](docs/DECISION_AFTER_BASELINES_V4.md) was to separate pose and intrinsics constraints before developing selective adaptation. These are development observations, not a validated new method.

The subsequent [pose/intrinsics ablation is complete](results/prior_ablation_v5/RESULTS.md): 39 validated runs favor both penalties overall (score 0.9269; 2/9 harmful executions), versus pose-only (0.9506; 5/9) and intrinsics-only (0.9391; 3/9). The earlier visual-only Fire benefit did not repeat; visual-only was unstable in this new batch. A [calibration audit](docs/PRIOR_ABLATION_V5.md) identified focal, resize and camera-parameterization discrepancies without changing the benchmark inputs. The [current decision](docs/DECISION_AFTER_PRIOR_ABLATION_V5.md) is to retain both penalties and isolate repeatability/calibration effects before a controller or MASt3R transfer. The session cost approximately USD 0.66; GPU compute is stopped.

A separate [local memory probe](results/local_memory_v3_full40/RESULTS.md) now completes 40 full-resolution adaptation steps on the RTX 3080 at 8.63 GiB peak allocated memory by offloading inactive modules. Matched zero-step outputs were bitwise identical. Reserved memory reached 10.57 GiB in Windows/WSL, so this is not a universal physical 10 GB fit guarantee or a speed/accuracy equivalence result. A [pose-loss precision audit](results/rotation_precision_v6/RESULTS.md) motivates a separately frozen local engineering comparison. [Mathematical notes](docs/MATH_NOTES.md) explain the bias, stopping and observability hypotheses behind the research plan.

## Setup and use

Linux with CUDA is the reference execution environment. Local development uses an RTX 3080; full five-view 518-pixel adaptation was measured at about 11.5 GiB allocated memory on a 48 GB RTX 6000 Ada. The measured configuration does not fit a 10 GB 3080 without changing memory behavior. Reduced two-view 224-pixel runs are debugging only.

```bash
python scripts/setup_upstream.py
docker build -f environment/Dockerfile -t tco-research .
python scripts/download_assets.py --root /path/to/assets --scenes heads chess fire
```

Prepare the public dataset using the registration protocol linked in [THIRD_PARTY.md](docs/THIRD_PARTY.md), preserving original frame IDs. The local vectorized preprocessing utility is excluded because it derives from separately licensed SimpleRecon code. The public source and registration requirements are documented so a researcher can obtain it directly. No datasets, checkpoints, or external source trees are redistributed here. This setup fetches dependencies over the network; the hashed wheel inventory records our reference cloud environment, not a universal cross-platform lock.

With prepared data mounted at `/data` and this repository at `/workspace`:

```bash
python scripts/run_baseline.py --run-id example_heads_518_s40_seed42 \
  --data-root /data/data/7scenes --checkpoint /data/checkpoints/VGGT-1B \
  --scene heads/seq-01 --frames 0 200 400 600 800 --size 518 --steps 40 --seed 42
```

Each run starts a fresh model and records configuration, source/input hashes, finite gradients, timing, memory and evaluation results. It refuses to overwrite an existing run directory. Raw evidence remains local; public aggregates do not contain private cloud connection or billing information.

CPU validation:

```bash
python -m unittest discover -s scripts -p 'test_*.py'
```

The CPU validation suite requires NumPy, CPU PyTorch 2.7.1, OmegaConf and Matplotlib, plus the pinned upstream checkout for camera-loss tests. The GitHub workflow installs these explicitly and checks both v4 and v5 frozen manifests. GPU inference also requires the upstream dependencies and Dockerfile. There is no blanket license grant over upstream projects, datasets or checkpoints; consult their own terms.
