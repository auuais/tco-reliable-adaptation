# Pose and intrinsics component ablation

Prepared 2026-11-11 before cloud execution. This is a development experiment on three previously inspected sequences, not a held-out benchmark or a new-method claim.

## Question and frozen design

Does the benefit of TCO's camera constraints come from pose, intrinsics, or their interaction? We rerun all four conditions on one physical GPU, at the same learning rate, frame selection, checkpoint, visual loss and evaluation protocol. Historical v4 results provide motivation only; they are not substituted for contemporary controls.

| Condition | Pose coefficient | Intrinsics coefficient | Visual coefficient |
|---|---:|---:|---:|
| Both | 1 | 0.01 | 0.2 |
| Pose only | 1 | 0 | 0.2 |
| Intrinsics only | 0 | 0.01 | 0.2 |
| Visual only | 0 | 0 | 0.2 |

Pose contains the released rotation and normalized translation terms, with translation weight 2. The visual term is the released RGB multiview term. Reference depth and reconstruction error are evaluation-only. Pose-only and intrinsics-only name the active camera penalty; both retain visual adaptation.

Three scenes (Heads seq-01, Chess seq-03, Fire seq-03), three seeds (42, 123, 2026), four adaptation conditions and three fresh zero-step controls give **39 jobs / 1,440 updates**. Each adapted run uses 40 Adam steps at LR 0.0003, with five frames (0, 200, 400, 600, 800), input size 392 x 518. Every run starts in a fresh process from the original checkpoint. Condition order is deterministically shuffled within each seed block. The manifest records the complete schedule and source hashes before execution.

Primary score is the mean of accuracy / baseline accuracy and completeness / baseline completeness. Lower is better; 1 is the same-scene zero-step control. Summaries weight scenes equally. Report each execution, observed three-seed range, and paired within-scene/seed contrasts. Interaction is `both - pose_only - intrinsics_only + visual`. A negative interaction means the combined penalties outperform the additive prediction on this score; it does not establish statistical significance. A harmful execution has either distance metric more than 1% above its control. This flag is descriptive, not a population failure probability.

## Calibration audit and its implications

The independent audit examined all 15 selected development frames and did not mutate inference or evaluation tensors. Camera convention, depth storage, and exact-inverse projection round trips passed. Round trips establish internal coordinate consistency, not physical calibration accuracy.

Two discrepancies require a separate follow-up:

1. The depth-to-RGB registration implementation uses RGB focal length 525 pixels (depth focal 585), while TCO's released loader uses nominal focal 554.256258 pixels, 5.5726% higher. Neither value has been independently certified as the physical camera calibration in this project.
2. Resizing 640 x 480 to 518 x 392 applies x/y image scales 0.809375 and 0.816667. The released intrinsic transform applies the larger scale to both axes. Against the pixel-center-aware image affine, this raises fx by 0.9009%, shifts cx by 2.337 pixels, and gives a maximum displacement of 4.662 pixels among five image-coordinate probes.

| Intrinsics at 518 x 392 | fx | fy | cx | cy |
|---|---:|---:|---:|---:|
| Released loader | 452.6426 | 452.6426 | 261.2417 | 195.9083 |
| Axis-aware, nominal focal | 448.6012 | 452.6426 | 258.9047 | 195.9083 |
| Axis-aware, registration focal | 424.9219 | 428.7500 | 258.9047 | 195.9083 |

Some stored pose rotations are slightly nonorthonormal. The audit separately records the residual from the release's transpose-as-inverse convention and the exact inverse of the actual transform used for reference backprojection. It does not silently orthogonalize or replace poses.

Sources: [pinned TCO loader](https://github.com/cvlab-stonybrook/TCO/blob/65387333877723cf99086f7f29089379dd50c59a/datasets/sevenscenes.py), [pinned resize utility](https://github.com/cvlab-stonybrook/TCO/blob/65387333877723cf99086f7f29089379dd50c59a/datasets/utils/cropping.py), and [SimpleRecon registration](https://github.com/nianticlabs/simplerecon/blob/main/data_scripts/7scenes_preprocessing.py). Numerical evidence and source hashes are in `results/calibration_v5/summary.json`.

The factorial study **keeps released calibration and evaluation unchanged**. These findings do not yet explain any degradation or prove that corrected calibration improves reconstruction. A later study must separate axis-aware resizing from the registration-focal choice, rerun zero-step controls, and explicitly account for calibration-dependent reference point clouds. Never claim gains by silently changing only the reference geometry. Sim(3) and ICP alignment also mean current metrics do not establish absolute metric scale or SLAM drift.

## Local acceptance and cloud execution

The local RTX 3080 passed 21 unit tests, including actual upstream camera loss/gradient isolation, provenance rejection and synthetic projection checks. Six real-GPU integration runs used two 168 x 224 views: two zero-step controls and one update for each condition. The zero-step predictions were bitwise equal, inputs/priors/reference targets matched, all four backward passes completed, and frozen base parameters remained unchanged. This validates implementation plumbing, not full-resolution quality. Raw smoke evidence remains local; its public summary is in `results/prior_ablation_v5_preflight/smoke.json`.

Freeze with `python scripts/run_prior_ablation_batch.py --freeze`, then verify with `--dry-run`. Commit and push the complete code and manifest before starting paid compute. The cloud wrapper verifies cached dependencies, input/checkpoint hashes, renderer and GPU compatibility; reruns tests and the calibration audit; executes fresh jobs; validates all 39 outputs; and packages evidence even after failure. Stop on the first unexpected failure, with no automatic retries or overwritten attempts.

Session ceiling is **USD 3**, charged to the existing USD 20 prior-ablation allocation within the USD 200 overall ceiling. The experiment has a 70-minute deadline and each job a 360-second timeout. Collection and setup also consume rental time. Verify the live quote, retain a shutdown margin, and explicitly stop the provider GPU after collecting and checksumming the archive. A script timeout alone does not stop provider billing. Do analysis and publication locally after shutdown.

## Decision rules

- If one camera penalty consistently accounts for the benefit, focus the next mechanistic study on its residual scaling, calibration sensitivity and interaction with visual gradients.
- If rankings differ by scene, test why before designing an adaptive controller. Scene-specific oracle selection is not an implementable online method.
- If observed variation overlaps the candidate benefit, repeat fixed configurations before claiming a contribution.
- Treat calibration correction as a separately controlled hypothesis. A fix to an input transform can be useful engineering evidence without constituting a high-end conference contribution.
- Freeze any proposed selection rule without reconstruction ground truth, then evaluate on held-out scenes and a second dataset before transferring it to MASt3R.

## Additional local parameterization check

After freezing the cloud experiment, a separate CPU derivative audit inspected the released decoder and intrinsics loss. It changes no cloud inputs, sources or evaluation settings. The decoder fixes the principal point to `(W/2, H/2) = (259, 196)`; the supplied released prior is approximately `(261.2417, 195.9083)`. The principal-point contribution to the intrinsics loss is therefore a constant, approximately 0.290978 before the 0.01 weight, or 0.00290978 after weighting. Its gradient with respect to the pose encoding is exactly zero in the tested fixture, and removing that constant leaves the full encoding gradient bitwise identical.

This constant does not itself cause bad updates. It matters for interpreting absolute residual magnitudes or designing future residual-based gates. The focal-length portion uses Smooth L1 on focal lengths in pixels, despite an upstream comment describing it as relative. Its scale and transition point therefore depend on resolution; normalized camera residuals are a separate hypothesis to test, not an improvement already demonstrated here.

Evidence: `results/calibration_v5/parameterization.json`, generated by `scripts/audit_intrinsics_parameterization.py` using the actual pinned decoder and loss. Sources: [camera decoder](https://github.com/cvlab-stonybrook/TCO/blob/65387333877723cf99086f7f29089379dd50c59a/utils/geometry.py) and [intrinsics loss](https://github.com/cvlab-stonybrook/TCO/blob/65387333877723cf99086f7f29089379dd50c59a/constraints.py). The test uses FP32 CPU arithmetic and does not substitute for CUDA adaptation or establish a new-method gain.
