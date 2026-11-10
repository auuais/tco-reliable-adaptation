# Reproduce the bounded-depth development screen

This screen is a postprocessing test. Each method receives **the same** TCO endpoint and initial depth. It does not update model parameters, bypass adaptation, or save 40 training steps. It tests whether part of the quality loss comes from how much dense shape changed while fitting the cameras.

For every pixel let `u = log(z_final / z_initial)`. Let `g` be the median of `u` over all pixels of all five views, and `r = u - g`. The common factor `exp(g)` is preserved. Keep the final camera center and ray, replacing its depth by `z_initial * exp(g + h(r))`. Unchanged TCO uses `h(r)=r`; fixed shrinkage uses `r/2`; frozen shape uses zero; the candidate clips `r` to `[-log(1.05), log(1.05)]`. This bounds local relative depth between 1/1.05 and 1.05 after removing the scene scale. The five-percent bound is fixed, not selected by searching the scores.

The camera fit is independent of reference geometry. The final camera and final depth are recovered from the model's **own** pixel/point correspondences. The preceding [coordinate control](../results/pointmap_precision_v9/RESULTS.md) validates recovery against known camera and depth heads; the [mechanism audit](../results/camera_shape_v10/RESULTS.md) reports all recovery outcomes. In a live TCO implementation the final predicted camera and depth heads are directly available. Stored endpoint reconstruction is an offline convenience.

**Protocol.** All 13 unchanged-calibration, both-prior v6 trajectories enter, including same-seed repeats. The primary comparison is nine paired outcomes on three public 7-Scenes development scenes and three seeds. Four extra executions complete two repeat groups. Tests use TCO's original released reference, and the normalizer is the zero-step point map reconstructed in FP32 on the same saved model heads. The scores are averages of accuracy and completeness divided by their scene-specific zero errors; both distances are also published separately. A normalized score below one need not imply improvement on *both* distances.

**Reproduction.** Use the project runtime and the extracted v6 evidence tree, which must contain `runs/`. The v6 and v9 summaries and all frozen source hashes are checked. Start from a clean results destination (in a fresh checkout, keep the committed evidence separately and generate outputs in a scratch copy). The runner refuses to overwrite previous outputs. Use these commands from the repository root of a fresh scratch clone. First preserve its published v11 folder under a new name; do not do this in an experiment workspace with work in progress:

    python -c "from pathlib import Path; p=Path('results/depth_trust_v11'); q=p.with_name('depth_trust_v11_published'); assert p.is_dir() and not q.exists(); p.rename(q)"

Then generate fresh predictions and results:

    python -m unittest discover -s scripts -p test_depth_trust.py
    python scripts/screen_depth_trust_v11.py
    python scripts/screen_depth_trust_v11.py --prepare --evidence /path/to/v6/evidence
    python scripts/screen_depth_trust_v11.py --score --evidence /path/to/v6/evidence
    python scripts/verify_public_development_tables.py
    python scripts/plot_depth_trust_v11.py

Preparation reads model predictions, not reference pointclouds or masks. It writes 52 predicted maps and a hash seal. The evaluation step verifies that seal, opens references, and scores the precomputed maps in fresh CPU processes. In our execution the seal was also [published before scoring](https://github.com/auuais/tco-reliable-adaptation/commit/2b5d35a). The 52 scores are paired counterfactual outputs on 13 trajectories, not 52 independent training runs or data sequences.

**Decision.** Compare the candidate to the lowest-score simple comparator among unchanged TCO, half change, frozen shape and zero adaptation. To pass it must gain 5% in normalized score, increase no scene error score by more than 1%, introduce no extra harmful cells, and exceed the observed same-seed range. This conservative resource rule is not a hypothesis test. Even passing on clean supplied cameras cannot establish robustness to wrong priors, cross-dataset success, or conference novelty.

[Completed results and paired plot](../results/depth_trust_v11/RESULTS.md) · [Funding decision](STAGE_DECISION_AFTER_DEPTH_TRUST.md)
