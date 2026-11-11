# Replaying the dense focal-calibration pilot

### Verify the published observations on CPU

From a checkout containing the completed results, the following commands replay the 72 GPU records, the 40 logged updates in each record, the focal decisions from the observed model-head predictions, and the paired reconstruction comparisons. These commands use the published metric measurements. Recomputing a nearest-neighbor distance from the original point coordinates requires the full replay below. [CPU environment](../../.github/workflows/cpu-validation.yml) · [Experiment design](../../docs/DENSE_FOCAL_PILOT.md)

```sh
python scripts/score_dense_focal_pilot.py verify
python scripts/release_dense_focal_traces.py verify
python scripts/dense_focal_headroom.py verify
python scripts/diagnose_dense_focal_readout.py replay
python scripts/plot_dense_focal_pilot.py
```

### Recreate the predictions

Use a separate Linux checkout of the research repository at `01b69605ff41a3829e33277cd3f96816156f02ec`. It contains the execution manifest and predates the prediction seal. Use original TCO at `65387333877723cf99086f7f29089379dd50c59a`, the pinned VGGT weights and the Heads, Chess and Fire 7-Scenes data. The parent replay gives the original public downloads, dataset directory layout, short dataset-cache recipe, environment and compatibility patch. [Parent full-model setup](../canonical_sensor_factorial/REPLAY.md) · [Public input checksums](../../manifests/dense_focal_inputs.json) · [Camera replication](../relative_focal_windows/REPLAY.md)

Prepare RGB, supplied cameras and both depth references locally. The geometry check reprojects the prepared reference points onto the cropped grid. Keep the `evaluation` directory on the scoring machine. The GPU needs only the inference files, checkpoint and research checkout. Our model-only policy uses the original initial VGGT camera predictions at execution time. It requires no feature-matching inference.

```sh
python scripts/prepare_dense_focal_pilot.py --assets /data
python scripts/check_dense_focal_inputs.py --assets /data
python scripts/run_dense_focal_pilot.py dry-run --inputs reports/dense_focal_pilot/inference --checkpoint /data/checkpoints/VGGT-1B
python scripts/run_dense_focal_pilot.py run --inputs reports/dense_focal_pilot/inference --checkpoint /data/checkpoints/VGGT-1B
```

Run all 72 conditions on the same GPU class for a matched hardware replication. The original experiment used one 24 GiB RTX 3090. Every execution started a fresh process and model. `reports/dense_focal_pilot/terminal` records attempts, while `runs/dense_focal_pilot` contains the original and final point maps, adapter tensors, random states and optimizer traces. Preserve these two directories when moving to the CPU machine. The two copies of our original raw archives are inventoried with hashes and sizes in the session record. [Session record](session.json)

### Recompute dense metrics from predictions

Use the locally prepared RGB and evaluation files with the raw predictions. Seal their hashes before scoring. The original run published its seal before launching the CPU evaluator. The evaluator validates both the reference archive and the dense tensor bytes and reuses a calculation only for the same reference and point tensor. Each scored job includes initial and final accuracy, completeness, and normal consistency under both reference conventions. [Original prediction seal](seal.json)

```sh
python scripts/score_dense_focal_pilot.py seal
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/score_dense_focal_pilot.py evaluate --workers 3
python scripts/score_dense_focal_pilot.py publish
python scripts/score_dense_focal_pilot.py verify
```
