# Dense focal pilot preparation

The local preparation produced nine distinct 518 × 392 five-view image batches, six supplied camera rigs and eighteen cropped RGB-D references. Each metadata-fault/valid-zoom pair shares exactly the same supplied rig. Only the RGB batches and provided cameras are in the inference payload. The two evaluation references and their depth-source hashes stay on the scoring machine. The 25 preparation artifacts were checked on two drives. [Input hashes and job grid](../../manifests/dense_focal_inputs.json) · [Registered dense experiment](../../docs/DENSE_FOCAL_PILOT.md)

An independent projection calculation checked all 15,317,624 valid reference points against the cropped RGB pixel coordinates. The largest reprojection discrepancy was 0.000080 pixel. Corresponding registered and released points retained the same axial depth within 0.00000024 metre. [Per-batch checks](input_geometry_checks.json)

As an end-to-end check of the scoring path, we submitted the Heads reference point cloud itself as the prediction. The actual similarity alignment, ICP, nearest-neighbor distances and normal evaluation ran on 811,621 points per reference. Mean accuracy and completeness errors were 0.34 micrometre on registered RGB and 0.24 micrometre on the released geometry. Normal consistency exceeded 0.99999 for both. [Identity metrics](metric_identity.json)

One original full-size TCO forward pass on CPU exercised the focal selection before adapter optimization. On the Heads base input with a spurious longer focal, the relative rule selected the neighbor median. The unadapted world-point tensor captured for scoring matched upstream's returned tensor bit for bit across 1,015,280 points. The original caller's camera matrices remained intact. The actual model output differed from the separately computed FP32 calibration-head result by at most 0.00022 pixel in focal length. This check used zero optimizer updates. [Actual forward-pass record](cpu_initial_readout.json)

### Replay

Use the pinned TCO checkout, checkpoint, registered 7-Scenes development data and dataset metadata cache from the earlier full-model experiment. The source-archive hashes are in the offset-25 manifest. For the CPU comparison, first run the offset-25 camera collector at least through the Heads base batch. Run the following from the repository root in the pinned Linux environment. [Parent data and checkpoint](../../docs/REPLAY_FOCAL_ZOOM.md) · [Full-model environment](../canonical_sensor_factorial/REPLAY.md)

```sh
python scripts/prepare_dense_focal_pilot.py --assets /data
python scripts/check_dense_focal_inputs.py --assets /data
python scripts/check_dense_metric_identity.py
python scripts/check_dense_focal_cpu.py --checkpoint /data/checkpoints/VGGT-1B --output /tmp/focal-cpu-check.json
```

After a qualifying offset-25 replication, freeze the execution manifest. A GPU job consumes only the `inference` directory and the model checkpoint. Transfer its terminal evidence back for CPU evaluation on the retained references. The local smoke check spent $0 on RunPod.

```sh
python scripts/dense_focal_support.py --write
python scripts/run_dense_focal_pilot.py freeze
python scripts/run_dense_focal_pilot.py dry-run --inputs /inputs --checkpoint /data/checkpoints/VGGT-1B
python scripts/run_dense_focal_pilot.py run --inputs /inputs --checkpoint /data/checkpoints/VGGT-1B
python scripts/score_dense_focal_pilot.py seal
python scripts/score_dense_focal_pilot.py evaluate
python scripts/score_dense_focal_pilot.py publish
python scripts/score_dense_focal_pilot.py verify
```

### Same-target repeatability check

Each adaptive run has an independently executed control with the same images, supplied cameras, seed and final choice of focal target. The 24 pairs provide a direct estimate of how far two TCO runs can diverge even when they use the same target. After scoring, verify that both runs actually passed identical focal values to their loss, then compare the initial and final point-map hashes, adapter tensors, random states and metric differences. Report the signed mean, median absolute and largest absolute difference in the normalized distance score on each reference. This secondary check helps interpret the size of any observed gain. The progression criterion is specified in the [registered dense experiment](../../docs/DENSE_FOCAL_PILOT.md).

```sh
python scripts/audit_dense_focal_pairs.py write
python scripts/audit_dense_focal_pairs.py verify
```

### Can choosing between these two focal targets improve 3D error?

After scoring, compute three diagnostic comparisons from the 48 independently executed unconditional controls. First, use the known fault or genuine-zoom label to select the physically correct focal target. Second, select the lower measured 3D error separately for each condition and seed. This gives an optimistic bound for those two recorded endpoints on those same measurements. Third, select the lower-error arm on seed 42 and measure it on seed 123, then reverse the seeds. The last comparison measures how well the choice repeats on the same images. Both selections that use measured 3D error require reference depth and serve only as diagnostics. Keep the registered progression rule unchanged. [Analysis code](../../scripts/dense_focal_headroom.py)
