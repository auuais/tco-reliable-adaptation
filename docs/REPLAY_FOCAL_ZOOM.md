# Replaying the fresh-image focal experiments

Use a Linux checkout of this project at `2c85160` to collect fresh results in the original directory layout. That revision contains both frozen study manifests and precedes their observed outcomes. Work from the repository root using Python 3.11 and the repository's pinned PyTorch 2.7.1+cu128, NumPy 2.1.2 and OpenCV 4.11.0 environment. All steps below run on CPU. [Container definition](../environment/Dockerfile) · [Original experimental protocol](FOCAL_ZOOM_GRADIENT.md)

### Public inputs

Place upstream TCO at `third_party/TCO`, detached at `65387333877723cf99086f7f29089379dd50c59a`. Download the original Heads, Chess and Fire scene archives and VGGT checkpoint with the command below. The readers accept Microsoft's original scene ZIPs directly. They also accept extracted sequence ZIPs named, for example, `chess-seq-03.zip`. The image reader verifies each complete five-image tensor against an earlier public hash, and the camera model verifies all pretrained weight bytes. [Input and license references](THIRD_PARTY.md)

```sh
python scripts/download_assets.py --root /data --scenes heads chess fire
```

The `--support` folder can be outside this repository. The following commands populate the exact paths read by the frozen matcher. The two download checksums, Python source fingerprints and package versions are recorded in the matching manifest. [Matching environment](../manifests/learned_support_v8.json)

```sh
mkdir -p /support/third_party /support/reports/learned_support_v8_weights/hub/checkpoints
git clone https://github.com/cvg/LightGlue.git /support/third_party/LightGlue
git -C /support/third_party/LightGlue checkout --detach 5a650c444718caffc2669dc0094cde908f42dd73
python -m pip install --no-deps --target /support/reports/learned_support_v8_python kornia==0.8.2 kornia_rs==0.1.10
curl -fL https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth -o /support/reports/learned_support_v8_weights/hub/checkpoints/depth-save.pth
curl -fL https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth -o /support/reports/learned_support_v8_weights/hub/checkpoints/disk_lightglue_v0-1_arxiv.pth
```

### Collect and verify the primary experiment

Run the first two commands in separate processes or sequentially. The two workers write separate outputs, verify completed inputs on restart, and report elapsed time per five-view batch. Store the entire `reports/focal_zoom_gradient` directory. The remaining commands compute the original upstream TCO loss gradients and independently check their focal components from the released loss equations.

```sh
python scripts/focal_zoom_observations.py camera --archives /data/archives --checkpoint /data/checkpoints/VGGT-1B/model.safetensors --threads 5
python scripts/focal_zoom_observations.py tracks --archives /data/archives --support /support --threads 2
python scripts/focal_zoom_gradient.py run --archives /data/archives
python scripts/focal_zoom_gradient.py verify
python scripts/verify_focal_zoom_derivatives.py --write
python scripts/verify_focal_zoom_derivatives.py
```

### Run the classical visual calibration benchmark

The iterative two-view focal solver is the PoseLib implementation recommended by Kocur, Kyselica and Kúkelová. This project adds the square-pixel coordinate conversion, aggregation over multiple neighbors and the comparison at the actual VGGT camera outputs. The solver receives the same input images and supplied intrinsics as the primary method. Its input excludes the supplied poses. [Authors' code](https://github.com/kocurvik/robust_self_calibration) · [Benchmark configuration](../manifests/visual_focal_baseline.json)

```sh
python -m pip download --no-deps --only-binary=:all: --dest /tmp/focal-wheels poselib==2.0.5
echo '5035f1053cb4ae70aec4b57b56552e17eef9e96dc9ac3940ad83f0875e12f33c  /tmp/focal-wheels/poselib-2.0.5-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl' | sha256sum -c -
python -m pip install --no-deps --no-index --find-links /tmp/focal-wheels poselib==2.0.5
python scripts/visual_focal_baseline.py pack
python scripts/visual_focal_baseline.py run --archives /data/archives
python scripts/verify_visual_focal_baseline.py --rerun-solver
```

The checksum above identifies the Linux x86-64 CPython 3.11 wheel used here. The public evidence contains derived correspondences, model camera encodings, gradients and checksums. Images and model weights come from their original owners. Timing depends on the host and thread count.
