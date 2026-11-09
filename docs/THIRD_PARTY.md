# Third-party sources and attribution

This repository contains independent experiment orchestration and aggregate evidence. Obtain upstream source, weights and datasets from their owners and follow their terms. Public availability does not imply permission to relicense those assets.

- [TCO](https://github.com/cvlab-stonybrook/TCO), commit `65387333877723cf99086f7f29089379dd50c59a`: method, model wrapper, loss and evaluation functions. [Paper](https://arxiv.org/abs/2604.03878).
- [VGGT](https://github.com/facebookresearch/vggt): pretrained reconstruction backbone; checkpoint revision and hash are in REPRODUCTION.md.
- [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/): public research dataset. Download separately; images, depth, poses and checkpoint weights are excluded from Git.
- [Pi3](https://github.com/yyfz/Pi3), commit `97ce97b8e067e7bff5b5c8e507776e4e95189fac`: reference frame-sampling protocol. Our manifests preserve the selected frame IDs.
- [SimpleRecon](https://github.com/nianticlabs/simplerecon), *SimpleRecon: 3D Reconstruction Without 3D Convolutions*, Sayed et al., 2022: 7-Scenes depth registration used by the referenced preprocessing pipeline. Obtain the [preprocessing source](https://github.com/nianticlabs/simplerecon/tree/main/data_scripts) and [license](https://github.com/nianticlabs/simplerecon/blob/main/LICENSE) directly. Its noncommercial license is separate; the local modified preprocessing script is not redistributed here.

Registration uses the published depth-to-RGB rigid calibration, depth focal length 585, RGB focal length 525, and nearest-pixel z-buffering. Keep original frame IDs, strip blank TestSplit lines only, and save registered depth as `.depth.proj.png`. The local vectorized implementation was checked against the reference scalar round/min loop on the first frame of every prepared sequence. Preserve that validation when preparing new data. Nominal calibration is part of the benchmark protocol, not an assertion of physically exact sensor calibration.

Runtime components include PyTorch, Open3D, gsplat, NumPy and Hugging Face libraries. They retain their own licenses. The published compatibility patch is a minimal bfloat16 export repair; the full upstream tree is fetched separately and excluded from version control.
