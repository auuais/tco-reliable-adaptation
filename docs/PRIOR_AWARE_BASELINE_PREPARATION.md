# Official baseline preparation

The ongoing MASt3R screen needs appropriate comparisons before a publication claim. Official source trees were fetched without changing the running GPU environment. Both model modules import in the existing tco-mast3r:pilot image. No baseline GPU inference or reconstruction-quality evaluation has run in this preparation stage.

## Pinned assets

| Asset | Revision |
|---|---|
| [Pow3R source](https://github.com/naver/pow3r) | 25d7bb68e01336585b5277d5d6aafadddf9e1b8d |
| Pow3R DUSt3R submodule | ad9d1a0421b19d7be72801f6e2e8bfcf79dc3c68 |
| Pow3R CroCo submodule | d7de0705845239092414480bd829228723bf20de |
| [Test3R source](https://github.com/nopQAQ/Test3R) | a2eb94bc716df27521f053d417fcf6afa9870ff5 |

Pow3R's official 512-resolution checkpoint is downloaded locally, with 2,223,286,338 bytes and SHA-256 b766fbf81873a579b8d633d3b1bb30580fa55d8ecf906759d868b6aec8cd624f. This is a locally computed digest of the official-vendor HTTPS download, not an author-supplied checksum. Its recorded definition strictly matches 555,733,504 parameters and 906 float32 weight tensors under a meta-device schema check. That check does not perform numerical inference. See results/prior_aware_baselines for the provenance and model definition. No checkpoint is redistributed in Git.

To prepare another checkout, clone the official repositories into third_party/pow3r and third_party/Test3R, check out the revisions above, and initialize Pow3R's recursive submodules. Test3R's CroCo code is a vendored tree in this revision, despite its retained .gitmodules entry. Then run scripts/prepare_prior_aware_baselines.py with an explicit local checkpoint destination. The script checks source revisions and preserves incomplete downloads for review. Run scripts/inspect_pow3r_checkpoint.py inside the existing image with PYTHONPATH pointing only to third_party/pow3r.

## Pow3R comparison contract

Use the exact frozen RGB tensors, native pixel-center transform, sparse depth samples, calibration and clean/noisy pose packets. The official helpers construct rays, normalize sparse depth and normalize relative translation. Use and document those conventions rather than feeding raw metric tensors directly into the conditioning fields. Explicitly state whether post-inference ray or pose fixing is enabled. The high-resolution demo enables extra processing and is not a drop-in evaluator for our five-view benchmark. Define fusion/alignment and metric-scale recovery before opening reference labels. Include an appropriate within-Pow3R modality control; differences between pretrained backbones alone do not establish the benefit of our adaptation rule.

## Test3R comparison contract

The released multi-view launcher loads a DUSt3R model with visual prompts. Its default script uses 32 prompts, one epoch, LR 0.00001 and gradient accumulation 2. It trains the prompts with cross-pair point-map disagreement. A separate function accepting reference targets exists in the source but is not the function called by the released adaptation loop. Keep target depth outside our adaptation wrapper and audit the exact called path.

The released prompt encoder assumes two encoded input images. Its prompt insertion and replacement schedule must be preserved for a faithful reproduction. A MASt3R-specific port is a separate implementation and must be labeled accordingly. Inference from the source: zero-valued added prompt tokens can change attention normalization, so measure a zero-update prompt control rather than assuming an identity initialization. This has not been measured here. Compare with a parameter- and compute-accounted decoder-adapter variant, not a differently tuned hidden baseline.

Official evaluation preparation and any future MASt3R port remain pending. Baseline preparation does not release rental funds or establish a new method. Preserve the upstream licenses and notices; use the authoritative LICENSE files linked in THIRD_PARTY.md.
