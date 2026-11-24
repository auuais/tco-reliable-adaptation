# Full 40-step TCO with selective depth-head recomputation

**Study in progress.** Three ten-view Heads resource pilots completed 40 updates using the full LoRA inventory on a 10 GiB RTX 3080. The pilots used the same seed in three fresh processes. The locked study tests other seeds, transfer windows and five-view reconstruction quality.

## Native memory limit

The instrumented five-view reference finished all 40 steps in 77.29 seconds. Peak allocated memory was 7.82 GiB and peak reserved memory was 8.73 GiB. The separate uninstrumented reference attempt reached the 20-minute wall-clock limit. Its terminal receipt identifies a timeout.

The instrumented ten-view run completed the initial prediction and the first joint transformer and camera-head calls. At the depth-head entrance, 7.03 GiB was allocated. The first eight-frame depth chunk then failed at its high-resolution fusion convolution during update zero. CUDA requested 260 MiB with 8.69 GiB already allocated under the 8.75 GiB cap. The uninstrumented ten-view attempt failed at the same convolution. [Native profiles](profile.json) · [Attempt receipts](initial_evidence.json)

## Two-update factorial

All eight tests used ten 392 × 518 views of Heads/seq-01 and seed 42. Chunk size controls the number of views per depth-head call. Every transformer call and rendering loss still receives all ten views. Recomputing a chunk during backward avoids holding its internal activations from the original forward pass.

| Maximum frames per chunk | Ordinary execution | Selective recomputation |
|---:|---|---|
| 8 | Forward OOM | Backward OOM |
| 4 | Forward OOM | Backward OOM |
| 2 | Forward OOM | Completed at 8.41 GiB peak allocated and 6.90 s/update |
| 1 | Forward OOM | Completed at 8.29 GiB peak allocated and 6.54 s/update |

One frame per chunk ranked first by the registered two-update time. Across three unscored full-length runs the median was **156.52 seconds** with **8.29 GiB peak allocated** and **8.75 GiB peak reserved**. The full-length same-chunk control hit a CUDA OOM. The locked design uses one frame per chunk with recomputation and retains every chunk-only setting as a ten-view control. [Screen evidence](screen.json) · [Locked registration](../../manifests/tco_full40_recompute_locked_v1.json)

## Numerical qualification

The pinned PyTorch 2.7.1 CPU fixture reproduced depth, confidence and all four tap gradients exactly when it toggled recomputation at each fixed chunk size. With one image per chunk, the reduced-resolution GPU fixture also matched exactly across two fixed-adjoint updates. This covered 1,536 gradient and updated parameter tensors, both render orders, and the final predictions. Swapping the evaluation surface and mask left the prediction hash identical.

Changing from the native chunk size to one image per chunk introduced numerical differences before recomputation. At full-size zero-step inference the largest relative L2 difference across the exported predictions was 0.034%. The fixed-adjoint test showed that gradients can be more sensitive than predictions to this chunk-size change. The registered five-view tests measure how much these differences matter after 40 real updates. [Qualification receipts](qualification.json)

## Local reproduction

The local launcher uses the pinned offline image, the verified VGGT checkpoint, read-only input data and one fresh process per case. Measurements use the same expandable allocator and 87.5% process cap in all arms. RunPod compute cost is **USD 0**. [Handoff](../../docs/TCO_FULL40_RECOMPUTE_HANDOFF.md) · [Registration](../../manifests/tco_full40_recompute_v1.json) · [Preflight](preflight.json)

```powershell
python scripts/tco_full40_recompute_protocol.py verify
python scripts/run_tco_full40_recompute_local.py initial --cohort fixture --cohort reference --cohort profile --cohort screen
python scripts/run_tco_full40_recompute_local.py initial --job promote_c1_on_r1
python scripts/check_tco_full40_recompute.py profile
python scripts/check_tco_full40_recompute.py screen
python scripts/check_tco_full40_recompute.py qualification --candidate c1_on
python scripts/run_tco_full40_recompute_local.py initial --cohort repeat
python scripts/tco_full40_recompute_protocol.py verify --registration locked
python scripts/run_tco_full40_recompute_local.py locked --cohort ten --cohort development --cohort confirmation
python scripts/test_tco_full40_recompute_replay.py
```
