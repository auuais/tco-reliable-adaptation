# Gradient replay diagnosis

November 18, 2026. This is engineering evidence, not an accuracy result. No original model weight changed and no rental GPU was used.

The first registered bridge, `gradient01` at commit `8e33ba3`, passed pristine-versus-zero-LoRA pointmap equality but failed retained/replayed gradient agreement. Its relative gradient discrepancy was 0.005883 against a maximum of 0.0001. The forward losses were identical. The failure remains archived.

The no-update diagnostic `replay_precision01` at commit `47e162a` localized the discrepancy. Recomputed pointmaps and loss adjoints were exactly equal, and supplied adjoints were not mutated. However, repeating the same joint backward pass changed the BF16 LoRA gradient by relative L2 0.007711. The FP32 repeated-backward discrepancy was 0.00002939. Thus the original result did not isolate a replay-chain-rule error: backward execution itself varied at a comparable scale. Original and adapter hashes stayed unchanged.

The strict deterministic diagnostic `replay_deterministic01` at commit `d68b68b` set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, strict deterministic algorithms, deterministic cuDNN and disabled convolution benchmarking. Repeated joint backward, joint-versus-replay, individual pair VJPs and sequential accumulation were all exactly equal in both BF16 and FP32: relative L2 zero and cosine one. Original and adapter hashes remained unchanged. Wall time was 69.43 seconds, versus 61.97 seconds for the first diagnostic; these are one-run engineering observations, not a performance benchmark.

The pinned MASt3R DPT head uses bilinear interpolation. The pinned PyTorch 2.7.1 `functional.interpolate` dispatches CUDA bilinear input to `_upsample_linear_vec` when deterministic algorithms are enabled, including during `no_grad` inference. The raw CUDA bilinear backward uses atomic additions, while the built-in deterministic path avoids the observed variation. Strict execution fixes the measured discrepancy, but this experiment alone does not isolate every possible nondeterministic kernel. [PyTorch 2.7.1 functional implementation](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/nn/functional.py), [CUDA bilinear kernel](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu), [deterministic algorithms](https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html).

The deterministic path also changes floating-point forward arithmetic slightly. The BF16 diagnostic loss moved from 25.951331333991202 to 25.950049611326445. Therefore every new teacher, neural arm, frozen comparator and endpoint must use the same runtime. Historical predictions produced under the old arithmetic cannot substitute for the new matched controls. The method, weights, data splits, loss constants and acceptance thresholds are unchanged.

`scripts/profiled_runtime.py` implements and records the common runtime policy. It refuses an incompatible cuBLAS setting after CUDA initialization, enables strict deterministic algorithms, disables cuDNN benchmarking and sets both matmul/cuDNN TF32 consistently. Real finite-difference checks temporarily disable **both** TF32 flags, because DPT contains convolutional heads, and restore both before the BF16 macro-step. The original diagnostic's FP32 comparisons disabled matmul TF32 only; they remain diagnostic evidence and do not stand in for the corrected finite-difference acceptance.

Preserve engineering v1. Register the revised source and runtime as engineering v2, then rerun the unchanged acceptance suite: zero-adapter fidelity, retained/replayed gradients, real adapter finite differences, a full 20-pair macro-step and the fresh native/final endpoint bridge. The diagnostic alone does not release the 36-state accuracy study. No held-source or dense accuracy scores have been read for this goal.

Evidence:

- `results/lora_profiled_v1/engineering/gradient01/`
- `results/lora_profiled_v1/engineering/replay_precision01/`
- `results/lora_profiled_v1/engineering/replay_deterministic01/`

**Formal acceptance and Windows artifact boundary**

Engineering v2 passed `gradient02`: exact retained/replay agreement, all three finite-difference steps, unchanged original state and a full 20-pair optimizer step. The macro-step used 4825.71 MiB peak allocated and took 23.85 seconds. `endpoint01` then passed exact pristine/zero-adapter native and final-refined geometry, matching and confidence equality, with unchanged original and adapter hashes.

The first scientific-v1 P0 run also completed successfully, with no adapter and no accuracy scoring. Its host queue verifier then stopped because Docker's five canonical-cache filenames contain ASCII colons, exposed on this Windows bind mount as U+F03A. Counts were equal (47 files); names differed. This was not a model or numerical failure. The shared artifact helper now applies only that Windows logical-name mapping, rejects collisions/path escape, and verifies the unchanged bytes. The preserved P0 audit passes all 47 raw and six public files, including the completion report.

To keep the study under one execution registration, that unscored P0 is archived and excluded from the new scientific-v2 inventory. Its raw/public files were copied to E: and hash-verified before moving the C: copies into the archive. The public [archive record](../results/lora_profiled_v1/archived_cases/scientific_v1_heads_earlier_P0_screen01.json) gives original and archived paths plus every checksum. All 36 states will run under scientific v2; the mathematical design and model source remain unchanged. This is an execution repair, not the one allowed scientific redesign.
