# Why ordinary TCO slowed down on the local 3080

The current Heads seed-42 Full10 control took 130.62 s. The archived ordinary `four` implementation took 27.61 s on the same RGB, camera priors, reference, five frames and seed. The current run used 8,192 additional peak live bytes and 2 MiB additional peak reserved GPU memory. Both reported 9,440,329,728 free GPU bytes before the scene and zero free bytes after the scene. The current process reported seven allocator retries. The earlier study did not record that counter. Its first loss equals the current first loss, 0.64774108. Its final loss was 0.18661089 versus 0.18499647 in the current run. This establishes persistent numerical agreement at initialization and a difference by the end of optimization. [Audited receipts and source hashes](../results/tco_two_track_v1/a_native_harness_provenance.json)

## What each harness retains

| During or after adaptation | Archived ordinary worker | Current ordinary worker | Timing interpretation |
|---|---|---|---|
| Rendering objective and trainable scope | Named seeded rank-4 LoRA in all 24 frame/global stages, four retained feature taps, production renderer, full Adam updates | Same pinned core paths, ordinary arm uses the same `four` stage implementation | These are comparisons to the project's optimized ordinary control. Stock upstream timing needs its own matched measurement. |
| Scalar objective | First and last detached loss | All detached losses | Old post-loop payload is two scalars. The longer current lifetime can affect allocator reuse. |
| CUDA view permutation | Native draw used by the renderer, no retained per-update array in clean production | The original draw is returned to the renderer and a detached view remains in a list until after timing | At 10 updates, ten five-element int64 orders and ten FP32 scalar losses amount to 440 bytes of logical values, 432 bytes above the archived first/last losses. At 40 updates the difference is 1,752 bytes. Allocator blocks and segments are larger than these values. |
| Adam | Non-owning post-step counter; upstream local Adam remains live for all updates and the native final forward | Audit owns Adam and factor references past model return through the external delivered readout | The checkpoint header gives 3,145,728 full-scope FP32 adapter elements. Two Adam moment arrays total 24 MiB. Extending them beyond the model return could affect the external readout. Both policies already hold them during the earlier backwards. |
| IO and bookkeeping | Per-step prints redirected into a CPU string buffer and printed after synchronization. One post-step counter; frozen-base CPU version checks. | Per-step prints stream to Docker; current ordinary pre-hook checks grad presence across 384 tensors. Loss and order exports occur after the timer. | The native `.item()` calls used to format training logs happen in both. The time bridge keeps stdout, host thread settings and the delivered readout common. |
| Timer and cap | Starts after variant installation; CPU checkpoint load reported separately. Allocator uses 87.5% of the reported physical capacity. | Starts before installing the audit, includes placement, optimization and the FP32 readout; separate cold load. Exact 8.75 GiB cap. | On this adapter the older nominal cap is 458,752 bytes smaller. Both new diagnostic runs use the same current absolute cap and time boundary. |

Detached tensors share source storage and release autograd history. A list of detached scalars does not hold the entire backward graph. Pinned small blocks can still change segment placement near an allocation limit. Peak counters do not show when the allocator waited, what allocation triggered a retry, or Windows driver residency. These hypotheses have different timing implications. [PyTorch detach semantics](https://docs.pytorch.org/docs/2.7/generated/torch.Tensor.detach.html) · [Allocator counters](https://docs.pytorch.org/docs/2.7/generated/torch.cuda.memory_stats.html)

The first Fire seed-44 Full40 attempt reached the pre-backward loss print for zero-based update 22 and then timed out. Its watchdog snapshots included the autograd engine and checkpoint recomputation. This localizes part of the delay inside optimization. Keeping Adam through a later external readout cannot, by itself, explain that earlier delay. The archived Fire44 Full40 run completed in 74.92 s. [Diagnostic evidence](../results/tco_two_track_v1/a_native_harness_provenance.json)

## What the post-core comparison can decide

Run one fresh process with archived observer lifetimes on Heads/42/Full10, followed by one fresh process invoking the unchanged current worker under the same host lock. Both receive the same RGB and calibrated cameras. No scene surface is mounted. They use the same current image, allocator, native sampler, seed, synchronized timer, stdout treatment and delivered FP32 readout. The lean process calls the archived `install_variant(...,"four",...,None)` and `observe_losses(...,None,False)` and lets the local Adam go out of scope when the model returns. Its private per-process allocator counters are sampled outside the timer. It retains no per-update CUDA telemetry. [Registered cases and branch rule](../manifests/tco_two_track_v1_native_bridge.json)

After both outputs are sealed, verify actual optimizer counts and the first scalar objective. The lean worker independently advances a second CUDA generator by the expected native draws after timing. It compares final RNG states and the inferred per-step orders with the current worker's directly observed orders. The CPU replay reports endpoint losses, field shapes, dtypes, finite support, native-coordinate RMS and maximum absolute differences. A difference in terminal geometry remains descriptive because the production renderer has repeat spread. This check opens no evaluation surface. [Bridge runner](../scripts/run_tco_native_lifetime_bridge.py) · [Independent comparison](../scripts/tco_native_harness_bridge.py)

| Heads pair | Consequence |
|---|---|
| Current reproduces the slowdown and lean is substantially faster, with matching arithmetic and more retries in current | The observer package is a credible contributor on this case. A one-pair, fixed-order contrast cannot isolate which member caused it. Describe speed against the measured instrumented control and measure an independently repeated low-observer comparator before generalizing a TCO speedup. |
| Both finish quickly | The large historical discrepancy did not recur in this pair. Preserve the original slow observation in the primary result; host and allocator timing variability remain unresolved. |
| Both are slow, lean is slower, or arithmetic/RNG checks fail | Evidence does not isolate an observer-induced slowdown. A failed arithmetic check also prevents interpreting the timing as an equivalent-model comparison. |

One lean Fire44 Full40 run is permitted only if both registered Fire44 primary attempts ended in timeout and the contemporary Heads pair meets all three operational conditions. Current must take at least twice the lean time, be at least 30 s slower, and report more retries. Those conditions govern whether one more diagnostic is worthwhile. A Fire completion establishes feasibility under the lean harness on that replay, with primary watchdog times treated as censored. It never substitutes for a primary result. The evidence also retains the same-schedule Refresh1/Refresh1Uncached comparison, where early-stage updates, view sampling and Adam counts can be checked directly.

```powershell
python -B -S scripts/tco_native_harness_bridge.py audit
python -B -S scripts/tco_native_harness_bridge.py register
python -B scripts/test_tco_native_harness_bridge.py
# Commit the registration and both workers before the diagnostic.
python -B scripts/tco_native_harness_bridge.py heads
python -B scripts/tco_native_harness_bridge.py fire
python -B scripts/tco_native_harness_bridge.py replay-heads
```

Each process has the existing 1,200 s deadline. Each diagnostic case has one attempt. The local cost is USD 0 in RunPod charges.
