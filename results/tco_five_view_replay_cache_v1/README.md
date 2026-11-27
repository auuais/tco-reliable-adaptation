# Five-view TCO on the RTX 3080

Last-refinement camera gradients saved **1.14 GiB** of peak allocated memory against native execution at both full-raster sizes in the paired two-update screen. Last-refinement, whole-camera checkpointing, and both combined retention policies each completed all nine clean 40-update development runs. Native TCO and standalone transformer retention completed three of nine each. The six missing native completions closed the registered selection gate. The **1% reconstruction quality result remains unresolved**. [Registered design](../../docs/TCO_FIVE_VIEW_REPLAY_CACHE_HANDOFF.md) · [CPU-replayed decision](decision.json)

## Real-gradient replay

Three production replays with identical forward values, optimizer state, RNG and real incoming derivatives still differed by **0.632–0.645%** and **0.685–0.737%** in pooled pre-clipping LoRA gradients at updates one and two. The independently evaluated renderer derivatives differed by about 1e−8 to 6e−8 in relative L2. Three deterministic fixed-derivative replays gave exact gradients and optimizer updates. On the deterministic anchor, using the first independently regenerated derivative changed the pooled gradient by **0.681%** and **0.655%**. The two backends required separate forward anchors. These are matched two-update findings from the reduced-raster Heads fixture. The specific operation behind the variation remains unidentified. [Gradient analysis and controls](GRADIENT_REPLAY.md)

Both camera policies reproduced the native signed camera-token gradients exactly at both matched production states. Their complete production comparisons crossed the envelope from the three native calibration repeats on some tensors. An additional unmodified native repeat also crossed that envelope. The registered supplemental test at the already verified deterministic anchor gave exact agreement for both camera policies and all three retention policies across the 384 LoRA tensors and optimizer state. This supports the arithmetic of the interventions under that diagnostic backend. The 40-step timings below use the production backend. [Production camera qualification](qualification.json) · [Deterministic camera qualification](deterministic_screen_qualification.json) · [Production retention qualification](retention_qualification.json) · [Deterministic retention qualification](deterministic_development_qualification.json)

## Forty updates with all five views

The tables report the median over completed attempts for each arm and scene. Each arm was scheduled for three seeds. Scene time includes placement, adaptation and final FP32 reconstruction. Scoring and container startup sit outside that timer. Every arm preserves one five-image depth batch, all 24 frame/global stage pairs, all LoRA factors and the production rendering loss. Heads and Chess use 392 × 518 rasters. Courtyard uses 350 × 518. [Per-seed resources and paired contrasts](development_resources.json)

| Method | Heads completed | Heads time / allocated | Chess completed | Chess time / allocated | Courtyard completed | Courtyard time / allocated |
|---|---:|---:|---:|---:|---:|---:|
| Native | 0/3 | — | 0/3 | — | 3/3 | 68.44 s / 7.501 GiB |
| Whole-camera checkpoint | 3/3 | 78.14 s / 6.302 GiB | 3/3 | 78.86 s / 6.302 GiB | 3/3 | 69.43 s / 5.981 GiB |
| Last-refinement gradients | 3/3 | 77.53 s / 6.682 GiB | 3/3 | 75.81 s / 6.682 GiB | 3/3 | 68.71 s / 6.361 GiB |
| Transformer retention alone | 0/3 | — | 0/3 | — | 3/3 | 140.47 s / 7.969 GiB |
| Last refinement with ranked retention | 3/3 | 76.07 s / 7.199 GiB | 3/3 | 76.20 s / 7.199 GiB | 3/3 | 67.35 s / 6.829 GiB |
| Last refinement with evenly spaced retention | 3/3 | 75.99 s / 7.199 GiB | 3/3 | 75.95 s / 7.199 GiB | 3/3 | 68.08 s / 6.829 GiB |

The **last-refinement** method lowered peak allocated memory by **15.2%** versus native TCO on Courtyard and increased median scene time by **0.4%** there. Whole-camera checkpointing used about **0.38 GiB less** than last-refinement across all three scenes. The combined method lowered peak allocated memory by **9.0%** and median time by **1.6%** against native on Courtyard. Against the equal-memory retention control, it was 0.07 s slower on Heads, 0.25 s slower on Chess, and 0.73 s faster on Courtyard. This small, inconsistent difference gives no evidence of a generally better placement. Retention alone doubled median time and increased peak allocation on Courtyard. The timeouts on the larger rasters rule out a general speed comparison with native. [Resources by case](development_resources.json) · [Terminal attempts](development_evidence.json)

The instrumented native and zero-save Heads cases both reached the 1,200 s watchdog. The two Courtyard diagnostic profiles completed. The study used the registered whole-block fallback because it lacked successful operator traces from both rasters. The fixed screen evaluated eight named transformer sites on each raster and ranked their two-update time saved per added MiB. The combination screen selected global stages **0 and 7**. Frame stages **6 and 18** served as the evenly spaced control and matched allocated peak exactly on both screens and all three full-length development windows. The 40-update results above are from separate fresh processes. [Single-site evidence](screen_evidence.json) · [Combination evidence](combination_evidence.json) · [Locked development design](development_plan.json)

## Geometry and completion

All **60** full-length development slots were attempted. Of these, **46** completed and **14** reached the watchdog. Both Native attempts at each of Chess seeds 42 and 43 reached the watchdog with the same inputs, optimizer settings and seed. At seed 44, the native repeat completed in 465.32 seconds of timed scene work and the primary native case timed out. These identical-setting repeats were already in the schedule. Further duplicate attempts were omitted after repeated stalls. The native reference was incomplete across the required development windows, so the lock contains no promoted policy. The 90 proposed held-out reconstructions and four linked rescores remained gated. [Resource selection](locked_plan.json) · [Terminal registration](../../manifests/tco_five_view_replay_cache_locked_v1.json)

All three Courtyard native repeat pairs produced different reconstructions. Same-seed accuracy-error differences ranged from **174 to 462 mm** and completeness-error differences ranged from **49 to 324 mm** after native Umeyama alignment and rigid ICP. Reference input hashes match within each pair. A separate CPU-only container rescored all seven saved native reconstructions, including the Chess repeat, using the original reference geometry. The largest difference from the archived scores was **1.66 × 10⁻¹¹ mm**, within the 0.01 mm tolerance. [Native rescore receipt](native_cpu_rescore.json)

All GPU work used the single local 10 GiB RTX 3080, with a configured PyTorch process fraction of 0.875. One completed Courtyard standalone-retention case reported a peak **reserved** counter 2.44 MiB above the nominal fraction limit while its peak **allocated** counter was 7.969 GiB. The first host replay treated the reserved counter as a hard cap. The release reader now verifies the recorded allocator configuration and allocated bound and retains the observed reserved value in the evidence. The original registration, runner bytes, and quality margin are preserved. [Allocator replay correction](allocator_reservation_replay.json)

The launch ledger documents **166 attempts**, **144 completions**, **16 timeouts**, and **six forward-identity rejections** when testing the shared production anchor under deterministic mode. The GPU and CPU cases in the ledger took **8 hours 49 minutes 41 seconds** from first launch to final receipt. Rental compute cost was **USD 0**. [Full replay](decision.json) · [Second-disk archive](archive_receipt.json)

## CPU verification

```powershell
python -B -S scripts/replay_five_view_release.py replay
python -B -S scripts/replay_five_view_release.py allocator-receipt
python -B -S scripts/replay_five_view_release.py resources --source evidence
python -B -S scripts/test_tco_five_view_replay_cache.py
python -B scripts/test_tco_five_view_gradients.py
python -B scripts/replay_five_view_release.py archive verify --snapshot study-v1 --check-source
```
