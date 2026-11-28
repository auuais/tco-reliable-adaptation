# Five-view TCO on the RTX 3080

This study asks two questions. Can a cached ten-update adaptation recover the geometry of ordinary ten-update TCO and finish sooner? Can combining independent forty-update reconstructions improve the reconstructed surface? Every geometry score compares a reconstruction with the measured reference. Percent changes compare those ground-truth errors between methods. Smaller errors are better.

## Accuracy at forty updates

Combining three saved reconstructions increased accuracy error by **2.55%** and reduced completeness error by **0.58%**, relative to the mean of the three ordinary Full40 runs within each development room. Accuracy improved in Fire and worsened in Heads and Chess. The three solutions supplied visible depths at 94–96% of anchor pixels. Taking their median did not produce a more accurate surface across these rooms. The registered branch ended after development, before new-scene or longer-run comparisons. [Room-level results, coverage and controls](b_development_summary.md) · [Independent decision](b_development_independent.json)

## Efficiency at ten updates

The fastest development candidate, **Narrow**, allows only stage pair 11 and the final twelve pairs to adapt in its first update. On the remaining nine updates it changes just the last twelve and reuses the frozen features. All nine registered reconstructions finished. Against ten updates of ordinary TCO, it delivered **2.39× speedup** and **21.77% lower peak live GPU memory**, with equal-room changes of **−1.03% accuracy error** and **−1.59% completeness error**. The percentage for each metric averages three within-room ratios. Heads improved substantially, while Chess worsened by 6.52% and 11.18%. The respective upper 95% bounds of +4.74% and +8.09% cross the +1% and +3% quality margins. [Matched room measurements](a_hybrid_selection_development.md) · [Registered gate and bootstrap](a_hybrid_stress_entry.json)

Against the existing Cache10 baseline, Narrow took about 2% less time, used 2.40% more peak memory and changed the two errors by −0.52% and +0.14%. Its engineering check reproduced all 737 signed comparisons exactly. The registered next comparison uses Office and Courtyard, three seeds each, with ordinary Full10 and Cache10 as controls. [Engineering check](a_narrow_engineering.json) · [Stress roster](../../manifests/tco_two_track_v1_stress_hybrid.json)

Refresh1 lets all transformer stages adapt on the first update, then freezes the first twelve stage pairs and reuses their outputs for the remaining nine. The uncached twin follows the same parameter and optimizer schedule and recomputes those outputs. Full10 updates all stages ten times. Cache10 updates only the last twelve stage pairs ten times. Full40 supplies the full-length quality and resource reference.

Over all **nine room/seed pairs**, Refresh1 used 53.94% less scene time than Full10. Its mean within-room accuracy error ratio improved by 0.82%, and its completeness error ratio improved by 2.72%. It passed all three development point thresholds. The upper bounds for the error ratios were 1.0442 and 1.0542, so these three rooms cannot establish the requested 1% and 3% tolerances on new scenes. Relative to Cache10, the early update improved the measured errors by 0.24% and 0.77% while adding 9.55% time and 29.54% peak live allocation. The matched uncached twin isolated a 26.17% time reduction from feature reuse. All 36 ten-update runs and six of nine Full40 runs completed. The registered alternative restricts the early update to one additional stage pair to test whether the peak cost can be recovered. [Full development results](a_development_summary.md)

The same RGB windows and ground-truth surfaces also have archived controls. Refresh1's accuracy and completeness errors were 6.72% and 13.09% lower than image-only VGGT, respectively. Against the archived Full40 predictions, the reductions were 3.36% and 6.37%. These are retrospective geometry comparisons on development rooms. The current Full40 completion count remains six of nine. [Matched historical controls](a_comparator_development_with_prior.json) · [Both VGGT camera readouts](a_retrospective_zero_readouts.json) · [Independent scoring coverage](a_development_rescore_coverage.json)

At full resolution the cached and uncached engineering twins matched on all 1,177 signed-gradient, effective-weight and output comparisons. A deliberately stale cache kept its old values. Invalidating it changed 5,327,156 entries and recovered the independently recomputed values exactly. The first deterministic diagnostic exceeded the 8.75 GiB cap. The successful diagnostic kept saved autograd tensors on CPU. The two production controls used the ordinary GPU path. These tests establish implementation fidelity. Speed and geometry require the clean runs and surface evaluation. [Engineering evidence](a_engineering.json)

### First matched ten-update window

All four Heads seed-42 outputs received independent CPU rescores. This is one prespecified development window. The selection decision uses the full development roster.

| Ten-update method | Accuracy error (mm) ↓ | Completeness error (mm) ↓ | Scene time (s) | Peak live (GiB) | Allocator retries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full10 | 13.021 | 13.917 | 130.62 | 7.822 | 7 |
| Cache10 | 11.718 | 11.755 | 19.54 | 5.976 | 0 |
| Refresh1 | 11.990 | 11.939 | 21.20 | 7.741 | 0 |
| Refresh1 uncached | 11.732 | 11.736 | 28.56 | 7.741 | 0 |

Refresh1 reduced both errors relative to Full10 on this window. Cache10 produced the lowest errors and shortest time. Holding the update schedule fixed, caching reduced time by 25.8%, while the production runs differed in accuracy error by 2.2%. Full10's seven allocator retries warrant a separate matched timing check. The same seed and window in the historical Full10 study took 27.61 s. [Paired scores](a_pilot_rows.json) · [Allocator measurements](a_allocator_pilot.json) · [Native timing bridge](../../docs/TCO_TWO_TRACK_NATIVE_TIMING_BRIDGE.md)

A previous local study also saw native Chess seed 42 take 1,032.22 s and 88.72 s in independent Full40 runs at the same 7.822 GiB peak. The slowdown therefore predates the new timing harness. The bridge tests whether bookkeeping contributes additional variability on this workstation. [Prior native stall diagnosis](../tco_native_stalls_quality_capacity_v1/README.md)

## Local reproduction

Run commands from the repository root in PowerShell. The registered Windows launcher uses the pinned offline Docker image on the physical 3080. Supply the separate five-view model and reference packets at the locations in the handoff. The model container sees only RGB and the permitted camera priors. The CPU scorer checks the model-output hash and uses the fixed ground-truth support.

```powershell
python -B scripts/tco_two_track_protocol.py verify
python -B scripts/lock_tco_two_track_jobs.py
python -B scripts/run_tco_two_track_fusion_local.py prepare
python -B scripts/run_tco_two_track_fusion_local.py infer
python -B scripts/run_tco_two_track_fusion_local.py leak-check
python -B scripts/run_tco_two_track_fusion_local.py score
python -B scripts/run_tco_two_track_fusion_local.py independent
python -B scripts/release_tco_two_track_fusion.py
python -B -S scripts/replay_tco_two_track.py b-development
python -B scripts/run_tco_two_track_a_local.py --jobs manifests/tco_two_track_v1_engineering_v2.json
python -B scripts/qualify_tco_two_track_a.py --jobs manifests/tco_two_track_v1_engineering_v2.json
python -B scripts/run_tco_two_track_a_local.py --jobs manifests/tco_two_track_v1_pilot.json
python -B scripts/qualify_tco_two_track_pilot.py
python -B scripts/run_tco_two_track_a_local.py --jobs manifests/tco_two_track_v1_development.json
python -B scripts/score_tco_two_track_a.py development
python -B -S scripts/replay_tco_two_track.py a-development
```

Clean model runs and CPU scoring occupy separate time windows. Each scored endpoint retains its paired inputs, elapsed scene time, peak live and reserved allocation, update counts and scoring provenance. Reported rental compute for both tracks is USD 0. The handoff specifies conditional experiments and confidence bounds. [Registered design](../../manifests/tco_two_track_v1.json) · [Execution handoff](../../docs/TCO_TWO_TRACK_SOL_HANDOFF.md)

### Continuing after a native timeout

The clean launcher allows one unchanged retry for an ambiguous watchdog timeout. If it stops after the second timeout, the following command preserves completed and exhausted slots and continues in the original order. Run it only after the prior launcher exits.

```powershell
python -B scripts/resume_tco_two_track_clean.py --jobs manifests/tco_two_track_v1_development.json
```

After all 45 slots have terminal receipts, run the development scorer and replay above, then run the commands below. The replay includes failed slots in coverage. The allocator audit retains all completed times and separately describes paired zero-retry cases. The native timing bridge checks whether observer bookkeeping affected ordinary TCO's timing.

```powershell
python -B -S scripts/plan_tco_two_track_a_followup.py alternative
python -B scripts/audit_tco_two_track_development.py development --save
python -B -S scripts/audit_tco_two_track_timing.py --stages development
python -B scripts/run_tco_native_bridge_local.py heads
python -B scripts/run_tco_native_bridge_local.py fire
```

The alternative gate chooses at most one additional refresh schedule. If it chooses Narrow, generate and commit the engineering ledger, run the cached and uncached diagnostic twins, and verify their signed derivatives before launching the nine clean cases. The next decision is `python -B -S scripts/plan_tco_two_track_a_followup.py stress-hybrid`. This selects the candidate, if any, for Office and Courtyard. Only the subsequent `distillation-gate` decision opens the six-visit Training intake. Fresh Validation runs require a frozen nominee. [Conditional decisions](../../scripts/plan_tco_two_track_a_followup.py) · [Teacher objective](../../manifests/tco_two_track_v1_distillation_policy.json)
