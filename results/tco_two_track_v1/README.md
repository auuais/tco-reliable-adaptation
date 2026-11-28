# Five-view TCO on the RTX 3080

This study asks two questions. Can a cached ten-update adaptation recover the geometry of ordinary ten-update TCO and finish sooner? Can combining independent forty-update reconstructions improve the reconstructed surface? Every geometry score compares a reconstruction with the measured reference. Percent changes compare those ground-truth errors between methods. Smaller errors are better.

## Accuracy at forty updates

Combining three saved reconstructions increased accuracy error by **2.55%** and reduced completeness error by **0.58%**, relative to the mean of the three ordinary Full40 runs within each development room. Accuracy improved in Fire and worsened in Heads and Chess. The three solutions supplied visible depths at 94–96% of anchor pixels. Taking their median did not produce a more accurate surface across these rooms. The registered branch ended after development, before new-scene or longer-run comparisons. [Room-level results, coverage and controls](b_development_summary.md) · [Independent decision](b_development_independent.json)

## Efficiency at ten updates

Refresh1 lets all transformer stages adapt on the first update, then freezes the first twelve stage pairs and reuses their outputs for the remaining nine. The uncached twin follows the same parameter and optimizer schedule and recomputes those outputs. Full10 updates all stages ten times. Cache10 updates only the last twelve stage pairs ten times. Full40 supplies the full-length quality and resource reference.

At full resolution the cached and uncached engineering twins matched on all 1,177 signed-gradient, effective-weight and output comparisons. A deliberately stale cache kept its old values. Invalidating it changed 5,327,156 entries and recovered the independently recomputed values exactly. The first deterministic diagnostic exceeded the 8.75 GiB cap. The successful diagnostic kept saved autograd tensors on CPU. The two production controls used the ordinary GPU path. These tests establish implementation fidelity. Speed and geometry require the clean runs and surface evaluation. [Engineering evidence](a_engineering.json)

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
