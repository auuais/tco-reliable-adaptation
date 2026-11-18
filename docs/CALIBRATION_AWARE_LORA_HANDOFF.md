# Execution handoff for calibration aware LoRA research

This is the operational guide for a successor agent, including Solus. Read it with the [goal contract](CALIBRATION_AWARE_LORA_GOAL.md), [mathematical specification](FROZEN_LORA_ACCURACY_PLAN.md) and [design settings](../manifests/lora_profiled_design_v1.json). The contract defines the objective and scientific gates. This guide explains how to implement, run and verify them in the actual workspace. Use files and live process state to establish the current checkpoint before taking action.

**Objective and current checkpoint**

**Scientific release update:** both model bridges passed. The initial scientific-v1 P0 completed but its host verifier rejected Linux colon versus Windows U+F03A cache names. The narrow shared filename codec now verifies the same bytes. That unscored P0 is preserved in the C:/E: archive; use scientific v2 and evaluator execution v2 for the full 36-state screen. See the live progress file for actual running handles. Historical v1 commands and failed attempts below are retained for audit, not instructions to restart an obsolete run.

**Latest verified update:** deterministic execution resolved the earlier discrepancy. `gradient02`, under engineering v2 at commit `8c3de74`, passed exact retained/replay agreement, all three finite-difference perturbations, original-state integrity and a complete 20-pair macro-step (23.85 seconds; 4825.71 MiB peak allocated). `endpoint01` was launched next; inspect [progress.json](../results/lora_profiled_v1/progress.json) for its live/terminal state. Do not repeat the diagnosis below merely because it is retained as history. Scientific registration still requires the endpoint report. See [the engineering diagnosis](PROFILED_LORA_ENGINEERING_DIAGNOSIS.md) and [formal gradient evidence](../results/lora_profiled_v1/gradient_bridge.json).

The objective is to determine whether differentiating calibration while training only LoRA improves reconstructed geometry. MASt3R is the first model because its existing decoder adaptation fits this machine. VGGT transfer is conditional on a useful, repeatable MASt3R result. The original encoder, decoder and task-head parameters remain frozen; only newly installed LoRA factors are trainable.

Historical checkpoint immediately after `gradient01` (superseded by the update above): Preparation and CPU calculus checks pass. The real-model bridge, final calibration, endpoint bridge and single-case runner have now been implemented, but the GPU acceptance checks have **not** passed. The first local model run passed pristine-versus-zero-LoRA output equality and then failed retained-versus-replayed gradient agreement. No scientific adapter run, accuracy improvement or rental qualification exists yet. Consult [progress.json](../results/lora_profiled_v1/progress.json) for the live checkpoint; a file's existence does not establish that its experiment passed.

The original blocker was to diagnose this engineering discrepancy. That diagnosis and the repaired gradient acceptance are now complete; continue from the live checkpoint. At execution commit `8e33ba3839338daf9944939834c2834ee4f9d2d4`, both losses were `25.951331333991202`, but gradient relative L2 error was `0.0058830234318835144` (about 0.588%) and cosine was `0.9999827114918957`. The registered limits are relative L2 <= `1e-4` and cosine >= `0.99999`. Identical losses do not imply identical derivatives. See [failed comparison](../results/lora_profiled_v1/engineering/gradient01/replay_failure.json) and [failure traceback](../results/lora_profiled_v1/engineering/gradient01/failure.json). The container `tco-profiled-gradient01` exited with code 1, without an OOM; no research container was running at this checkpoint. Inspect live state again on resume.

Read in this order: this operational checkpoint, the goal contract, the mathematical specification, the design and bridge-threshold manifests, then the referenced implementation. User instructions take precedence. The contract defines the scientific decisions; the manifests freeze settings; actual reports establish completion. A conflict between prose, settings and code is an engineering issue to resolve explicitly before execution. Never silently substitute a setting or promote an old successful report across changed source hashes.

The user has authorized research execution, Git commits and publication of permitted code/results. Continue useful local work without repeatedly asking permission. The user allows best judgment for routine preferences unanswered after ten minutes. This does not waive monetary caps, access requirements or an explicit prohibition. Do not pause for a model switch unless the user requests it. The active goal remains active until its completion, blocking or user-requested pause conditions are met.

**Workspace and tools**

| Resource | Exact location or identity |
|---|---|
| Windows repository | `C:/SVNProjects/Research/tco-research` |
| Public Git remote | `https://github.com/auuais/tco-reliable-adaptation` |
| Container working directory | `/workspace` |
| Existing Docker image | `tco-mast3r:pilot` |
| Pinned image ID | `sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50` |
| Checkpoint directory | `E:/ResearchData/TCO/checkpoints/MASt3R/hf` mounted read-only at `/weights` |
| Checkpoint file | `model.safetensors` |
| Checkpoint SHA256 | `0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb` |
| Original public-data assets | `E:/ResearchData/TCO/data/7scenes` |
| Compact public evidence | `results/lora_profiled_v1/` |
| Large private execution outputs | `reports/lora_profiled_v1/` |
| Second physical-disk backup | `E:/ResearchData/TCO/research_results/lora_profiled_v1/` |

The verified container has Python 3.11.13, PyTorch 2.7.1+cu128 and CUDA 12.8. The GPU is an RTX 3080 with approximately 10 GiB VRAM. Host Python has a different PyTorch version. Use the pinned container for model experiments. CPU verifiers may run on the host when their dependencies are already available. Do not upgrade the research environment globally.

There are two unrelated pre-existing modifications: `docs/BASELINE_LENGTH_CONTROL.md` and `docs/NEIGHBOR_SPARSE_DEPTH.md`. Preserve them. Stage explicit file paths rather than `git add .`. If an isolated checkout is necessary, create it while retaining access to verified data and caches; do not erase changes to manufacture a clean tree.

**Start each resumed session**

In PowerShell, use the repository as the working directory. Inspect current state before deciding that a previous process ended:

```powershell
Set-Location C:/SVNProjects/Research/tco-research
git status --short
git log -3 --oneline
docker ps --format '{{.Names}} {{.Status}} {{.Image}}'
docker image inspect tco-mast3r:pilot --format '{{.Id}}'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
Get-Content results/lora_profiled_v1/progress.json
```

Read the native active-goal state and weekly usage counter using the available tools. If a recorded local job might still be running, inspect that exact container with `docker inspect NAME --format '{{json .State}}'` and `docker logs --tail 30 NAME`. A stale status file or observation timeout does not prove the job stopped. Reattach or poll the live handle; never start a duplicate merely because a previous observation timed out.

Require the current image ID to equal the pinned value in the table before using the tag. An old environment report cannot prove that a mutable tag still names the same image. Launch by the full inspected image ID when practical.

At 5% or less weekly allowance remaining, checkpoint the research, stop any paid research pod, verify shutdown and pause the active goal under the user's standing instruction. Do not confuse local Docker containers with RunPod pods. The retained cloud workspace must remain intact.

**Replay preparation before touching a model**

These commands are existing preparation/verifier entry points. They do not train a model:

```powershell
python scripts/prepare_profiled_inputs.py --verify
python scripts/check_profiled_splits.py --verify
python scripts/check_profiled_calibration.py --verify
python scripts/check_profiled_final_calibration.py --verify
python scripts/check_lora_profiled_gradient.py --verify
python scripts/profiled_baseline_graph.py --verify
```

For an original-source and native-remapping audit, including the actual checkpoint:

```powershell
python scripts/prepare_profiled_inputs.py --verify --assets E:/ResearchData/TCO/data/7scenes --weights E:/ResearchData/TCO/checkpoints/MASt3R/hf/model.safetensors --verify-native
```

The preparation registration already exists. Do not rerun its registration mode to overwrite it. Its verifier rejects changed hashes or splits. An intentional scientific change needs a new version, preserved predecessor and explicit explanation.

The graph verifier additionally needs the retained local `reports/mast3r_neural_transfer/cache/*_zero` caches. Integrate it as `matches, audit = load_baseline_graph(scene, device="cuda")`. It validates the archived pretrained matches and returns ten unordered pixel-edge dictionaries per scene. It does not supply learned confidence maps; use the prescribed frozen canonical confidences when preparing the geometric layer. The graph audit is not runnable on a public checkout lacking those large caches without reconstructing or restoring them.

The environment check already ran in local container `tco-profiled-environment02`. Read `results/lora_profiled_v1/environment.json`. The first attempt failed because Linux Git interpreted Windows CRLF checkout files as changed. The corrected audit compares every tracked source blob with its pinned Git object and accepts only explicitly recorded CRLF-to-LF equivalence. No upstream file was rewritten. See `engineering_failures.json`. Never bypass source-integrity checks by deleting evidence or resetting the upstream checkout.

For a new container operation, choose a fresh descriptive attempt name and keep its logs after failure. The standard local launch shape is:

```powershell
$profiledAttempt = 'tco-profiled-solver-' + (Get-Date -Format yyyyMMdd-HHmmss)
docker run --name $profiledAttempt --mount type=bind,source=C:/SVNProjects/Research/tco-research,target=/workspace --workdir /workspace sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50 python scripts/check_profiled_calibration.py --verify
```

Model jobs add `--gpus all` and the read-only checkpoint mount `--mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly`. The image already sets the MASt3R import path. Do not start a second GPU experiment while the first is active. The runner CLI now exists and its missing-manifest refusal has been checked; its actual training and endpoint execution remain unverified.

**Data roles and exact schema**

Each permitted adaptation packet is an existing file `results/mast3r_partner_shape/earlier.npz` or `later.npz`. Array names are `heads_25`, `chess_225`, and so forth. Columns are `[target_u_518, target_v_392, axial_depth_m, source_frame, source_u_640, source_v_480, source_depth_mm]`. Only the selected temporal packet is loaded for fitting a case. The opposite packet is development evaluation and must not be passed to its optimizer, weights, gauges or stopping rule.

Target images are frames `[25,225,425,625,825]`. The source offset is -4 for earlier and +4 for later. Scene identities are Heads/seq-01, Chess/seq-03 and Fire/seq-03. Use the existing case-sensitive asset paths recorded in input evidence.

`results/lora_profiled_v1/splits.npz` contains fields with prefixes such as `heads_earlier_25`:

| Suffix | Meaning |
|---|---|
| `__rows` | Original packet row indices, in the existing native-sample order |
| `__xy` | Integer native x,y pixels, shape N by 2 |
| `__source_ids` | Source frame,u,v, shape N by 3 |
| `__group` | Connected-component ID within the scene/side case |
| `__fold` | 0 or 1; support uses macro-step parity, query uses the opposite value |

Obtain measured depths as `packet[key][rows,2]`. Do not treat row numbers as sensor identities. Groups union repeated source rays across relevant views and shared native 8x8 cells within a view. The seeded split uses IDs, coordinates and group counts, with no depth-dependent search. The registered 1,225 retained observations have a minimum fold count of 11. Original-source preparation verified 75 files and reconstructed all 30 sparse packets exactly while opening zero target-depth files.

Native model pixels are 512 by 384. Packet pixels are 518 by 392. Use the existing mapping `round((xy+0.5)*[512/518,384/392]-0.5)` and the registered nearest-depth duplicate selection. Do not substitute floor, change half-pixel offsets or split duplicate rays independently.

The supplied clean pose packet is `results/mast3r_neural_transfer/pose_packets.npz`. Its matrices are world-to-camera; invert to obtain the camera-to-world matrices required by the new solver. Obtain native intrinsics through the existing `intrinsic()` function in `scripts/run_mast3r_neural_transfer.py`. It includes the established resize and 1.12 focal convention. These supplied reference-derived cameras define the initial oracle-prior condition; label it accordingly.

Support and query both train LoRA. Opposite-source scores may nominate a method only after all registered predictions are sealed. Consequently these scenes are development data. Publication tests require prospectively frozen new scenes and a second dataset. Do not count two temporal directions as six independent scenes.

**Code responsibilities and interfaces**

| Existing file | Reuse or inspect |
|---|---|
| `scripts/prepare_profiled_inputs.py` | Immutable source audit, grouped split registration and replay |
| `scripts/profiled_calibration.py` | New differentiable five-scale geometry |
| `scripts/check_profiled_calibration.py` | Production-solver CPU checks |
| `scripts/check_profiled_environment.py` | Pinned image/checkpoint/source and CUDA verification |
| `scripts/profiled_baseline_graph.py` | Sealed pretrained correspondence graph and audit; confirm current CLI before use |
| `scripts/profiled_network.py` | Case loading, fixed teacher, two spatial states, arm losses, output adjoints and pair replay |
| `scripts/check_profiled_lora_bridge.py` | Registered real-model gradient/finite-difference/full-macro-step acceptance; first attempt failed |
| `scripts/diagnose_profiled_replay.py` | Arithmetic diagnostic without an optimizer update; implemented, real-model run pending |
| `scripts/profiled_final_calibration.py` | Common final gauge, native camera convention and five-scale refinement |
| `scripts/check_profiled_endpoints.py` | Pristine/zero-adapter endpoint acceptance and reusable fresh capture; GPU check pending |
| `scripts/register_profiled_execution.py` | Versioned immutable engineering/scientific registration and fail-closed source verification |
| `scripts/run_profiled_lora.py` | One registered case, eight macro-steps or P0, fresh endpoints and sealed attempt evidence; GPU run pending |
| `scripts/mast3r_network_priors.py` | LoRA installation, state digest, native sample mapping and original D objective |
| `scripts/mast3r_network_capture.py` | Image preprocessing, frozen features, pair decoder, native capture |
| `scripts/run_mast3r_neural_transfer.py` | Native intrinsics, original triplet schedule and historical training |
| `scripts/mast3r_calibrated_alignment.py` | Native KDP prior callback |
| `scripts/mast3r_alignment_hook.py` | Context-managed alignment callback installation |
| `scripts/score_partner_shape.py` | Existing held-source evaluation conventions |
| `scripts/partner_bias_surface.py` | Matched surface evaluation and component reporting |

The new calibration API uses tensors on a common device:

```python
depth, confidence = canonical_depth(pointmaps, confidences)
# pointmaps [partners,H,W,3], confidences [partners,H,W]

state = prepare_calibration(
    initial_depths, intrinsics, c2w, support, matches, fused_confidences
)
# depths/confidences [5,H,W], intrinsics [5,3,3] or [3,3], c2w [5,4,4]
# support/query: five dictionaries {'xy': [N,2], 'depth': [N]}
# matches: unordered dictionaries {'i': i, 'j': j, 'xy_i': [M,2], 'xy_j': [M,2]}

loss, scales = profiled_objective(
    current_depths, state, query, graph_weight=0.1, detach_solve=False
)
```

Read the module docstrings and current implementation before calling it. Fixed observations, gains and confidence weights are detached clones. The small solve uses float64. `detach_solve=True` changes backward propagation only; its forward scales and objective must match the full-gradient version. Confidence weights stay pretrained throughout the initial factorial. Negative candidate query predictions retain their residual; candidate-dependent masks must not let the model discard difficult evidence.

Initial invalid matched endpoints are filtered using the frozen teacher and their counts are recorded. Nonpositive support predictions require a prospective engineering decision preserving split integrity. Do not silently remove support after looking at query error. Nonpositive fitted scales invalidate the unconstrained solve. Record that failure and test a positive-scale formulation separately before a scientific batch.

All new calibration medians use the midpoint convention, `torch.quantile(values, 0.5)`, including support gauges, sensor normalizers and final g0. P1 preserves its legacy `torch.median` lower-middle convention for an even number of samples. Do not homogenize these helpers while claiming to preserve the existing D comparator.

**Remaining work, in dependency order**

1. Completed: run the arithmetic diagnostic described below. It compares graph-cut values, loss adjoints, individual pair VJPs, accumulation, repeated backward passes and FP32/BF16 behavior. Establish the cause of `gradient01` before changing production code. No accuracy label is needed.
2. Completed for engineering v2: repair and re-register the deterministic implementation. Re-run zero-LoRA equality, retained/replay agreement, the real adapter finite differences, frozen-state hashes and the full 20-pair macro-step. Keep the failed attempt. The finite-difference and macro-step checks were not reached in `gradient01`.
3. Run the existing endpoint bridge with fresh pristine and zero-adapter caches. Require separate native and final-calibrated equality, finite alignment callbacks, matching provenance and unchanged model/adapter state. The CPU final-calibration fixture alone does not establish this real endpoint bridge.
4. Review and exercise the existing `run_profiled_lora.py` after both bridges pass. Register the scientific manifest against the finished source and passed reports. Its dry-run verifies one case; the registration tool enumerates the complete 36-case inventory. Check one complete local scientific state before proceeding serially through the registered inventory, without opening opposite-source scores. A completed first state can count toward the screen only if its settings and code are unchanged.
5. Implement the still-missing `scripts/score_profiled_lora.py` and its global seal/nomination checks from the contract below before opening scores. Dense evaluation follows the existing registered conventions and matched input controls. Do not reuse the old scorer unchanged: it expects historical `zero/real/shuffled` shape experiments, not the six new neural arms.
6. Add compact replay/CI entry points as meaningful evidence becomes available. Keep numerical fixtures separate from GPU-dependent checks. Publish all planned runs, including failures, and record why conditional experiments were not run.

**Exact gradient replay algorithm**

Keep one adapter state fixed for an entire macro-step. In the first pass, run each of the 20 directed pairs without retaining decoder activations. Save the point tensors needed by the selected objective. Turn their detached copies into differentiable leaves, fuse target depths, solve calibration and compute the loss. Differentiate the loss with respect to these leaves to obtain output adjoints. Recompute each pair at identical parameters, precision, random state and buffers, then backpropagate its saved adjoint. Accumulate all pair contributions. Only then clip the full gradient and take one optimizer step.

No optimizer step may occur between replayed pairs. Do not retain 20 full decoder graphs just because the output tensors fit. Freeze confidence/matching selection in the training objective and maintain the same decoder-call schedule in all neural arms. Features can be cached because the encoder is frozen. P1 reuses its two output branches and the arithmetic mean over `schedule(20)`; repeated use of a pair contributes summed adjoints. P2–P5 use the fused self-depth objective. One factor's initial zero gradient is expected when the opposite LoRA factor is zero; the active factor must receive finite useful gradients.

Reduced precision can make tiny finite differences meaningless. Run the numerical direction check with an fp32 decoder and TF32 disabled if necessary, with fixed features and several prespecified perturbation sizes. Record arithmetic settings and tolerances before opening accuracy results. A gradient check must cover the implementation actually claimed; a diagnostic of a different precision path cannot prove every reduced-precision detail.

**Immediate diagnostic and release commands**

The following commands are entry points, not claims that these runs have passed. Start with the inspection commands above, check free disk space and weekly allowance, and commit the exact diagnostic source before running it. Use an unused attempt/container name; the scripts intentionally refuse to overwrite an attempt. No paid GPU is involved.

```powershell
$profiledImage = 'sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50'
$profiledAttempt = 'replaydiag-' + (Get-Date -Format yyyyMMdd-HHmmss)
$profiledContainer = 'tco-profiled-' + $profiledAttempt
docker run --name $profiledContainer --gpus all --mount type=bind,source=C:/SVNProjects/Research/tco-research,target=/workspace --mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly --workdir /workspace $profiledImage python scripts/diagnose_profiled_replay.py --attempt $profiledAttempt --weights /weights --image $profiledImage
```

Read `results/lora_profiled_v1/engineering/ATTEMPT/report.json`, or its `failure.json` and partial precision reports. This diagnostic makes no optimizer update. It records the actual RoPE implementation; the first run used the slow PyTorch fallback, so do not assume a compiled extension caused the mismatch. Its current checks distinguish the following possibilities:

| Evidence | Next decision |
|---|---|
| Forward outputs differ between graph-free and gradient-enabled recomputation | Inspect autocast, features, random state, buffers and execution paths before reasoning about accumulation |
| Outputs agree but graph-cut loss adjoints differ | Inspect canonical fusion, detached state, objective construction and tensor aliasing |
| Outputs/adjoints agree but individual pair VJPs differ | Inspect backward kernels, tensor mutation and precision; preserve supplied-adjoint mutation checks |
| Individual pair VJPs agree but joint backward differs | Examine accumulation/cast placement and repeated-run variation; do not call this exact retained-graph agreement |
| FP32 agrees but BF16 fails | A precision cause is supported only after the component comparisons agree; revise arithmetic consistently across all neural arms and re-register if necessary |
| Both precisions fail or finite differences fail | Treat as an unresolved implementation/math error; no scientific launch |
| All relevant comparisons pass after a fix | Run the registered acceptance bridge, including finite differences and the full macro-step; the diagnostic is not a substitute |

Retain the original thresholds and failed evidence. If a tolerance revision is scientifically justified by measured numerical repeatability, document that as a new prospective engineering protocol with independent review; never edit the archived threshold manifest to turn a failed run into a pass. Prefer repairing the computation or using a common more accurate arithmetic path.

After diagnosis and any source fix, preserve `lora_profiled_engineering_v1.json` and create a new version. V1 is a valid historical snapshot and is expected to reject the changed current source. It must not be overwritten or hand-edited.

```powershell
python scripts/register_profiled_execution.py --prepare --phase engineering --output manifests/lora_profiled_engineering_v2.json --supersedes manifests/lora_profiled_engineering_v1.json
python scripts/register_profiled_execution.py --verify --phase engineering --manifest manifests/lora_profiled_engineering_v2.json
```

Commit the revised source and new registry, then run these **sequentially**, using new names if any already exist. Do not start the endpoint job if the gradient job fails.

```powershell
docker run --name tco-profiled-gradient02 --gpus all --mount type=bind,source=C:/SVNProjects/Research/tco-research,target=/workspace --mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly --workdir /workspace $profiledImage python scripts/check_profiled_lora_bridge.py --weights /weights --manifest manifests/lora_profiled_engineering_v2.json --attempt gradient02 --image $profiledImage
if ($LASTEXITCODE -ne 0) { throw 'Gradient bridge failed; endpoint launch blocked' }
$profiledGradientReport = Get-Content results/lora_profiled_v1/gradient_bridge.json -Raw -ErrorAction Stop | ConvertFrom-Json
if ($profiledGradientReport.pass_all -ne $true) { throw 'Gradient report is not passing; endpoint launch blocked' }
docker run --name tco-profiled-endpoint01 --gpus all --mount type=bind,source=C:/SVNProjects/Research/tco-research,target=/workspace --mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly --workdir /workspace $profiledImage python scripts/check_profiled_endpoints.py --weights /weights --manifest manifests/lora_profiled_engineering_v2.json --attempt endpoint01 --image $profiledImage
if ($LASTEXITCODE -ne 0) { throw 'Endpoint bridge failed; scientific registration blocked' }
```

Only passed reports at `results/lora_profiled_v1/gradient_bridge.json` and `endpoint_bridge.json`, bound to current sources/inputs/runtime, release scientific registration. A report with a stale source hash is not a pass for the new implementation. Reuse of a successful evidence filename after a scientific code change requires preserving/versioning the old report and updating its registry references; never delete the old report to bypass the guard.

```powershell
python scripts/register_profiled_execution.py --prepare --phase scientific
python scripts/register_profiled_execution.py --verify --phase scientific --manifest manifests/lora_profiled_execution_v1.json
python scripts/run_profiled_lora.py --run-id heads_earlier_P4 --manifest manifests/lora_profiled_execution_v1.json --attempt screen01 --weights /weights --dry-run
```

At the present checkpoint the first two commands cannot succeed because both bridges are missing; the runner also refuses a missing scientific manifest. Commit the scientific manifest before a real job. Real jobs require the explicit inspected image ID (`--image`, or `PROFILED_DOCKER_IMAGE`); the registry's `planned_argv` is a template that the launcher must complete with this identity.

```powershell
docker run --name tco-profiled-heads-earlier-p4-screen01 --gpus all --mount type=bind,source=C:/SVNProjects/Research/tco-research,target=/workspace --mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly --workdir /workspace $profiledImage python scripts/run_profiled_lora.py --run-id heads_earlier_P4 --manifest manifests/lora_profiled_execution_v1.json --attempt screen01 --weights /weights --image $profiledImage
```

**Scheduling, recovery and evidence ownership**

The registry enumerates 3 scenes x 2 adaptation sources x 6 arms = 36 states, ordered by scene, source and P0 through P5. Thirty trained states require 240 total optimizer updates; six P0 controls require none. Each trained state uses 20 teacher calls and eight steps of 20 graph-free plus 20 differentiable calls, followed by separately counted fresh reconstruction calls. The single-case runner does not schedule this factorial. Validate one complete neural path and one P0 path before scheduling the remainder, without viewing development accuracy.

An eventual launcher must run one GPU job at a time, require a successful exit and valid `complete.json`, and stop on engineering failures. Before skipping a completed case, recompute its manifest, source, input and output hashes. The presence of a folder or adapter file is insufficient. Choose the first valid completed attempt under the same frozen protocol; never choose a rerun by accuracy. Partial runs are preserved and restarted from pretrained weights with a fresh attempt ID because mid-step resume is not implemented. Diagnose repeat failures; do not run an unlimited retry loop. The `scientific_gpu.lock` protects only this runner, so inspect other containers/processes as well. Remove a stale lock only after proving its owner process/container ended and preserving its contents.

Public attempt evidence belongs in `results/lora_profiled_v1/cases/RUN_ID/ATTEMPT/`; large arrays and fresh caches belong in the corresponding `reports` path. Required outputs include source/manifest/input hashes, original-state hashes, initial/final adapter hashes, eight per-step reports for trained cases, actual call counts, finite gradient checks, timing/memory, native/final endpoint identities, and the completion/failure record. The runner's 1 GiB free-space guard is a minimum, not a 36-case storage estimate. Measure the first complete case's bytes and reserve enough local disk for the remaining cases and both physical backups before batching. Never place pretrained weights or complete restricted datasets in Git.

**Scoring and selection contract to implement**

The rules are frozen separately in [lora_profiled_evaluation_v2.json](../manifests/lora_profiled_evaluation_v2.json). The scientific registry binds that specification; an evaluation-execution manifest must additionally bind the final scorer source and its fixture results before accessing opposite-source scores. Neither a per-case completion record nor a registry flag implements a global prediction seal.

Create a global seal with exactly the 36 registered run IDs, one valid attempt per ID, common scientific-registry hash, completion-report hashes, adapter identities, and native/final endpoint hashes. Verify all referenced bytes, model/input identities and complete status. Reject missing/duplicate cases, mixed registrations, partial attempts and changed predictions. Seal all predictions before loading the opposite-source evaluation packet into the scoring process. An unrecoverable case must remain a reported failure under a prospectively amended termination protocol; it cannot disappear from the denominator or yield a qualifying nominee.

Reuse the *mapping and overlap logic* of `score_partner_shape.disjoint_test`, not its historical scorer. For each target, map/deduplicate the opposite packet with the registered native-pixel rule and exclude pixels used by the entire allowed adaptation packet (both spatial folds). Exclusion is determined only by input identity, never candidate error/confidence. Keep the same evaluation observations for every arm. For arm a, scene s, side t and view v, compute `MAE[a,s,t,v] = mean(abs(predicted_final_depth - measured_opposite_depth))` in meters over those fixed points. Average equally over five views, then equally over three scenes, keeping earlier/later directions separate. Define directional gain as `1 - mean_MAE[a,t] / mean_MAE[P1,t]`, not the mean of per-point percentage changes. Fail on empty views or nonfinite predictions; count nonpositive predictions and disqualify affected candidates rather than excluding their pixels.

Only P3 and P4 are candidates for the initial implicit-gradient hypothesis. P0, P1, P2 and P5 are controls. A detached arm winning is useful evidence against the mechanism, not permission to rename it a successful full-gradient method. Nominate at most one candidate with >=1% directional gain in each direction, improvement on at least two of the same three scenes in both directions, and no numerical failures. Maximize the smaller directional gain. Compare gains rounded to 12 decimal places for ties, then lower measured total runtime across its six cases, then smaller numeric arm ID. Always publish all six arms and the P3/P2 and P4/P5 contrasts. Do not use native-only results to select when the final-refined primary result fails.

The scorer needs meaningful fixtures for a missing/tampered seal, a mixed-registry case, overlap exclusion, unequal observation counts, invalid predictions, the two-same-scenes rule, tie handling and a deliberately wrong endpoint. Fixtures must pass before study labels are scored. Publish compact per-view predictions/observations sufficient for independent CPU replay where the dataset terms permit.

For M3, freeze a separate executable dense-evaluation manifest before its scores are read. Use `registered_rgb` as the primary surface reference and `released` as alternate, with the pinned similarity-plus-ICP evaluator and fixed reference masks. For each scene/condition/reference, let A be mean accuracy distance and C mean completeness distance in meters. Normalize each component by the same-condition P0 component and set `surface_score = 0.5*(A/A_P0 + C/C_P0)`; aggregate equally by scene. Compare the nominee against the lowest aggregate primary score among the predeclared eligible same-input controls, minimally P0/P1/P2/P5, with smaller arm ID breaking exact ties. Freeze the eligible pool before dense scoring; add compatible established baselines only after documenting their exact inputs. Historical runs with two-source data or a different final calibration cannot replace these controls. Use that same selected control for per-scene, component and alternate-reference checks; report raw meter-valued distances as well.

The M3 manifest must enumerate both temporal directions, centered/shifted grids and clean/noisy supplied-pose conditions: eight gate cells, each with three scenes and matched controls. Apply all rental criteria separately to every cell; do not pool away a failed direction or stress condition. Shifted-source packets and spatial splits need their own verified preparation. Specify camera-noise seeds, masks, grid construction, native evaluator revision and all runtime commands before opening these scores. Initially preserve the established pose-only stress channel; any physical sensor-projection corruption is a separate registered test. Perform three fresh-process paired confirmations with the same optimizer seed and predeclared corruption draws, reporting deterministic execution variation separately from input perturbation. The old manifest is a source of conventions, not authority to reuse unmatched old predictions.

**Final reconstruction convention**

All arms use fresh model outputs, fresh matching and native KDP alignment with 300 coarse/300 fine iterations and pose weight 0.01. Use identical permitted observations and supplied inputs. Report native-only outputs as the secondary endpoint.

For the primary endpoint, retain native optimized K and rotations. Compute a single global g0 from the median of valid measured/native-depth ratios over all permitted adaptation observations. Convert both native depths and translations with this same factor: `C_i=g0*(C_i_native-C_0_native)` and `z_i=g0*z_i_native`. Fit five relative scales around one using the same graph weight 0.1 and frozen final normalizers, and reconstruct `X_i=C_i+s_i*R_i*inv(K_i)*p*z_i`. Freeze the fresh native correspondences during this final solve. Do not replace native camera centers with supplied centers or apply g0 twice.

Use this exact solver convention to avoid the default per-view gauge in `prepare_calibration`. Keep `native_depth` in its original native units and pass the common conversion explicitly:

```python
metric_c2w = native_c2w.clone()
metric_c2w[:, :3, 3] = g0 * (native_c2w[:, :3, 3] - native_c2w[0, :3, 3])
final_state = prepare_calibration(
    native_depth, native_K, metric_c2w,
    all_allowed_adaptation_samples, fresh_native_matches, fresh_fused_confidences,
    metric_scales=torch.ones(5, device=native_depth.device, dtype=torch.float64) * g0,
)
A, b = assemble_system(native_depth, final_state, graph_weight=0.1)
scales = solve_scales(A, b, final_state["ridge"])
final_depth = scales[:, None, None] * g0 * native_depth
```

Compute g0 with the conventional midpoint median, matching `torch.quantile(values, 0.5)` on positive finite ratios. This final all-observation fit has no support/query split and does not call `profiled_objective` or `query_residual`. Both of those require disjoint adaptation folds and serve training. Reconstruct the world points using `final_depth`, `metric_c2w` and native K, with no additional scale multiplication.

Every permitted final fitting observation must have finite, positive predicted depth. If any fails, reject the attempt; do not interpret "valid ratios" as permission for a candidate-dependent subset. Negative predictions outside fitting observations remain in the saved depth map and are counted. The final graph's frozen initial validity filtering is recorded separately; it must not be confused with permission to drop difficult sensor observations.

Every comparator gets the same final intervention. A baseline without final calibration cannot isolate the neural objective. Both endpoint outputs and the evaluator's actual input hashes must be saved.

For the secondary axial-depth score, convert the saved native depth to meters with the same sealed adaptation-only global g0: `native_metric_depth = float64(native_depth) * g0`. Do not multiply by the five fitted relative scales. Raw native depth can have an arbitrary gauge and must not be labeled a meter-valued prediction. The scorer checks positive finite g0/scales and the full saved relationship `final_depth = scales * g0 * native_depth` before comparison. Evaluation rules v2 preserve v1 and clarify this unit convention before scientific execution; the primary endpoint and all advancement thresholds are unchanged.

**Decisions and reporting**

The exact six-arm design, 1% nomination gate, 5% surface rental gate, tie rule, repeatability requirement and bounded redesign are in the goal contract. Implement those rules once in a scorer and verify them on a small meaningful fixture. Do not choose the best endpoint after seeing scores. Report the full-versus-detached contrasts even when another baseline gives a larger apparent improvement.

If the full-gradient arm ties its detached control within repeat variation, the proposed mechanism is unsupported. If the surrogate improves while the native surface worsens, investigate solver absorption and the reconstruction mismatch. If engineering checks fail, fix implementation before interpreting accuracy. If a sound method fails, retain that evidence and register the single permitted targeted local redesign. Do not automatically spend the transfer reserve after failure.

For each completed attempt, retain the execution commit, source/config hashes, permitted packet and split hashes, initialization, optimizer steps, gradient diagnostics, predictions, evaluator revision, runtime, peak memory and cost. Keep large arrays in excluded reports and copy the full archive to the second disk. Publish compact data sufficient for the claimed CPU replay, subject to dataset/code licenses. Verify both backup hashes before treating evidence as durable.

**Budget and stopping rules**

Initial work is local and adds USD 0 rental compute. The existing conditional USD 25 is a total ceiling, not a balance to use up: USD 8 for transfer feasibility, USD 12 for one confirmation batch, USD 5 for contingency/storage. It does not add funds to the preferred USD 200 project ceiling. Check actual balance, live quote, ledger and retained storage before any paid launch. Existing credit observations are stale and private.

Prepare data, wheels, code and manifests locally. Start a single paid job stream only after its evidence gate, with the tested provider-side deadline and heartbeat/lease. A shell timeout does not stop billing. On success, failure or weekly reserve threshold, preserve outputs, stop the paid pod and verify provider state. The retained workspace `749tvrn5gwvuel` and its 20 GB volume must not be deleted. Do not request deletion again without a material change.

**First actions for the successor**

Read the live progress file, inspect actual jobs, and replay preparation only when files/runtime changed or their validity is uncertain. The preserved `gradient01` failure remains historical. Use the passed `gradient02` evidence for gradient/finite-difference/full-step acceptance, and verify the current endpoint state before proceeding. Complete the production bridges and freeze the scientific manifest before scheduling the factorial. After each meaningful result, update the evidence paths, job state, next command, run counts, costs and Git/CI state. The next agent should be able to resume from these files without reconstructing this conversation.
