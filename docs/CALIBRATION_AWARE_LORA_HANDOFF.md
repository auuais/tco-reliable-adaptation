# Execution handoff for calibration aware LoRA research

This is the operational guide for a successor agent, including Solus. Read it with the [goal contract](CALIBRATION_AWARE_LORA_GOAL.md), [mathematical specification](FROZEN_LORA_ACCURACY_PLAN.md) and [design settings](../manifests/lora_profiled_design_v1.json). The contract defines the objective and scientific gates. This guide explains how to implement, run and verify them in the actual workspace. Use files and live process state to establish the current checkpoint before taking action.

**Objective and current checkpoint**

**Operational checkpoint, November 18, 2026, 11:55 UTC:** both real-model bridges and the first P0/P1 scientific validation pair passed. The full local `screen02` queue is active under invocation `20261118T104117Z_ede54062`; its observation session is `27750`. At the last observation, 17 of 36 states had completed and verified, and `tco-profiled-screen02-chess-earlier-p5` was running. This count is a timestamped checkpoint, not a live assertion. Read [progress.json](../results/lora_profiled_v1/progress.json), the latest invocation receipts and actual Docker state before acting. Observation handles may not survive a model/session switch; their absence never authorizes a duplicate queue.

Scientific registration is [lora_profiled_execution_v2.json](../manifests/lora_profiled_execution_v2.json); scorer registration is [lora_profiled_evaluation_execution_v2.json](../manifests/lora_profiled_evaluation_execution_v2.json). The first two-state invocation `20261118T103337Z_7333c162` finished with `status=success` and `selected_jobs_verified=2`; it is history, not the operation to resume. Keep the current queue running without changing its bound sources or inputs. No study accuracy scores have been accessed; no accuracy or rental gate has passed. Whole-milestone progress remains 25% (M0 and M1). Rental compute for this goal remains USD 0.

**Successor's immediate task:** observe the existing queue, finish the unbound M3 CPU infrastructure where useful, then seal all 36 predictions and use the already registered M2 scorer. Do not redesign the objective, spend on transfer, or open a convenient subset of scores while waiting. The later M3 implementation map and decision table identify exactly what remains.

`gradient02`, under engineering v2 at commit `8c3de74`, passed exact retained/replay agreement, all three finite-difference perturbations, original-state integrity and a complete 20-pair macro-step (23.85 seconds; 4825.71 MiB peak allocated). `endpoint01` passed exact pristine/zero-LoRA native and final endpoint equality, matching/confidence equality and unchanged original/adapter states. Preserve these reports and verify their registered hashes; do not repeat them without a relevant source/runtime change. See [the engineering diagnosis](PROFILED_LORA_ENGINEERING_DIAGNOSIS.md), [gradient evidence](../results/lora_profiled_v1/gradient_bridge.json) and [endpoint evidence](../results/lora_profiled_v1/endpoint_bridge.json).

The objective is to determine whether differentiating calibration while training only LoRA improves reconstructed geometry. MASt3R is the first model because its existing decoder adaptation fits this machine. VGGT transfer is conditional on a useful, repeatable MASt3R result. The original encoder, decoder and task-head parameters remain frozen; only newly installed LoRA factors are trainable.

**Preserved history, not pending work:** `gradient01` at commit `8e33ba3839338daf9944939834c2834ee4f9d2d4` passed zero-adapter prediction equality but failed retained/replayed derivatives: relative L2 `0.0058830234318835144` and cosine `0.9999827114918957`, against unchanged limits `1e-4` and `0.99999`. Strict deterministic arithmetic resolved the discrepancy. See its [failed comparison](../results/lora_profiled_v1/engineering/gradient01/replay_failure.json) and [traceback](../results/lora_profiled_v1/engineering/gradient01/failure.json). The scientific-v1 `screen01` P0 subsequently completed in 89.29 seconds with exit code 0 and unchanged original weights, but its host verifier rejected five Linux colon/Windows U+F03A cache names. The narrow filename codec now verifies all original bytes. That unscored case is archived on C: and E: with [its archive receipt](../results/lora_profiled_v1/archived_cases/scientific_v1_heads_earlier_P0_screen01.json); it does not count toward scientific v2. Neither historical failure should be rerun under its old attempt name.

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

Model jobs add `--gpus all`, `--env CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the read-only checkpoint mount `--mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly`. The image already sets the MASt3R import path. The registered batch launcher supplies these arguments and pins the local Docker engine. Use it for scientific states. Both production bridges and the first P0/P1 invocation passed; the full scientific queue is now active. Do not start a second GPU experiment while it is active. CPU-only M3 probes use names beginning `tco-m3-`, no `--gpus`, and `CUDA_VISIBLE_DEVICES=`; the M2 launcher deliberately treats `tco-profiled-` containers as possible competing jobs.

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
| `scripts/check_profiled_lora_bridge.py` | Registered real-model gradient/finite-difference/full-macro-step acceptance; gradient02 passed |
| `scripts/diagnose_profiled_replay.py` | Completed no-update arithmetic diagnosis; retained as an engineering tool |
| `scripts/profiled_final_calibration.py` | Common final gauge, native camera convention and five-scale refinement |
| `scripts/check_profiled_endpoints.py` | Passed pristine/zero-adapter endpoint acceptance and reusable fresh capture |
| `scripts/register_profiled_execution.py` | Versioned immutable engineering/scientific registration and fail-closed source verification |
| `scripts/run_profiled_lora.py` | One registered case, eight macro-steps or P0, fresh endpoints and sealed attempt evidence |
| `scripts/run_profiled_batch.py` | Serial local Docker queue, validated skips, per-invocation timeout and durable exit evidence |
| `scripts/profiled_artifacts.py` | Narrow logical filename codec for byte-identical Linux/Windows cache verification |
| `scripts/seal_profiled_predictions.py` | Global 36-state seal, complete byte audit and prospective failure classification |
| `scripts/score_profiled_lora.py` | Frozen held-source scorer, nomination rule and independent replay output |
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

1. Completed: deterministic arithmetic diagnosis, both real-model acceptance bridges, scientific v2 registration and scorer v2 registration. The scorer's 62 synthetic fixtures and the batch/filename helper's 21 CPU fixtures passed before release. Preserve the failed attempts and old registrations.
2. Completed: the initial `screen02` P0/P1 validation invocation, successful exits, valid completion seals and successful launcher terminal record. Do not repeat it.
3. Observe the active full `screen02` queue. Resume only after establishing that its launcher and owned container have ended and classifying any failure. The launcher rechecks completed hashes before skipping them. Seal all 36 states before scoring; do not edit registered source or inputs during the screen.
4. Create and verify the global seal, score with the already registered evaluator, and independently replay the result. Apply the declared nomination rule. A passing derivative test, smaller training loss or isolated good scene cannot substitute for this gate.
5. If a candidate qualifies, prepare the still-pending dense evaluation with the prescribed matched controls and eight gate cells before opening dense scores. Otherwise diagnose failure and use at most the one targeted local redesign allowed by the contract. No rental allocation is released by a failed gate.
6. Publish planned runs including failures, verified backups, compact replay evidence and CI results; record why conditional stages did not run.

**Exact gradient replay algorithm**

Keep one adapter state fixed for an entire macro-step. In the first pass, run each of the 20 directed pairs without retaining decoder activations. Save the point tensors needed by the selected objective. Turn their detached copies into differentiable leaves, fuse target depths, solve calibration and compute the loss. Differentiate the loss with respect to these leaves to obtain output adjoints. Recompute each pair at identical parameters, precision, random state and buffers, then backpropagate its saved adjoint. Accumulate all pair contributions. Only then clip the full gradient and take one optimizer step.

No optimizer step may occur between replayed pairs. Do not retain 20 full decoder graphs just because the output tensors fit. Freeze confidence/matching selection in the training objective and maintain the same decoder-call schedule in all neural arms. Features can be cached because the encoder is frozen. P1 reuses its two output branches and the arithmetic mean over `schedule(20)`; repeated use of a pair contributes summed adjoints. P2–P5 use the fused self-depth objective. One factor's initial zero gradient is expected when the opposite LoRA factor is zero; the active factor must receive finite useful gradients.

Reduced precision can make tiny finite differences meaningless. Run the numerical direction check with an fp32 decoder and TF32 disabled if necessary, with fixed features and several prespecified perturbation sizes. Record arithmetic settings and tolerances before opening accuracy results. A gradient check must cover the implementation actually claimed; a diagnostic of a different precision path cannot prove every reduced-precision detail.

**Current v2 inspection and resume commands**

First inspect the existing operation. Poll its observation session while available. Docker state, the batch lock, per-job exit records and the invocation terminal record remain durable references if that session disappears. A missing `terminal.json` means the launcher has not recorded a terminal result; it does not authorize a duplicate.

```powershell
docker ps --format '{{.Names}} {{.Status}}'
Get-Content reports/lora_profiled_v1/batch.lock -ErrorAction SilentlyContinue
Get-Content results/lora_profiled_v1/batches/screen02/invocations/20261118T104117Z_ede54062/terminal.json -ErrorAction SilentlyContinue
Get-ChildItem reports/lora_profiled_v1/batches/screen02/invocations/20261118T104117Z_ede54062/*.log
```

The first two-state invocation already passed. While the full invocation is active, observe it rather than executing this resume block. If it finishes successfully with all 36 verified states, proceed directly to the global seal. If a later continuation is necessary, first inspect `*.exit.json`, individual completion records, lock owner and actual container state; preserve and classify the interruption. Only after proving there is no live owner and resolving the interruption, run this block sequentially. Each failure blocks the next command:

```powershell
python scripts/register_profiled_execution.py --verify --phase scientific --manifest manifests/lora_profiled_execution_v2.json
if ($LASTEXITCODE -ne 0) { throw 'Scientific v2 verification failed' }
python scripts/score_profiled_lora.py verify-registration --registration manifests/lora_profiled_evaluation_execution_v2.json
if ($LASTEXITCODE -ne 0) { throw 'Evaluator v2 verification failed' }
python scripts/run_profiled_batch.py --manifest manifests/lora_profiled_execution_v2.json --attempt-prefix screen02 --dry-run
if ($LASTEXITCODE -ne 0) { throw 'Batch preflight failed; preserve and investigate' }
python scripts/run_profiled_batch.py --manifest manifests/lora_profiled_execution_v2.json --attempt-prefix screen02 --execute --max-wall-hours 6 --min-free-gib 20
if ($LASTEXITCODE -ne 0) { throw 'Batch stopped; inspect its terminal and per-job exit records before recovery' }
```

The last command verifies and skips completed states under the same prefix, then runs the remaining manifest-ordered states. It does not restart the first P0/P1 pair or select attempts by accuracy. The wall-clock bound applies to this local invocation and does not authorize a rental. The user's 5% weekly reserve still applies. The launcher pins the local Docker named pipe, exact image and mounts, deterministic cuBLAS environment and one job at a time. On timeout/interruption it stops only its own labeled container. No automatic retries occur.

After all 36 predictions have individual completion seals, create and verify the global seal. These commands audit byte hashes without deserializing scientific predictions or accessing held-source accuracy:

```powershell
python scripts/seal_profiled_predictions.py prepare --manifest manifests/lora_profiled_execution_v2.json --evaluation manifests/lora_profiled_evaluation_execution_v2.json
if ($LASTEXITCODE -ne 0) { throw 'Global prediction seal failed' }
python scripts/seal_profiled_predictions.py verify --manifest manifests/lora_profiled_execution_v2.json --evaluation manifests/lora_profiled_evaluation_execution_v2.json
if ($LASTEXITCODE -ne 0) { throw 'Global prediction seal verification failed' }
```

Require 36 states and `failure_review_required=false` before scoring. If the seal already exists, use `verify`; `prepare` refuses to overwrite it. The following is the first planned release of study labels, followed by independent recomputation from the sealed predictions:

```powershell
python scripts/score_profiled_lora.py score --registration manifests/lora_profiled_evaluation_execution_v2.json
if ($LASTEXITCODE -ne 0) { throw 'Scoring failed; preserve all partial evidence' }
python scripts/score_profiled_lora.py verify --registration manifests/lora_profiled_evaluation_execution_v2.json
if ($LASTEXITCODE -ne 0) { throw 'Score replay disagreed' }
```

If scores already exist, use `verify`, never another `score` call to overwrite them. Registrations, successful bridge names and historical attempts are immutable. A relevant source change needs a new registration and an explicit scientific protocol decision; do not promote old results to a changed source hash. The completed engineering commands and failure diagnosis are preserved in `PROFILED_LORA_ENGINEERING_DIAGNOSIS.md` and archived reports, rather than being next actions for this successor.

**Scheduling, recovery and evidence ownership**

The registry enumerates 3 scenes x 2 adaptation sources x 6 arms = 36 states, ordered by scene, source and P0 through P5. Thirty trained states require 240 total optimizer updates; six P0 controls require none. Each trained state uses 20 teacher calls and eight steps of 20 graph-free plus 20 differentiable calls, followed by separately counted fresh reconstruction calls. The single-case runner does not schedule this factorial. The prerequisite one-neural/one-P0 validation has passed without viewing development accuracy.

The implemented launcher runs one GPU job at a time, requires a successful exit and valid `complete.json`, and stops on engineering failures. Before skipping a completed case, it recomputes manifest, source, input and output hashes; a folder or adapter file is insufficient. The same-prefix resume selects only its own verified completions. The global sealer separately audits every discovered attempt under the same frozen protocol and selects the earliest valid completion by completion time, then attempt ID; never select a rerun by accuracy. Partial runs are preserved and restarted from pretrained weights with a fresh prefix because mid-step resume is not implemented. A new prefix runs a new inventory; it does not automatically reuse another prefix's successes. A timeout can leave a stale `scientific_gpu.lock`; remove it only after proving its owner process/container ended and preserving its contents. Do not run an unlimited retry loop or mix scientific-v1 attempts into v2's active case directories.

There is currently no automatic continuation path past a failed case: same-prefix preflight rejects its partial/failure record, and a new prefix starts from the first registered job. The sealer can classify a recorded `FloatingPointError`, but it still requires a valid completed attempt for every run ID; classification alone does not authorize dropping that state. If an arm deterministically fails numerically, stop, preserve evidence and register an explicit failure/termination amendment before accessing scores or changing execution. Such an amendment must specify which unrun cases continue, how terminal failures appear in the full denominator, candidate disqualification and global-seal handling. Do not retry a known deterministic failure, hand-select later jobs to evade the stop, or edit bound code during another active job. This is an unresolved recovery capability, not evidence that a current arm has failed.

The [prospective terminal-failure policy](PROFILED_LORA_TERMINAL_FAILURE_POLICY.md) specifies that conditional amendment, classification audit, unchanged remaining inventory, unavailable metrics and required new evaluator fixtures. It is not implemented or activated; the current batch continues unchanged unless a qualifying failure occurs.

Public attempt evidence belongs in `results/lora_profiled_v1/cases/RUN_ID/ATTEMPT/`; large arrays and fresh caches belong in the corresponding `reports` path. Required outputs include source/manifest/input hashes, original-state hashes, initial/final adapter hashes, eight per-step reports for trained cases, actual call counts, finite gradient checks, timing/memory, native/final endpoint identities, and the completion/failure record. The runner's 1 GiB free-space guard is a minimum, not a 36-case storage estimate. Measure the first complete case's bytes and reserve enough local disk for the remaining cases and both physical backups before batching. Never place pretrained weights or complete restricted datasets in Git.

**Implemented scoring and selection contract**

The rules are frozen separately in [lora_profiled_evaluation_v2.json](../manifests/lora_profiled_evaluation_v2.json). Scientific v2 binds that specification, and [evaluation execution v2](../manifests/lora_profiled_evaluation_execution_v2.json) binds the scorer source and passed fixture evidence. The implemented sealer provides the required global prediction gate; a per-case completion record or registry flag alone cannot release scoring.

Create a global seal with exactly the 36 registered run IDs, one valid attempt per ID, common scientific-registry hash, completion-report hashes, adapter identities, and native/final endpoint hashes. Verify all referenced bytes, model/input identities and complete status. Reject missing/duplicate cases, mixed registrations, partial attempts and changed predictions. Seal all predictions before loading the opposite-source evaluation packet into the scoring process. An unrecoverable case must remain a reported failure under a prospectively amended termination protocol; it cannot disappear from the denominator or yield a qualifying nominee.

Reuse the *mapping and overlap logic* of `score_partner_shape.disjoint_test`, not its historical scorer. For each target, map/deduplicate the opposite packet with the registered native-pixel rule and exclude pixels used by the entire allowed adaptation packet (both spatial folds). Exclusion is determined only by input identity, never candidate error/confidence. Keep the same evaluation observations for every arm. For arm a, scene s, side t and view v, compute `MAE[a,s,t,v] = mean(abs(predicted_final_depth - measured_opposite_depth))` in meters over those fixed points. Average equally over five views, then equally over three scenes, keeping earlier/later directions separate. Define directional gain as `1 - mean_MAE[a,t] / mean_MAE[P1,t]`, not the mean of per-point percentage changes. Fail on empty views or nonfinite predictions; count nonpositive predictions and disqualify affected candidates rather than excluding their pixels.

Only P3 and P4 are candidates for the initial implicit-gradient hypothesis. P0, P1, P2 and P5 are controls. A detached arm winning is useful evidence against the mechanism, not permission to rename it a successful full-gradient method. Nominate at most one candidate with >=1% directional gain in each direction, improvement on at least two of the same three scenes in both directions, and no numerical failures. Maximize the smaller directional gain. Compare gains rounded to 12 decimal places for ties, then lower measured total runtime across its six cases, then smaller numeric arm ID. Always publish all six arms and the P3/P2 and P4/P5 contrasts. Do not use native-only results to select when the final-refined primary result fails.

The registered scorer fixtures cover missing/tampered seals, mixed registrations, overlap exclusion, unequal observation counts, invalid predictions, the two-same-scenes rule, tie handling, endpoint/gauge errors and the filename codec. Re-run them when relevant source changes require new evaluator registration. Publish compact per-view predictions/observations sufficient for independent CPU replay where the dataset terms permit.

For M3, freeze a separate executable dense-evaluation manifest before its scores are read. Use `registered_rgb` as the primary surface reference and `released` as alternate, with the pinned similarity-plus-ICP evaluator and fixed reference masks. For each scene/condition/reference, let A be mean accuracy distance and C mean completeness distance in meters. Normalize each component by the same-condition P0 component and set `surface_score = 0.5*(A/A_P0 + C/C_P0)`; aggregate equally by scene. Compare the nominee against the lowest aggregate primary score among the predeclared eligible same-input controls, minimally P0/P1/P2/P5, with smaller arm ID breaking exact ties. Freeze the eligible pool before dense scoring; add compatible established baselines only after documenting their exact inputs. Historical runs with two-source data or a different final calibration cannot replace these controls. Use that same selected control for per-scene, component and alternate-reference checks; report raw meter-valued distances as well.

The [M3 preparation contract](PROFILED_LORA_M3_PREPARATION.md) resolves the implementation choices before M2 score access: signed-only shifted packets, identical supplied-pose perturbations across three total fresh-process repeats, repetition-zero control selection retained across repeats, fixed-mask validity handling and an explicit full-versus-detached execution-variation test. Implement and register that contract rather than reusing historical two-direction packets or choosing thresholds after evaluation.

The [M3 preparation evidence](../results/lora_profiled_m3_v1/README.md) now supplies signed shifted packets, fixed pose packets, a CPU-to-network input adapter and a pure aggregate gate. Use input plan v2 with attempt `inputs02`; preserve the earlier engineering failures. All 24 input cases passed CPU conversion, and the six centered/clean cases exactly match the original loader. These checks do not release M3 model execution: nomination, execution registration, real endpoint acceptance, prediction sealing and dense scoring still remain.

The M3 manifest must enumerate both temporal directions, centered/shifted grids and clean/noisy supplied-pose conditions: eight gate cells, each with three scenes and matched controls. Apply all rental criteria separately to every cell; do not pool away a failed direction or stress condition. Shifted-source packets and spatial splits need their own verified preparation. Specify camera-noise seeds, masks, grid construction, native evaluator revision and all runtime commands before opening these scores. Initially preserve the established pose-only stress channel; any physical sensor-projection corruption is a separate registered test. Perform three fresh-process paired confirmations with the same optimizer seed and predeclared corruption draws, reporting deterministic execution variation separately from input perturbation. The old manifest is a source of conventions, not authority to reuse unmatched old predictions.

**M3 implementation map and exact remaining scope**

The CPU modules below are implemented and tested. Their existence does not mean an M3 scientific run has occurred. No M3 nominee, execution registration, dense score, or rental authorization exists at this checkpoint.

| Component | Verified capability | What it does not establish |
|---|---|---|
| `prepare_profiled_m3_inputs.py`, input plan v2 / `inputs02` | Signed-source shifted measurements, fixed scene pose matrices, 20 preparation fixtures, all 24 CPU cases | No learned prediction or accuracy result |
| `profiled_m3_network.py` | 18 fixtures; all 24 actual CPU packets convert, six centered/clean cases equal M2; endpoint body matches M2 AST | Mocked endpoint and input equality are not a fresh model endpoint acceptance |
| `profiled_m3_points.py` | 19 synthetic fixtures, also passed in the pinned container; native/final depth/K/camera conversion and fixed-mask validity counters | Does not open, seal or score real study artifacts |
| `profiled_m3_inventory.py` | 18 synthetic fixtures; pure 360-state inventory and serial stage decisions | Descriptors and supplied receipt hashes are not byte verification, executable commands or launch permission |
| `profiled_m3_gate.py` | 23 synthetic fixtures; full-inventory aggregation and frozen thresholds | Cannot accept a partial inventory or authorize label access/rental |
| `check_profiled_m3_dense_runtime.py` | Two fresh CPU processes exactly repeated synthetic Sim(3), ICP, normals and metrics | No actual dense wrapper, universal determinism or study accuracy claim |

The combined 78 M3 unit tests passed. The source-preparation and 24-case tensor checks are separate. Read [the kernel audit](PROFILED_LORA_M3_KERNEL_AUDIT.md) and [runtime comparison02](../results/lora_profiled_m3_v1/runtime_checks/comparison02.json) before implementing the remaining evaluator. Do not edit the hash-bound `PROFILED_LORA_M3_PREPARATION.md`, input plan v2 or its source dependencies; new work belongs in separate files until explicitly versioned registration.

Accepted **CPU dense evaluation** settings are the pinned image, Torch intraop/interops 1/1, OpenCV threads 1, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` all 1 before Python starts. Preserve the existing cKDTree `workers=-1`. The earlier two-thread probes showed approximately machine-precision ICP differences; their reports and exact old source remain archived. The accepted one-thread setting affects this future CPU evaluator only; do not alter registered M2/model training. Never call historical `evaluate_points`, which sets Torch threads to two and uses insufficient cache identities.

Implement the following in order if M2 nominates P3 or P4. Pure synthetic preparation can proceed while M2 runs, but actual study arrays and GPU work remain behind their release conditions:

1. **Dense leaf kernel and camera/boundary adapter.** Preserve the audited scene-wide similarity-plus-rigid-ICP function and exact upstream metric helpers. Use the new point converter; retain transforms, fixed-mask validity, ICP fitness/RMSE/counts, raw meter-valued components, native/final endpoint identities and reference-specific boundary diagnostics. Axial metric-depth error uses unaligned metric depth and the registered-RGB reference's own pinned cameras. Alternate axial error remains unsupported until its distinct camera convention is verified. Test synthetic equivalence and failures described in the kernel audit; do not copy historical caches or per-view alignment.
2. **Byte-verifying registration and controller.** Resolve actual nominee from verified M2 seal and replayed scores. Bind every source, checkpoint, input/split/noise packet, runtime, RGB/reference archive and fixed mask. `build_plan` consumes these identities but cannot verify their truth. Enforce one shared bool mask across both references and all arms, not merely equal point counts or separate per-reference hashes. Make fresh cache identities include all these dependencies. Preserve failures; never create invented receipts to satisfy the pure planner.
3. **GPU endpoint acceptance.** In a separately registered engineering attempt, validate that the M3 centered/clean path produces the expected native/final endpoint under exactly matched M2 settings. Check noisy-prior routing with the declared input packet. Distinguish original supplied approximately rigid matrices from native optimized rigid output rotations; do not orthogonalize the former or silently relax the latter's converter checks. A source/runtime change requires revisiting affected acceptance evidence.
4. **Prospective execution and evaluation registration.** Commit the full inventory, real executable commands, deterministic order, source/runtime hashes, terminal-failure policy, staged prediction-release rules, reuse audits and early-cell predicate before any M3 dense label access. The early predicate may reject an already failed necessary gate; it may never declare an incomplete study successful. A plan with placeholders is not registration. Do not modify M2 scoring or reuse its full-screen release procedure as though it already registered M3's staged protocol.
5. **Execute and verify each stage.** The inventory is 8 cells × 3 scenes × 5 arms × 3 repeats = 360 states, with 720 reference rows and both endpoint metrics per row. Thirty M2 states are only reuse candidates for repetition-zero centered/clean conditions; if every identity audit passes, 330 states remain new. Failed reuse audits require fresh states, not waived identities. There are 23 barriers: first both directional centered/clean repetition-zero cells together, then the six other repetition-zero cells, then eight cells in repetition one and eight in repetition two. Seal every prediction within a stage before its score release. Record a failed gate and mark unexecuted later stages conditional/not run.
6. **Apply the complete gate.** Select the strongest eligible control by primary surface score at repetition zero separately for each cell; keep that control for both later repeats, scene/component checks and alternate reference. Keep the same scene corruption matrices and optimizer seed42 across all repeats. Repetitions measure execution variation, not independent initialization or noise diversity. Require the frozen full-versus-detached mechanism inequality as well as surface efficacy. Only the full 360-state verified inventory can pass the complete gate. Then consider M4/M5 and the conditional rental reserve.

The full-versus-detached contrast for P3 uses P2; for P4 it uses P5. They must remain forward-matched in calibration and endpoint. For normalized primary surface score S (lower is better), the preparation contract requires a positive full-gradient benefit in every repeat and a repetition-zero benefit larger than the sum of the two arms' observed execution variations plus 1e-6. This is a mechanism check on this design, not a statistical significance claim or a guarantee of publication.

**Decision table after the registered M2 verdict**

| Observed outcome | Required next action | Forbidden inference |
|---|---|---|
| No eligible P3/P4 nominee | Archive/replay the screen; diagnose with the surrogate diagnosis guide; propose one measurable local redesign with a forward-matched control; register before running | Smaller training loss or one good scene cannot release M3/rental funds |
| Eligible nominee | Finish the M3 infrastructure and prospective registration above; then execute serially | M2 held-source MAE is development evidence, not final reconstruction/generalization proof |
| Engineering/infrastructure failure | Preserve logs and partial artifacts; repair and version affected engineering components before resuming | Do not count a broken implementation as a failed mathematical hypothesis or conceal it as a successful state |
| Sound efficacy, repeatability or mechanism failure | Use the same one-redesign allowance shared across all failure branches; preserve the negative result | A second failure branch does not grant another redesign or a new rental budget |
| Registered redesign fails | Complete independent replay, backups, public evidence and final CI; issue the bounded negative verdict, explicitly marking later conditional stages not run | The broader ambition of improving accuracy remains unmet |
| All development gates pass | Freeze new scenes/method before M5; measure independent generalization and conditional transfer within caps | Development gates alone cannot support an unseen-scene or state-of-the-art claim |

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

For the secondary axial-depth score, convert the saved native depth to meters with the same sealed adaptation-only global g0: `native_metric_depth = float64(native_depth) * g0`. Do not multiply by the five fitted relative scales. Raw native depth can have an arbitrary gauge and must not be labeled a meter-valued prediction. The scorer checks positive finite g0/scales and the full saved relationship `final_depth = scales * g0 * native_depth` before comparison. Evaluation rules v2 preserve v1 and clarify this unit convention; the primary endpoint and all advancement thresholds are unchanged.

**Narrow chronology erratum:** the v2 clarification and filename-verification repair followed completion of the subsequently retired scientific-v1 P0. They preceded every development-accuracy score and the fresh scientific-v2 screen. Any preserved manifest reason describing the change as before all scientific execution is too broad: it was before v2 execution and before scoring, not before that retired P0. Keep the immutable manifest bytes and this correction together; do not edit a bound manifest during execution. All 36 states are run fresh under scientific v2, and the retired v1 P0 is excluded from that inventory.

**Decisions and reporting**

The exact six-arm design, 1% nomination gate, 5% surface rental gate, tie rule, repeatability requirement and bounded redesign are in the goal contract. Use their registered scorer implementation and passed fixtures. Do not choose the best endpoint after seeing scores. Report the full-versus-detached contrasts even when another baseline gives a larger apparent improvement.

If the full-gradient arm ties its detached control within repeat variation, the proposed mechanism is unsupported. If the surrogate improves while the native surface worsens, investigate solver absorption and the reconstruction mismatch. If engineering checks fail, fix implementation before interpreting accuracy. If a sound method fails, retain that evidence and register the single permitted targeted local redesign. Do not automatically spend the transfer reserve after failure.

Use the [conditional surrogate diagnosis guide](PROFILED_LORA_SURROGATE_DIAGNOSIS.md) only after the M2 seal and verdict. It distinguishes small functional LoRA changes, calibration absorbing the update, and useful surrogate changes lost during native reconstruction. Its exact derivative includes the residual-response term and does not treat a ridge matrix as an orthogonal projector. These are diagnostic options, not additional authorized redesigns or evidence of an accuracy gain.

For each completed attempt, retain the execution commit, source/config hashes, permitted packet and split hashes, initialization, optimizer steps, gradient diagnostics, predictions, evaluator revision, runtime, peak memory and cost. Keep large arrays in excluded reports and copy the full archive to the second disk. Publish compact data sufficient for the claimed CPU replay, subject to dataset/code licenses. Verify both backup hashes before treating evidence as durable.

**Budget and stopping rules**

Initial work is local and adds USD 0 rental compute. The existing conditional USD 25 is a total ceiling, not a balance to use up: USD 8 for transfer feasibility, USD 12 for one confirmation batch, USD 5 for contingency/storage. It does not add funds to the preferred USD 200 project ceiling. Check actual balance, live quote, ledger and retained storage before any paid launch. Existing credit observations are stale and private.

Prepare data, wheels, code and manifests locally. Start a single paid job stream only after its evidence gate, with the tested provider-side deadline and heartbeat/lease. A shell timeout does not stop billing. On success, failure or weekly reserve threshold, preserve outputs, stop the paid pod and verify provider state. The retained workspace `749tvrn5gwvuel` and its 20 GB volume must not be deleted. Do not request deletion again without a material change.

**First actions for the successor**

The immutable M1 backup is verified on a second physical disk: `E:/ResearchData/TCO/research_results/lora_profiled_v1/snapshots/m1-20261118-verified01`. Its [public receipt](../results/lora_profiled_v1/backups/m1-20261118-verified01.json) records 136 files and 315,628,130 bytes. Verify it with `python scripts/archive_profiled_evidence.py verify --destination E:/ResearchData/TCO/research_results/lora_profiled_v1 --snapshot m1-20261118-verified01 --check-source`. Do not rerun copy with that existing snapshot ID. After the complete M2 prediction seal exists, use a fresh snapshot ID and `copy --seal results/lora_profiled_v1/prediction_seal.json` with the same destination, then `verify --check-source`; require successful exits from both commands. The archiver never decodes predictions, overwrites snapshots or copies mutable progress/logs. Failed copies remain preserved for diagnosis.

Read the live progress file and inspect the active full `screen02` operation. Replay preparation only when files/runtime changed or their validity is uncertain. Both production bridges, the first P0/P1 validation and the scientific/scorer v2 registrations are complete. Observe the current queue; use the continuation block only after an interruption has been classified and no owner remains alive. After all 36 states verify, create/verify the global seal, then score and replay. Preserve the historical `gradient01` and retired v1 P0 evidence. After each meaningful result, update evidence paths, job state, next command, run counts, costs and Git/CI state. No scientific NPZ or held-source score was opened while updating this handoff.

**Copyable instruction for a successor**

> Continue the existing active research goal in `C:/SVNProjects/Research/tco-research`. First read `docs/CALIBRATION_AWARE_LORA_HANDOFF.md`, the goal contract, mathematical specification and `results/lora_profiled_v1/progress.json`. Verify actual live process state before acting; do not start a duplicate queue or create a replacement goal. Follow the frozen v2 M2 protocol through all 36 predictions, global sealing, registered scoring and score replay. No accuracy improvement is established yet. Continue conditional M3 or the single registered local redesign according to the verified outcome. Keep original model weights frozen and train only LoRA. Public datasets only; local preparation and development add no rental compute. The existing USD 25 reserve is conditional and cumulative, not a fresh allocation. Preserve the retained RunPod workspace and unrelated local edits. At 5% weekly allowance remaining, checkpoint, stop and verify any paid research pod, then pause under the standing instruction. Commit permitted code/results, verify backups and CI, and report completed evidence separately from planned work. A plan, partial inventory or passing synthetic test does not complete this goal.
