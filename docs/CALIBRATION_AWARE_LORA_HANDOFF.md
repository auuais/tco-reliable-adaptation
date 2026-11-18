# Execution handoff for calibration aware LoRA research

This is the operational guide for a successor agent, including Solus. Read it with the [goal contract](CALIBRATION_AWARE_LORA_GOAL.md), [mathematical specification](FROZEN_LORA_ACCURACY_PLAN.md) and [design settings](../manifests/lora_profiled_design_v1.json). The contract defines the objective and scientific gates. This guide explains how to implement, run and verify them in the actual workspace. Use files and live process state to establish the current checkpoint before taking action.

**Objective and current checkpoint**

The objective is to determine whether differentiating calibration while training only LoRA improves reconstructed geometry. MASt3R is the first model because its existing decoder adaptation fits this machine. VGGT transfer is conditional on a useful, repeatable MASt3R result. The original encoder, decoder and task-head parameters remain frozen; only newly installed LoRA factors are trainable.

At this handoff, the mathematical solver, input preparation and local environment checks exist. The real-model gradient bridge, complete experimental runner, final calibration integration and new accuracy experiments remain to implement. A passed CPU derivative check establishes the solver calculus. It does not establish propagation into real LoRA weights or an accuracy improvement. Consult [progress.json](../results/lora_profiled_v1/progress.json) for milestone state and the next action. Do not inherit completion claims from an older handoff.

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

Model jobs add `--gpus all` and the read-only checkpoint mount `--mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly`. The image already sets the MASt3R import path. Do not start a second GPU experiment while the first is active. The commands for the future model runner must be written and tested when that runner exists; do not infer that a planned filename is executable now.

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

**Implementation order for the remaining model code**

The following filenames are proposed responsibilities, not claims of existing working runners. Confirm filesystem state before creating them.

1. Implement `scripts/profiled_network.py`. Reuse image preprocessing and frozen encoder features. Collect first-view pointmaps for every directed pair. For target i, fuse its four first-view outputs using pretrained confidence through `canonical_depth`. Use `canon[...,2]`, never `canon2` or second-view points in another camera frame. Prepare separate frozen geometric states for each spatial support fold at the pretrained model.
2. Implement `scripts/check_profiled_lora_bridge.py`. First compare zero-LoRA outputs with pristine outputs at identical precision and cached features. Check canonical fusion against pinned upstream code. Build the real five-view loss with two varying pair outputs and the other outputs fixed; compare retained-graph gradients with exact output-adjoint replay. Then perform a numerical adapter-direction check and one full 20-pair macro-step. Record actual module gradients, original state hashes, memory and calls.
3. Implement `scripts/run_profiled_lora.py`. Load one case and one arm, verify immutable inputs and the execution manifest, reset adapters/optimizer, run eight steps, capture adapters and all traces, then perform fresh final inference/matching. Use a unique attempt directory and refuse to overwrite an existing result or failure. Add a prepare/dry-run mode that enumerates all 36 states without training. Commit that runner and its frozen execution manifest before the scientific batch.
4. Implement `scripts/profiled_final_calibration.py`. Preserve both the native KDP output and the final five-scale refinement, with exactly the frame/gauge convention below. Provide a zero-adapter comparison for both endpoints.
5. Implement `scripts/score_profiled_lora.py`. Require a global prediction seal before reading opposite-source scores. Recompute the primary metric with explicit sample masks and counts; report all arms, per-scene values, both directions, failures and the deterministic nomination rule. Dense evaluation follows the existing registered conventions and matched input controls.
6. Add compact replay/CI entry points as meaningful evidence becomes available. Keep numerical fixtures separate from GPU-dependent checks. Publish all planned runs, including failures, and record why conditional experiments were not run.

**Exact gradient replay algorithm**

Keep one adapter state fixed for an entire macro-step. In the first pass, run each of the 20 directed pairs without retaining decoder activations. Save the point tensors needed by the selected objective. Turn their detached copies into differentiable leaves, fuse target depths, solve calibration and compute the loss. Differentiate the loss with respect to these leaves to obtain output adjoints. Recompute each pair at identical parameters, precision, random state and buffers, then backpropagate its saved adjoint. Accumulate all pair contributions. Only then clip the full gradient and take one optimizer step.

No optimizer step may occur between replayed pairs. Do not retain 20 full decoder graphs just because the output tensors fit. Freeze confidence/matching selection in the training objective and maintain the same decoder-call schedule in all neural arms. Features can be cached because the encoder is frozen. P1 reuses its two output branches and the arithmetic mean over `schedule(20)`; repeated use of a pair contributes summed adjoints. P2–P5 use the fused self-depth objective. One factor's initial zero gradient is expected when the opposite LoRA factor is zero; the active factor must receive finite useful gradients.

Reduced precision can make tiny finite differences meaningless. Run the numerical direction check with an fp32 decoder and TF32 disabled if necessary, with fixed features and several prespecified perturbation sizes. Record arithmetic settings and tolerances before opening accuracy results. A gradient check must cover the implementation actually claimed; a diagnostic of a different precision path cannot prove every reduced-precision detail.

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

Every comparator gets the same final intervention. A baseline without final calibration cannot isolate the neural objective. Both endpoint outputs and the evaluator's actual input hashes must be saved.

**Decisions and reporting**

The exact six-arm design, 1% nomination gate, 5% surface rental gate, tie rule, repeatability requirement and bounded redesign are in the goal contract. Implement those rules once in a scorer and verify them on a small meaningful fixture. Do not choose the best endpoint after seeing scores. Report the full-versus-detached contrasts even when another baseline gives a larger apparent improvement.

If the full-gradient arm ties its detached control within repeat variation, the proposed mechanism is unsupported. If the surrogate improves while the native surface worsens, investigate solver absorption and the reconstruction mismatch. If engineering checks fail, fix implementation before interpreting accuracy. If a sound method fails, retain that evidence and register the single permitted targeted local redesign. Do not automatically spend the transfer reserve after failure.

For each completed attempt, retain the execution commit, source/config hashes, permitted packet and split hashes, initialization, optimizer steps, gradient diagnostics, predictions, evaluator revision, runtime, peak memory and cost. Keep large arrays in excluded reports and copy the full archive to the second disk. Publish compact data sufficient for the claimed CPU replay, subject to dataset/code licenses. Verify both backup hashes before treating evidence as durable.

**Budget and stopping rules**

Initial work is local and adds USD 0 rental compute. The existing conditional USD 25 is a total ceiling, not a balance to use up: USD 8 for transfer feasibility, USD 12 for one confirmation batch, USD 5 for contingency/storage. It does not add funds to the preferred USD 200 project ceiling. Check actual balance, live quote, ledger and retained storage before any paid launch. Existing credit observations are stale and private.

Prepare data, wheels, code and manifests locally. Start a single paid job stream only after its evidence gate, with the tested provider-side deadline and heartbeat/lease. A shell timeout does not stop billing. On success, failure or weekly reserve threshold, preserve outputs, stop the paid pod and verify provider state. The retained workspace `749tvrn5gwvuel` and its 20 GB volume must not be deleted. Do not request deletion again without a material change.

**First actions for the successor**

Read the live progress file, inspect actual jobs and replay the existing input/solver checks. Read `profiled_calibration.py` and its tests. Complete the production LoRA bridge and freeze the model-run manifest. Measure one full local macro-step before scheduling the factorial. Keep this handoff and progress checkpoint current as actual milestones finish. The next agent should be able to identify the last verified artifact, any running handle and the exact next safe command without reconstructing this conversation.
