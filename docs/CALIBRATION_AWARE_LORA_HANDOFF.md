# Execution handoff for calibration aware LoRA research

This is the operational guide for a successor agent, including Solus. Read it with the [goal contract](CALIBRATION_AWARE_LORA_GOAL.md), [mathematical specification](FROZEN_LORA_ACCURACY_PLAN.md) and [design settings](../manifests/lora_profiled_design_v1.json). The contract defines the objective and scientific gates. This guide explains how to implement, run and verify them in the actual workspace. Use files and live process state to establish the current checkpoint before taking action.

**Objective and current checkpoint**

**Operational checkpoint, November 18, 2026:** the scientific-v2 `screen02` queue completed all 36 states successfully. Invocation `20261118T104117Z_ede54062` and observation session `27750` are terminal history, not work to resume. The global prediction seal passed before scoring. Registered score replay and independent compact arithmetic replay both confirmed **no eligible nominee**. No research GPU container remained running after the queue. Read [progress.json](../results/lora_profiled_v1/progress.json) and actual process state on resume; do not restart the completed screen.

Scientific registration is [lora_profiled_execution_v2.json](../manifests/lora_profiled_execution_v2.json); scorer registration is [lora_profiled_evaluation_execution_v2.json](../manifests/lora_profiled_evaluation_execution_v2.json). Preserve both and all completed artifacts. P3 gained only +0.1112% over P1 in the earlier direction and -0.0575% in the later direction. P4 gained -0.3637% and -0.1436%. Both failed the frozen >=1% gain requirement in each direction; the full/detached contrasts also reverse sign between directions. See the [complete result table](../results/lora_profiled_v1/README.md), [scores](../results/lora_profiled_v1/held_source_scores.json) and [independent replay](../results/lora_profiled_v1/independent_public_replay.json). M0/M1/M2 are scientifically complete; whole-milestone progress is 50%; publication/CI state is recorded in the live checkpoint. No qualifying accuracy improvement or rental gate has passed. Rental compute for this goal remains USD 0.

**Successor's immediate task:** follow the negative-screen branch. Verify the final archive/publication checkpoint, then implement and register the bounded, zero-update local diagnostic described below before reading additional cache/adaptor arrays for that diagnosis. Determine whether effective LoRA changes are tiny, dominated by removable scale, or lost downstream before selecting the one permitted scientific redesign. The detailed M3 infrastructure is retained as conditional work; do not finish or launch its 360-state study solely because the code exists. No paid transfer or parameter sweep is released.

The [D0 diagnostic design](../manifests/lora_profiled_diagnosis_design_v1.json) fixes a CPU-only saved-artifact probe: effective BA weight changes for all 30 adapters, and raw forward-output changes for the first registered scene/source (Heads/earlier) with P0/P1/P2/P3. A source-verified P0/later raw-forward cache may provide a same-RGB/model comparator; its different native alignment is not a repeat control. The fixed native grid begins at (8,8), stride 16; permitted adaptation coordinates are reported separately. Scalar-versus-shape decomposition uses model outputs only and cannot alter accuracy predictions. No target-depth/reference arrays, opposite-source measurements, optimizer update or new model forward is allowed in D0. The design is not executable registration: implement/test its loader/math, bind actual sources/runtime/artifact identities and commit that registration before decoding additional diagnostic tensors. This consumes zero scientific redesigns; the single redesign must follow evidence.

`gradient02`, under engineering v2 at commit `8c3de74`, passed exact retained/replay agreement, all three finite-difference perturbations, original-state integrity and a complete 20-pair macro-step (23.85 seconds; 4825.71 MiB peak allocated). `endpoint01` passed exact pristine/zero-LoRA native and final endpoint equality, matching/confidence equality and unchanged original/adapter states. Preserve these reports and verify their registered hashes; do not repeat them without a relevant source/runtime change. See [the engineering diagnosis](PROFILED_LORA_ENGINEERING_DIAGNOSIS.md), [gradient evidence](../results/lora_profiled_v1/gradient_bridge.json) and [endpoint evidence](../results/lora_profiled_v1/endpoint_bridge.json).

The objective is to determine whether differentiating calibration while training only LoRA improves reconstructed geometry. MASt3R is the first model because its existing decoder adaptation fits this machine. VGGT transfer is conditional on a useful, repeatable MASt3R result. The original encoder, decoder and task-head parameters remain frozen; only newly installed LoRA factors are trainable.

**Preserved history, not pending work:** `gradient01` at commit `8e33ba3839338daf9944939834c2834ee4f9d2d4` passed zero-adapter prediction equality but failed retained/replayed derivatives: relative L2 `0.0058830234318835144` and cosine `0.9999827114918957`, against unchanged limits `1e-4` and `0.99999`. Strict deterministic arithmetic resolved the discrepancy. See its [failed comparison](../results/lora_profiled_v1/engineering/gradient01/replay_failure.json) and [traceback](../results/lora_profiled_v1/engineering/gradient01/failure.json). The scientific-v1 `screen01` P0 subsequently completed in 89.29 seconds with exit code 0 and unchanged original weights, but its host verifier rejected five Linux colon/Windows U+F03A cache names. The narrow filename codec now verifies all original bytes. That unscored case is archived on C: and E: with [its archive receipt](../results/lora_profiled_v1/archived_cases/scientific_v1_heads_earlier_P0_screen01.json); it does not count toward scientific v2. Neither historical failure should be rerun under its old attempt name.

Read in this order: this operational checkpoint, the goal contract, the mathematical specification, the design and bridge-threshold manifests, then the referenced implementation. User instructions take precedence. The contract defines the scientific decisions; the manifests freeze settings; actual reports establish completion. A conflict between prose, settings and code is an engineering issue to resolve explicitly before execution. Never silently substitute a setting or promote an old successful report across changed source hashes.

The user has authorized research execution, Git commits and publication of permitted code/results. Continue useful local work without repeatedly asking permission. The user allows best judgment for routine preferences unanswered after ten minutes. This does not waive monetary caps, access requirements or an explicit prohibition. Do not pause for a model switch unless the user requests it. The active goal remains active until its completion, blocking or user-requested pause conditions are met.

**Reading order and current work board**

This handoff lets a successor execute the research without reconstructing the conversation. The goal is an evaluated accuracy improvement, or an honestly bounded negative investigation; writing another plan is not scientific completion. Original parameters and buffers remain frozen. Only LoRA factors may be trained after a prospective protocol is released.

Read this checkpoint and immediate D0 work package first, followed by [live progress](../results/lora_profiled_v1/progress.json), the [completed result](../results/lora_profiled_v1/README.md) and the [goal contract](CALIBRATION_AWARE_LORA_GOAL.md). Then read the [D0 design](../manifests/lora_profiled_diagnosis_design_v1.json) and [diagnosis mathematics](PROFILED_LORA_SURROGATE_DIAGNOSIS.md). The full [mathematical plan](FROZEN_LORA_ACCURACY_PLAN.md) supplies background; the long M3 appendix is conditional infrastructure.

The native goal's final sentence, "Start with M0", records its original launch instruction. M0/M1/M2 are now complete. Do not replace the active goal, reset its budget, or restart a completed milestone because an older summary was not rewritten. The user's current instructions govern; immutable registrations govern their experiments. If code, JSON and prose disagree, reconcile the actual evidence before taking the affected action. Never rewrite a sealed record to make it agree with newer prose.

| Work class | Current action |
|---|---|
| Do now | Verify the publication checkpoint; implement, test, register and run D0 |
| Already complete | M0; M1 real-model bridges; 36 M2 states; global seal; official and independent compact score replay |
| Conditional after diagnosis | One focused no-update D1 if needed; at most one scientific redesign with a new frozen protocol |
| Conditional after a qualifying method | Dense surface confirmation, mechanism/repeatability checks, new scenes and VGGT transfer |
| Do not launch | Completed M2 queue; old failed/retired attempts; original no-nominee M3 study; ungated rentals |

**Immediate work package D0**

D0 is a CPU-only diagnosis of saved artifacts, with no optimizer, no new model forward and no rental allocation. It does not make another accuracy comparison. The design is fixed but `execution_ready` is false: the code and executable registration below still have to be created. These proposed filenames and interfaces are an implementation specification, not tools that already exist.

| Deliverable to implement | Responsibility |
|---|---|
| `scripts/profiled_d0_math.py` | Pure float64 effective-weight norms, fixed-coordinate changes and scalar/shape decomposition; no I/O or model access |
| `scripts/profiled_d0_artifacts.py` | Resolve selected attempts from the global seal, verify bytes/field mappings, and stream CPU tensors |
| `scripts/run_profiled_d0.py` | Separate prepare, verify-registration, run and verify modes; exclusive outputs; measured resources; no training path |
| `scripts/test_profiled_d0_math.py`, `scripts/test_profiled_d0_artifacts.py` | Synthetic numerical and malformed-artifact tests; no real diagnostic tensors |
| `manifests/lora_profiled_diagnosis_execution_v1.json` | Prospective code/runtime/input/output contract committed before diagnostic decoding |
| `results/lora_profiled_d0_v1/source_layout.json` | Source-only writer/reader audit and exact field/frame/shape mapping |
| `results/lora_profiled_d0_v1/attempts/ATTEMPT/report.json` | All declared statistics, provenance, failures and resource counts |
| `results/lora_profiled_d0_v1/README.md` | Evidence table, interpretation limits and one explicit next decision |

Implement in this order:

1. **Audit sources without loading tensors.** Read the sealed runner/capture/LoRA writers and JSON inventories. Resolve the selected attempt from the global seal, never the newest directory. Source bytes must match the recorded revision. Reuse `profiled_artifacts.resolve_logical` for Windows filename handling. Do not rename evidence or infer missing arrays from another endpoint.
2. **Implement pure mathematics and fixtures.** For each adapted matrix compute `DeltaW=(alpha/r) B_L A_L`, alpha=1, r=4. Use `||B_L A_L||_F^2=tr((B_L^T B_L)(A_L A_L^T))`; validate against direct products. Divide by the corresponding frozen matrix norm. Aggregate with the square root of the sum of squared norms, over adapted matrices only. A zero denominator is undefined. Initial zero B implies zero initial effective update; factor norms are not a functional distance.
3. **Implement fixed-input comparisons.** Use exactly Heads/earlier P0/P1/P2/P3, all 20 directed pairs, the declared 768-pixel grid, and a separate table for allowed earlier adaptation coordinates. Retain per-pair statistics. With `delta=d1-d0`, compute `a=(d0^T delta)/(d0^T d0)` and `delta_shape=delta-a*d0`. Report total/scalar/remaining-shape change and fixed quantiles. This is an unlabeled diagnostic, not a recalibration of scored predictions. Keep invalid counts and mark undefined statistics; never shrink the coordinate set by candidate validity.
4. **Check the optional same-state comparator.** Heads/later P0 is allowed only for raw forward tensors proven to use identical RGB/model/runtime before prior-dependent optimization. Its native endpoint is not a repeat control. If equivalence cannot be established, record that the comparator is unavailable. One such comparison is not a general uncertainty estimate.
5. **Freeze executable registration.** Bind design and trigger score/seal/replay hashes; selected attempts and exact allowed file hashes; code/writer/runtime identities; checkpoint and current config; factor/field/frame mappings; coordinate members; formulas, aggregation and tolerances; failure policy; resource limits; exact commands and output roots. Preparation hashes bytes without deserializing tensors. Current config identity is prospective, not historical M2 proof. Keep the design unchanged; readiness belongs to the new verified registration. Commit before `run` decodes tensors.
6. **Execute once with a fresh attempt ID.** Verify bindings before decoding; map tensors to CPU and stream files. Honor the 30-minute wall cap and single-job limit, with elapsed checks between artifacts plus an outer process/container deadline. Preserve partial outputs on failure/timeout. Recheck inputs afterward. Record actual host memory/time, zero model forwards, zero optimizer updates and zero rental use.
7. **Interpret and preserve.** Independently recompute a representative statistic, archive source/report/bindings on C: and E:, add CPU fixtures to CI, and update progress and next command. Large caches/checkpoints stay out of Git. D0 does not consume a scientific redesign or establish accuracy.

Required synthetic checks: rectangular factor orientation; zero B; wrong shapes; NaN/Inf; non-unit alpha/r; invariance under `A_L -> c A_L, B_L -> B_L/c`; Gram/direct-product agreement; zero-denominator handling; exact scalar-only, orthogonal and mixed output changes; reconstruction identity; grid size/bounds and (x,y) indexing; invalid values retained; equal-pair aggregation; missing/extra/changed files; wrong source/checkpoint hashes; first-camera versus second-branch selection; denied access to measurement/reference arrays; and refusal to overwrite an attempt. Set numeric tolerances from fixtures before real decoding.

**Source map for D0 without broad loaders**

The source-only audit establishes the intended schema; runtime dtype/shape/finiteness remains to be checked during registered decoding.

| Item | Source and interpretation |
|---|---|
| Final adapter | Global seal `cases[*].selected.adapter.file` and SHA; writer `run_profiled_lora.py` saves named `.lora_A`/`.lora_B` parameters |
| Factor orientation | `third_party/TCO/peft/lora.py`: A `[4,in_features]`, B `[out_features,4]`; expect 144 modules and 288 arrays |
| Base matrix | Original checkpoint `MODULE.weight`; wrapped runtime name `MODULE.original_layer.weight`; verify exact key/shape, never infer a transpose |
| Cache identity | Selected completion and registry determine raw attempt root; `selected.raw_files` binds logical file names and digests |
| Pair path | `endpoint_cache/forward/md5(instance_i)/md5(instance_j).pth`; `sparse_ga.py` writes `(X_ii,C_ii,X_ji,C_ji)` |
| Self-camera depth | `tuple[0][...,2]`, expected `[384,512]`; XYZ `[384,512,3]`, confidence `[384,512]`. `tuple[2]` is partner points in the first camera and is not the partner's self-camera depth |
| Instance names | `mast3r_network_capture.py`: `f'{key}_{meta["sha256"][:12]}_{i}'`, with `key=scene+"_base"`, i=0..4. The hash is the complete input archive SHA, available from JSON metadata; it is not a decoded RGB-only digest |
| Coordinates | Only five authorized `heads_earlier_FRAME__xy` members of hash-bound `splits.npz`, plus the fixed image grid |
| P0 comparator | `sparse_global_alignment` calls `forward_mast3r` before native optimization; the prior hook changes `sparse_scene_optimizer`, not that raw forward |

Do not call `profiled_network.load_case` or `views_for` to get names/coordinates: they decode broader measurement/RGB archives, and RGB archives also contain reference fields. Load only declared NPZ members with `allow_pickle=False`, and hash-verified forward tuples with `torch.load(map_location='cpu', weights_only=True)`. Stream selected frozen checkpoint matrices without constructing MASt3R. D0 does not authorize canonical/native/final tensor inspection merely because those files were sealed.

Use these CLI semantics when implementing the runner:

```text
prepare              JSON/source/byte inventory; no tensor decode; exclusive registration
verify-registration  Verify bound bytes/runtime; no tensor decode
run                  Verify; read only declared tensors; write fresh report/receipt
verify               Recompute statistics into a fresh verification receipt
```

These are not shell commands yet. When the entry point exists, place exact working commands in its registration and the live progress file. Never substitute a placeholder command or invented receipt for release.

**From diagnosis to the one redesigned method**

Create an evidence table answering: actual effective weight change; raw functional change versus any valid same-state comparator; scalar versus shape component; evidence for calibrated-response cancellation; and unresolved downstream mismatch. For each, cite an artifact, mark supported/unsupported/unresolved, and name the smallest discriminating test. A large scalar secant does not prove the solver cancelled an actual parameter update.

If D0 cannot discriminate causes, register one focused D1 no-update probe first. Fix its parameter state, inference precision, tensor access, perturbations, tolerances and maximum calls before execution. Use the same selected scene/source and only its allowed adaptation packet. Choose one question: an exact solver-response check at a fixed state, or a paired replay isolating one training-to-endpoint operation. D1 has no optimizer update and zero rental spend. Cap it prospectively at 60 minutes of local model execution plus 30 minutes CPU analysis, one job at a time. If that bound cannot resolve the cause, report inconclusive evidence rather than silently expanding the search.

Preserve the exact calibration derivative. At fixed weights, define `H=A^T A+lambda I`, `r=A s-b`, `e=Q s-d`, `t_S=dot(A) s`, `t_Q=dot(Q) s`:

```text
dot(s) = -solve(H, A^T t_S + dot(A)^T r)
dot(e) = t_Q + Q dot(s)
outer directional derivative = e^T dot(e)
detached comparator at the SAME parameter state = e^T t_Q
```

The residual-response term is required. Ridge response `I-A H^-1 A^T` is generally not an orthogonal projector. Different final P3/P2 states do not isolate this local derivative. A network direction needs a real parameter derivative or registered parameter finite differences; retain the output-space label for an output-space probe.

| Supported diagnosis | Candidate for the single redesign | Required control |
|---|---|---|
| Software/precision/gradient fault | Engineering repair before scientific interpretation | Zero-adapter equality, frozen state and numerical derivatives; do not call a repair a scientific contribution |
| Direct change is cancelled by fitted calibration | A damped update in directions that preserve predictive geometry after calibration, using the full query response | Ordinary damping on the same objective at matched compute; full/detached response; exact derivatives/gauges |
| Useful surrogate change is lost at a specific native operation | Differentiate a fixed declared number of refinement steps, with fixed matches and explicit gauge | Forward-identical detached refinement, matched D and original profiled method; both endpoints |
| Positive effect exists but overshoots | Measured one-dimensional update sizing | Full/half steps and ordinary backtracking including extra forward cost |
| Coherent source-calibration error dominates | Correlated observation-noise model | Diagonal covariance with identical marginals and allowed inputs |
| Tiny changes or no identified cause | Resolve the measured numerical/gradient boundary within D1, or report inconclusive/negative evidence | No automatic learning-rate/rank/step sweep |

Choose one row only when its measurable prediction is supported. These options are conditional, not a menu to execute. Their mathematical tools are established; the contribution needs a specific mechanism and independent reconstruction gains.

Before redesigned training, write a versioned method specification and execution/evaluation manifests covering the observed failure, falsifiable remedy, exact equations, coordinate/gauge conventions, single changed mechanism, adapter placement/rank, optimizer/precision, matched controls, full scene/source inventory, common final endpoint, selection/failure/stopping rules, storage/runtime and cost. Use both directions of all three observed development scenes. Retain the 1% nomination and 5% surface gates. Explicitly name new eligible candidates and controls; do not apply the original P3/P4-only selector to unnamed replacement arms.

Default to rank 4, alpha 1, eight macro-steps and 20-pair accumulation unless the hypothesis specifically justifies another prospectively fixed schedule. Every comparator must share changed schedule and permitted information. A differentiated-refinement claim needs a forward-identical detached control; other mechanisms need the table's matched control. Keep P0, matched D and the original failed mechanism visible, with fresh runs unless a complete reuse audit proves equivalence. Historical config identity remains unproven; do not inherit an old success flag or M3 acceptance. Set `redesigns_executed=1` when this scientific redesign starts. Engineering retries remain visible and do not authorize a second method.

A qualifying redesigned method requires its own updated, registered confirmation path. Reuse audited M3 components only after dependency/acceptance checks; preserve the original no-nominee result. If the redesign fails, complete the negative evidence package and mark later stages not run. The broader accuracy-improvement ambition remains unmet.

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
docker --host npipe:////./pipe/dockerDesktopLinuxEngine ps --format '{{.Names}} {{.Status}} {{.Image}}'
docker --host npipe:////./pipe/dockerDesktopLinuxEngine image inspect tco-mast3r:pilot --format '{{.Id}}'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
Get-Content results/lora_profiled_v1/progress.json
```

Read the native active-goal state and weekly usage counter using the available tools. If a recorded local job might still be running, inspect that exact container with `docker --host npipe:////./pipe/dockerDesktopLinuxEngine inspect NAME --format '{{json .State}}'` and its logs with the same explicit host. A stale status file or observation timeout does not prove the job stopped. Reattach or poll the live handle; never start a duplicate merely because a previous observation timed out.

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
docker --host npipe:////./pipe/dockerDesktopLinuxEngine run --name $profiledAttempt --mount type=bind,source=C:/SVNProjects/Research/tco-research,target=/workspace --workdir /workspace sha256:74783a8eefee6daad6ff2714a51263ccdcdc5092b83cebee20515310bec46b50 python scripts/check_profiled_calibration.py --verify
```

Model jobs add `--gpus all`, `--env CUBLAS_WORKSPACE_CONFIG=:4096:8`, and the read-only checkpoint mount `--mount type=bind,source=E:/ResearchData/TCO/checkpoints/MASt3R/hf,target=/weights,readonly`. The image already sets the MASt3R import path. The completed registered batch launcher supplies these arguments and pins the local Docker engine. Never run two research GPU experiments concurrently. The v2 scientific queue has finished and must not be relaunched as a new study. CPU-only probes use names beginning `tco-m3-`, no `--gpus`, and `CUDA_VISIBLE_DEVICES=`; the original M2 launcher treats `tco-profiled-` containers as possible competing jobs.

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
3. Completed: all 36 `screen02` states, successful queue terminal record and no numerical failures. Preserve the finished queue; do not repeat it.
4. Completed: global seal prepared/verified before score access, official score/replay and independent compact arithmetic replay. No P3/P4 candidate qualified. Smaller training loss or an isolated good scene cannot substitute for the gate.
5. Current branch: diagnose failure locally and use at most the one targeted scientific redesign allowed by the contract. A redesigned method needs prospective registration and its own confirmation; it cannot inherit a nonexistent P3/P4 nomination. No rental allocation is released by this failed gate.
6. Publish planned runs including failures, verified backups, compact replay evidence and CI results; record why conditional stages did not run.

**Exact gradient replay algorithm**

Keep one adapter state fixed for an entire macro-step. In the first pass, run each of the 20 directed pairs without retaining decoder activations. Save the point tensors needed by the selected objective. Turn their detached copies into differentiable leaves, fuse target depths, solve calibration and compute the loss. Differentiate the loss with respect to these leaves to obtain output adjoints. Recompute each pair at identical parameters, precision, random state and buffers, then backpropagate its saved adjoint. Accumulate all pair contributions. Only then clip the full gradient and take one optimizer step.

No optimizer step may occur between replayed pairs. Do not retain 20 full decoder graphs just because the output tensors fit. Freeze confidence/matching selection in the training objective and maintain the same decoder-call schedule in all neural arms. Features can be cached because the encoder is frozen. P1 reuses its two output branches and the arithmetic mean over `schedule(20)`; repeated use of a pair contributes summed adjoints. P2–P5 use the fused self-depth objective. One factor's initial zero gradient is expected when the opposite LoRA factor is zero; the active factor must receive finite useful gradients.

Reduced precision can make tiny finite differences meaningless. Run the numerical direction check with an fp32 decoder and TF32 disabled if necessary, with fixed features and several prespecified perturbation sizes. Record arithmetic settings and tolerances before opening accuracy results. A gradient check must cover the implementation actually claimed; a diagnostic of a different precision path cannot prove every reduced-precision detail.

**Historical v2 execution and replay commands**

The following commands document how the completed screen was run. Its successful terminal record, global seal and scores now exist. Do not execute its training/resume or creation commands again. Use verification-only commands when an integrity concern warrants replay. For any future registered job, Docker state, its lock, per-job exits and terminal record determine whether an owner is still alive; an observation timeout never authorizes a duplicate.

```powershell
docker --host npipe:////./pipe/dockerDesktopLinuxEngine ps --format '{{.Names}} {{.Status}}'
Get-Content reports/lora_profiled_v1/batch.lock -ErrorAction SilentlyContinue
Get-Content results/lora_profiled_v1/batches/screen02/invocations/20261118T104117Z_ede54062/terminal.json -ErrorAction SilentlyContinue
Get-ChildItem reports/lora_profiled_v1/batches/screen02/invocations/20261118T104117Z_ede54062/*.log
```

This resume block is retained for reproducibility/history, not as the current next action. It was used only after checking ownership and classifying interruptions. Each failure blocked the next command:

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

The following seal commands have already succeeded. The existing immutable seal must not be overwritten; `prepare` is historical and `verify` is the replay entry point. They audit byte hashes without deserializing scientific predictions or accessing held-source accuracy:

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

The [prospective terminal-failure policy](PROFILED_LORA_TERMINAL_FAILURE_POLICY.md) specifies that conditional amendment, classification audit, unchanged remaining inventory, unavailable metrics and required new evaluator fixtures. It was not activated: this completed screen had no numerical failure. Its unimplemented recovery path must not be mistaken for an observed scientific failure.

Public attempt evidence belongs in `results/lora_profiled_v1/cases/RUN_ID/ATTEMPT/`; large arrays and fresh caches belong in the corresponding `reports` path. Required outputs include source/manifest/input hashes, original-state hashes, initial/final adapter hashes, eight per-step reports for trained cases, actual call counts, finite gradient checks, timing/memory, native/final endpoint identities, and the completion/failure record. The runner's 1 GiB free-space guard is a minimum, not a 36-case storage estimate. Measure the first complete case's bytes and reserve enough local disk for the remaining cases and both physical backups before batching. Never place pretrained weights or complete restricted datasets in Git.

**Implemented scoring and selection contract**

The rules are frozen separately in [lora_profiled_evaluation_v2.json](../manifests/lora_profiled_evaluation_v2.json). Scientific v2 binds that specification, and [evaluation execution v2](../manifests/lora_profiled_evaluation_execution_v2.json) binds the scorer source and passed fixture evidence. The implemented sealer provides the required global prediction gate; a per-case completion record or registry flag alone cannot release scoring.

Create a global seal with exactly the 36 registered run IDs, one valid attempt per ID, common scientific-registry hash, completion-report hashes, adapter identities, and native/final endpoint hashes. Verify all referenced bytes, model/input identities and complete status. Reject missing/duplicate cases, mixed registrations, partial attempts and changed predictions. Seal all predictions before loading the opposite-source evaluation packet into the scoring process. An unrecoverable case must remain a reported failure under a prospectively amended termination protocol; it cannot disappear from the denominator or yield a qualifying nominee.

Reuse the *mapping and overlap logic* of `score_partner_shape.disjoint_test`, not its historical scorer. For each target, map/deduplicate the opposite packet with the registered native-pixel rule and exclude pixels used by the entire allowed adaptation packet (both spatial folds). Exclusion is determined only by input identity, never candidate error/confidence. Keep the same evaluation observations for every arm. For arm a, scene s, side t and view v, compute `MAE[a,s,t,v] = mean(abs(predicted_final_depth - measured_opposite_depth))` in meters over those fixed points. Average equally over five views, then equally over three scenes, keeping earlier/later directions separate. Define directional gain as `1 - mean_MAE[a,t] / mean_MAE[P1,t]`, not the mean of per-point percentage changes. Fail on empty views or nonfinite predictions; count nonpositive predictions and disqualify affected candidates rather than excluding their pixels.

Only P3 and P4 are candidates for the initial implicit-gradient hypothesis. P0, P1, P2 and P5 are controls. A detached arm winning is useful evidence against the mechanism, not permission to rename it a successful full-gradient method. Nominate at most one candidate with >=1% directional gain in each direction, improvement on at least two of the same three scenes in both directions, and no numerical failures. Maximize the smaller directional gain. Compare gains rounded to 12 decimal places for ties, then lower measured total runtime across its six cases, then smaller numeric arm ID. Always publish all six arms and the P3/P2 and P4/P5 contrasts. Do not use native-only results to select when the final-refined primary result fails.

The registered scorer fixtures cover missing/tampered seals, mixed registrations, overlap exclusion, unequal observation counts, invalid predictions, the two-same-scenes rule, tie handling, endpoint/gauge errors and the filename codec. Re-run them when relevant source changes require new evaluator registration. Publish compact per-view predictions/observations sufficient for independent CPU replay where the dataset terms permit.

For M3, freeze a separate executable dense-evaluation manifest before its scores are read. Use `registered_rgb` as the primary surface reference and `released` as alternate, with the pinned similarity-plus-ICP evaluator and fixed reference masks. For each scene/condition/reference, let A be mean accuracy distance and C mean completeness distance in meters. Normalize each component by the same-condition P0 component and set `surface_score = 0.5*(A/A_P0 + C/C_P0)`; aggregate equally by scene. The implemented five-arm inventory contains exactly one nominee and controls P0/P1/P2/P5. Select the control with the lowest aggregate primary score from exactly that four-control pool, with smaller arm ID breaking exact ties. Additional compatible established baselines may be reported separately after documenting their inputs; they do not silently enter this frozen selection pool or change the 360-state denominator. Historical runs with two-source data or a different final calibration cannot replace these controls. Use that same selected control for per-scene, component and alternate-reference checks; report raw meter-valued distances as well.

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
| `profiled_m3_gate.py` | 26 synthetic fixtures; full-inventory aggregation, frozen thresholds, common masks and paired validity counts | Cannot accept a partial inventory or authorize label access/rental |
| `check_profiled_m3_dense_runtime.py` | Two fresh CPU processes exactly repeated synthetic Sim(3), ICP, normals and metrics | No actual dense wrapper, universal determinism or study accuracy claim |
| `profiled_m3_dense.py` | 16 fixtures and exact legacy equivalence in two fresh pinned CPU processes | No IO, caching, study-score release or mask provenance verification |
| `profiled_m3_diagnostics.py` | 22 fixtures and paired production-raster synthetic integration with the converter/kernel | No independent alignment; primary axial-camera provenance remains the wrapper's responsibility |
| `prepare_profiled_m3_references.py` | 13 fixtures; reference plan v1 verifies six archive byte identities and 15 source poses, reproducing historical reference-camera algebra | Zero archive decoding; actual mask semantics and reference reprojection remain gated |
| `profiled_m3_stage_gate.py` | 18 fixtures; available-repeat necessary checks exactly agree with the full gate | Cannot create verified seals, pass the incomplete study or release rental funds |
| `check_profiled_m3_reuse_inputs.py` | Six fixtures and an actual pinned CPU process: exact equality of 23 fitting tensors in each of six centered/clean cases, including the original seven-column native-prior rows | Fitting equality alone does not authorize M2 prediction reuse |
| `profiled_m3_model.py` | Nine CPU orchestration fixtures; unchanged M2 teacher/training arithmetic, exact update/call counts, original-state checks and mandatory checkpoint/config/RGB identity checks before and after execution | No actual model run, CLI, stage controller, shared GPU lock, endpoint acceptance or score release |
| `profiled_m3_registry.py` | Eleven synthetic fixtures; exact 360-job/23-stage inventory, 24 actual case identities and source/input/model byte bindings | Static consistency only; nomination proof and real endpoint acceptance are not inferred from file hashes |
| `profiled_m3_attempt_audit.py` | Nine fixture methods; exact recorded model/step/cache/endpoint inventories and selected-case fitting counts | Does not independently recompute saved tensor-state digests, verify Docker exit or establish freshness across different processes |
| `prepare_profiled_m3_conversions.py` | Eight fixtures; fresh, reference-free output conversion with exact replay and preserved failures | Caller must first audit model membership/completion and bind converter/runtime/receipt before stage sealing |
| `profiled_m3_reference_arrays.py` | Nine generated-data fixtures; common mask, five nonempty views, camera hash and strict historical self-projection check | Does not establish archive provenance, physical calibration accuracy or permission to read references |

The earlier 150-test suite is preserved in `results/lora_profiled_m3_v1/evaluator_unit_checks.json`. With the new fitting audit and model-core fixtures, all 165 tests passed together on the host; the newer evidence is `results/lora_profiled_m3_v1/model_preparation_checks01.json`. Source-preparation, 24-case tensor conversion and actual pinned fitting-equivalence checks are separate. Read [the kernel audit](PROFILED_LORA_M3_KERNEL_AUDIT.md), [runtime comparison02](../results/lora_profiled_m3_v1/runtime_checks/comparison02.json), [leaf acceptance](../results/lora_profiled_m3_v1/dense_leaf_checks/comparison01.json) and [integrated acceptance](../results/lora_profiled_m3_v1/dense_integration_checks/comparison01.json) before implementing the remaining wrapper. Do not edit the hash-bound `PROFILED_LORA_M3_PREPARATION.md`, input plan v2, reference plan v1 or their source dependencies; new work belongs in separate files until explicitly versioned registration.

Accepted **CPU dense evaluation** settings are the pinned image, Torch intraop/interops 1/1, OpenCV threads 1, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` all 1 before Python starts. Preserve the existing cKDTree `workers=-1`. The earlier two-thread probes showed approximately machine-precision ICP differences; their reports and exact old source remain archived. The accepted one-thread setting affects this future CPU evaluator only; do not alter registered M2/model training. Never call historical `evaluate_points`, which sets Torch threads to two and uses insufficient cache identities.

The following is the preserved initial M3 implementation route. M2 has now nominated neither P3 nor P4, so this route is not released. A qualifying redesign would require its own method-specific registration before these components are used:

1. **Dense leaf kernel and camera/boundary adapter: synthetic acceptance complete.** Use the modules and evidence above; do not rewrite the audited scene-wide similarity-plus-rigid-ICP kernel. The remaining production wrapper must preserve transforms, fixed-mask validity, ICP fitness/RMSE/counts, raw meter-valued components, native/final endpoint identities and reference-specific boundary diagnostics. Axial metric-depth error uses unaligned metric depth and the registered-RGB reference's own pinned cameras. Alternate axial error remains unsupported until its distinct camera convention is verified. After authorized reference decode, verify all five views are nonempty, shared masks are identical, and reference projection agrees with its pinned camera convention. Do not copy historical caches or per-view alignment.
2. **Byte-verifying registration and controller.** Resolve actual nominee from verified M2 seal and replayed scores. Bind every source, checkpoint, input/split/noise packet, runtime, RGB/reference archive and fixed mask. `build_plan` consumes these identities but cannot verify their truth. Enforce one shared bool mask across both references and all arms, not merely equal point counts or separate per-reference hashes. Make fresh cache identities include all these dependencies. Preserve failures; never create invented receipts to satisfy the pure planner.
3. **GPU endpoint acceptance.** In a separately registered engineering attempt, validate that the M3 centered/clean path produces the expected native/final endpoint under exactly matched M2 settings. Check noisy-prior routing with the declared input packet. Distinguish original supplied approximately rigid matrices from native optimized rigid output rotations; do not orthogonalize the former or silently relax the latter's converter checks. A source/runtime change requires revisiting affected acceptance evidence.
4. **Prospective execution and evaluation registration.** Commit the full inventory, real executable commands, deterministic order, source/runtime hashes, terminal-failure policy, staged prediction-release rules, reuse audits and tested early-cell predicate before any M3 dense label access. The implemented early predicate may reject an already failed necessary gate; it may never declare an incomplete study successful. Its early-registration schema omits the not-yet-existing global prediction seal and accepts separate stage-seal descriptors, whose bytes the controller must verify. Reference archive hashes freeze masks before decoding; later semantic mask hashes must be derived from exactly those bytes, not selected by score. A plan with placeholders or invented future seal hashes is not registration. Do not modify M2 scoring or reuse its full-screen release procedure as though it already registered M3's staged protocol.
5. **Execute and verify each stage.** The inventory is 8 cells × 3 scenes × 5 arms × 3 repeats = 360 states, with 720 reference rows and both endpoint metrics per row. Thirty M2 states are only reuse candidates for repetition-zero centered/clean conditions; if every identity audit passes, 330 states remain new. Failed reuse audits require fresh states, not waived identities. There are 23 barriers: first both directional centered/clean repetition-zero cells together, then the six other repetition-zero cells, then eight cells in repetition one and eight in repetition two. Seal every prediction within a stage before its score release. Record a failed gate and mark unexecuted later stages conditional/not run.
6. **Apply the complete gate.** Select the strongest eligible control by primary surface score at repetition zero separately for each cell; keep that control for both later repeats, scene/component checks and alternate reference. Keep the same scene corruption matrices and optimizer seed42 across all repeats. Repetitions measure execution variation, not independent initialization or noise diversity. Require the frozen full-versus-detached mechanism inequality as well as surface efficacy. Only the full 360-state verified inventory can pass the complete gate. Then consider M4/M5 and the conditional rental reserve.

The full-versus-detached contrast for P3 uses P2; for P4 it uses P5. They must remain forward-matched in calibration and endpoint. For normalized primary surface score S (lower is better), the preparation contract requires a positive full-gradient benefit in every repeat and a repetition-zero benefit larger than the sum of the two arms' observed execution variations plus 1e-6. This is a mechanism check on this design, not a statistical significance claim or a guarantee of publication.

**Implementing the remaining controller without guessing**

The following are required interfaces for new code, not commands that already exist. Keep them in separate M3 files; do not retrofit the completed M2 launcher or alter its frozen release policy. The public model entry point is `profiled_m3_model.execute(ModelContext(...))`. Read the dataclass and its binding verifier before constructing a context. It accepts no public injected backend, makes one fresh model attempt per process, and writes only unused M3 case/attempt directories. Tests use a private mock backend; never use that path for scientific execution.

The prospective execution registry must use `schema="profiled_m3_execution_v1"`, contain the complete exact `jobs` inventory, and provide `model_core` with `input_plan`, `input_attempt`, `source_sha256_lf`, `artifact_sha256`, `docker_image`, `checkpoint_sha256`, `checkpoint_config_sha256`, `original_state_sha256`, `runtime_identity`, and `cases`. Each `cases[case_id]` contains actual `input_sha256` and `case_input_digest`. The pure planner accepts an opaque input identity; the implemented static execution registry now explicitly defines `inventory.inputs.case_input_sha256` and each job's `input_identity_sha256` as that case's CPU-bridge fitting-tensor `input_digest`. It checks equality with `model_core.cases[case_id].case_input_digest`; do not substitute a metadata-descriptor hash. Context additionally supplies the exact job, fresh attempt, registry path/hash, weights path and actual execution commit. Bind all required imported sources, pinned upstream Git blobs and model-loading configuration files. The core checks its mandatory subset; it cannot prove that a caller supplied a complete study registry or authorized stage.

Implement five operations with these responsibilities:

| Operation | Required behavior and acceptance evidence |
|---|---|
| Prepare | Verify/replay the complete M2 verdict; bind the nominee, complete planned inventory, code/runtime/checkpoint/config/input identities, reference archive bytes and fixed evaluation policy; refuse placeholders and changed registrations |
| Audit reuse | For each proposed M2 state, join its selected global-seal attempt, complete report, actual raw/public files, batch start/exit/verified records, process isolation, fitting audit, initialization, arithmetic and endpoint evidence; emit accepted or rejected with reasons |
| Seal stage | Require exact expected state IDs, actual complete artifacts and byte hashes, no pending GPU owner, preserved failures and matching registration; create an exclusive immutable stage seal only after every required state is accounted for |
| Score stage | Verify that seal and registration again, create an exclusive first-access receipt, then decode only the registered references/predictions; run the fixed converter/kernel/diagnostics and retain all rows and exact r0 control identity |
| Next/finalize | Apply the tested early predicate only to fully scored stages; stop on a necessary-gate failure; permit a later stage only after its predecessor passes. Final success requires the complete inventory and full gate; no early stage can release rental |

The wrapper also owns a shared local GPU lock, live Docker inspection, bounded runtime/disk checks, subprocess exit/timeout records, and logs for failures before the core reserves its attempt directory. Preserve the original process/container identities and both registry and execution commits. Different commits can contain identical bound arithmetic; compare the actual bound sources instead of demanding arbitrary commit equality. A new scientific setting requires prospective versioning, while an engineering repair requires identifying and rerunning only affected acceptance evidence.

Suggested minimal auditor interfaces are `audit_m2_reuse(root, m2_seal, m3_registry, job, fitting_receipt, config_proof)` and `audit_m3_complete(root, registry, job, attempt)`. Return canonical accepted/rejected state receipts linked to actual original files; never rewrite original model evidence. The former must call existing M2 attempt/seal verification and cannot accept a `pass_all` flag alone. The latter checks schema `profiled_m3_model_attempt_v1`, status `model_outputs_complete_without_scoring`, registry/core identities, checkpoint and config, prepared-input digest, one model construction, distinct process/model UUIDs, all steps/calls, unchanged original state and complete logical raw/public inventories. `seal_stage(...)` then checks exact stage membership and predecessor reviews, converts both endpoints without reference labels, and binds both converted-output hashes as well as native/final source hashes. Keep dependencies acyclic: execution registry first; evaluation registry binds execution; byte stage seal binds both; score-access receipt binds that seal; later semantic records bind those existing objects.

Resolve the first-reference-access ordering explicitly. (1) Freeze byte identities for both reference archives, the fixed mask field, expected raster/dtype and any already known counts, plus cameras, policy and runtime, without decoding their arrays. (2) Produce and byte-seal both initial centered/clean directional cells together. (3) Create the exclusive score-access receipt. (4) Decode exactly those archives and verify one shared boolean mask across both references, five nonempty views and the reference camera/projection convention; record the derived mask tensor digest/counts immutably. (5) Materialize the pure early-registration object and stage rows from those verified identities. (6) Only after all stages exist may the final aggregate registration include the global prediction seal. A file hash is not a semantic mask hash. Do not invent a future seal, choose a mask after seeing errors, or overwrite the original byte-level registration to hide the chronology.

The predecode byte stage seal binds the already frozen execution/evaluation registries. Its later semantic descriptor refers to that immutable seal and the newly derived early-registration hash. Never rewrite the byte seal to insert the later hash or create a circular registry dependency. If scoring is interrupted, preserve its receipt and partial outputs. A repeat request must verify the original receipt, sealed bytes, sources and registration, then perform a separately recorded exact replay or safe continuation under those same identities. Changed source/input/registration after first access must refuse this replay path; a fresh receipt cannot erase prior label access. Classify and preserve any required amendment before proceeding.

The current [fitting-equivalence audit](../results/lora_profiled_m3_v1/reuse_input_checks/reuse01.json) passes all six cases in the pinned model image, with no GPU or evaluation-array access. It is only one reuse prerequisite. The historical M2 registry explicitly binds `model.safetensors`, but no `config.json` byte binding was found in the current M2 manifests/environment evidence. The new M3 core checks both files before and after each attempt. A hash measured now, a file modification time or a fresh endpoint match cannot retroactively establish the configuration bytes used by earlier processes. The read-only weights bind mount prevented container writes; it did not put the host configuration inside the pinned image. Keep all 30 reuse proposals unapproved unless an already-sealed configuration record tied to the original executions is recovered. Otherwise use the registered fresh-state fallback: 360 new states, with the same economical stage barriers, rather than 330 new states plus 30 reuses. This limitation does not change M2's frozen protocol or erase its results.

Before scientific M3 release, test the controller with missing/tampered files, wrong jobs, changed config/RGB, stale process ownership, duplicate attempts, interrupted stages, repeated score-access requests and mismatched masks. Verify a fresh centered/clean endpoint in the real pinned runtime, then preserve the receipt with the exact accepted sources. Passing CPU fixtures is not a substitute for that model acceptance. No additional rental money is authorized by this implementation work.

The static registry and attempt-audit interfaces described above are now implemented. `verify_execution(...)` deliberately returns `launch_authorized=False`: it hashes nomination/acceptance proof files without parsing or authenticating those scientific decisions. `audit_m3_complete(...)` checks actual saved bytes and exact fitting-population counts against the case bridge, but its process UUID checks do not substitute for provider/Docker start, exit and global non-overlap evidence. The shared-lock launcher, release verifier, stage-seal controller and actual scorer remain unfinished. Do not interpret the name of an implemented helper as completion of those responsibilities.

The expanded combined suite passed all 202 tests, recorded with source hashes and the actual execution log identity in `results/lora_profiled_m3_v1/controller_unit_checks01.json`. The attempt-audit fixtures include all six arm schemas and internally consistent, rehashed wrong-population cases for both P0 and P3. These are generated-data tests, not audits of a real M3 model run.

For conversions, call `prepare_profiled_m3_conversions.prepare(root, job_id, attempt, native_descriptor, final_descriptor, registered_configuration)` only with audited endpoint descriptors. It creates a fresh conversion directory; `verify(...)` recomputes the conversion by default. First acceptance must use this replay and seal the returned complete-receipt hash and both output hashes. A later byte-only recheck must be anchored to that independently pinned receipt. The prospective evaluation registry must additionally bind the full image, Python and deterministic runtime; the converter's NumPy/OpenCV strings alone are insufficient. The 17 conversion/reference-array fixtures passed in a fresh pinned CPU process, recorded in `io_semantics_checks01.json`.

All 24 real fitting-case reports, including noisy poses, exactly reproduce in a pinned CPU-only process (`tensor_bridge_pinned01_container.json`). The whole report equals the earlier CPU bridge, so current case digests can be used without a runtime-dependent substitution. This is input equivalence, not a fresh model endpoint check.

After authorized reference decode, call `profiled_m3_reference_arrays.validate(arrays, scene_spec, known_K)`. Bind `known_K` from the geometry-plan metadata already identified by reference preparation, never from a candidate. It reproduces the existing camera-audit projection and strict `<0.01` native-pixel rule. Both archived reference conventions use the same nearest-sampled sensor depth and shared mask; the alternate is a reference-convention sensitivity check, not an independent dataset or independent physical ground truth. Preserve their archived bytes and do not regenerate depth with different interpolation.

Before creating the early registration, also define its observation-only `native_rows_sha256` prospectively. Use `prepare_profiled_inputs.array_hash` on the five selected original seven-column row arrays and native XY arrays, with fixed frame-based keys such as `25__original_rows` and `25__xy`, for each scene/side/grid. Require the identity to agree between clean/noisy cases and retain the derivation receipt. Do not use a whole-case digest here: it includes cameras, whereas the observation identity intentionally does not depend on the pose condition.

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

**Portable verification and reporting boundary**

A public checkout can replay compact M2 arithmetic using the existing command below. Use a fresh receipt path; the default receipt already exists and is intentionally immutable.

```powershell
$profiledReplayReceipt = 'reports/ci/profiled_public_replay_' + (Get-Date -Format yyyyMMdd-HHmmss) + '.json'
python scripts/replay_profiled_public_scores.py --output $profiledReplayReceipt
```

This checks the 420-field compact evidence, 180 metric rows, recorded provenance linkage and nomination. It does not reconstruct raw overlap masks, repeat model inference or prove physical ground truth. Full local seal/archive verification additionally requires the retained caches, checkpoint and C:/E: evidence. Report these scopes separately. Never stop unrelated `hbbs`/`hbbr` Docker containers.

The native goal stays active unless the user pauses it or the standing weekly reserve rule triggers. Documenting the handoff does not satisfy the remaining research. Update `progress.json` after a real checkpoint with evidence paths, running/terminal job state, next executable action, redesign count, dollar ledger and observed CI. Keep milestone progress separate from accuracy: 50% currently reflects M0+M1+M2 completion, while qualifying accuracy improvement is false.

**First actions for the successor**

The immutable M1 backup is verified on a second physical disk: `E:/ResearchData/TCO/research_results/lora_profiled_v1/snapshots/m1-20261118-verified01`. Its [public receipt](../results/lora_profiled_v1/backups/m1-20261118-verified01.json) records 136 files and 315,628,130 bytes. The complete M2 prediction archive is also copied and independently verified against current source bytes: `E:/ResearchData/TCO/research_results/lora_profiled_v1/snapshots/m2-20261118-verified01`, 2,281 files and 6,059,018,880 bytes; see its [receipt](../results/lora_profiled_v1/backups/m2-20261118-verified01.json). Both disks are physically distinct. Do not rerun copy using either existing snapshot ID. An integrity replay uses `python scripts/archive_profiled_evidence.py verify --destination E:/ResearchData/TCO/research_results/lora_profiled_v1 --snapshot m2-20261118-verified01 --check-source`. Compact scores/replay/code have a separate [score-evidence backup](../results/lora_profiled_v1/backups/m2-scores-20261118-verified01.json). Failed attempts and historical snapshots remain preserved.

Read the live progress file, the full M2 result table and the verified score/replay receipts. Preserve the completed `screen02`, historical `gradient01`, and retired v1 P0 evidence. The current task is the bounded local failure diagnosis, followed by at most one registered redesign if justified. Re-run completed preparation only if relevant files/runtime changed or integrity is uncertain. After meaningful results, update evidence paths, job state, next command, costs and Git/CI state. Held-source development scores are now known; those scenes cannot become unseen final tests later.

**Copyable instruction for a successor**

> Continue the existing active research goal in `C:/SVNProjects/Research/tco-research`. Read `docs/CALIBRATION_AWARE_LORA_HANDOFF.md`, the goal contract, mathematical specification, `results/lora_profiled_v1/README.md` and `progress.json` in that result directory. M2 is finished: all 36 states, global seal, official replay and independent arithmetic agree that neither P3 nor P4 qualifies. Do not rerun the screen or launch the conditional 360-state M3 study. Implement/register the bounded zero-update local diagnosis, inspect its evidence, and select at most one targeted scientific redesign across all failure branches. No qualifying accuracy gain or rental authorization exists. Keep original weights frozen; train only LoRA when a new protocol is registered. Public datasets only; existing USD 25 reserve remains conditional and cumulative. Preserve retained RunPod workspace and unrelated edits. At 5% weekly allowance remaining, checkpoint, stop/verify paid research pods and pause under standing instruction. Commit permitted code/results and verify backups/CI. The goal remains active; the broader accuracy-improvement ambition is unmet.
