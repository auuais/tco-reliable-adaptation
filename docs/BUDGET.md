# Budget and rental discipline

Preferred ceiling: **USD 200 total**, including prior spending, compute, storage and recovery. USD 500 is a previously discussed stretch ceiling for compelling evidence; it is not automatically released. The completed component-ablation and repeatability/calibration sessions cost approximately USD 0.66 and USD 0.72 respectively, each within its own USD 3 cap. Later allocations are currently withheld. A new paid session requires a qualifying redesigned method, a frozen plan and a live quote.

| Allocation | Ceiling USD | Release condition |
|---|---:|---|
| Reproduction and numerical stability | 20 | Active; includes historical costs and the first baseline screen |
| Prior-quality and simple-baseline ablations | 20 | First screen complete and interpretable |
| Reliability-method prototypes | 40 | Useful validation signal beyond simple baselines |
| MASt3R transfer | 25 | VGGT method passes the development gate |
| Held-out scenes, second dataset and estimated priors | 45 | Frozen method and evaluation split |
| Independent repeats and final tables | 20 | Candidate contribution survives held-out evaluation |
| Storage, recovery and contingency | 30 | Keep a ledger; unused amounts remain unspent |
| **Total** | **200** | No automatic escalation to 500 |

Initial historical spending is approximately USD 1-2 based on rounded, delayed provider balances, not a settled invoice. Reconcile actual billing locally before allocating subsequent sessions; public records need only aggregate costs, not account details. The current retained 20 GB stopped workspace was quoted at roughly USD 0.13/day: 120 days would be about USD 15.60. Rates can change; do not treat this arithmetic as a price guarantee. The old duplicate has been removed.

Use the RTX 3080 for CPU analysis, data preparation, synthetic geometry tests, small two-view debugging, method logic and artifact generation. Keep those outputs separate from paper-resolution experiments. Use a rented GPU with at least 24 GB for full-resolution trials after memory checks; the existing cached 48 GB Ada environment is convenient for the first study. Compare real time-to-result and availability before switching to another GPU. No full-backbone training or multi-GPU purchase is planned.

Prepare and test code, manifests, data and wheels before starting paid compute. Launch one job stream with fresh processes, a per-job timeout, a batch deadline and no automatic retries. Preserve a failed attempt under its original ID. Stop on the first unexpected failure; do not spend the remaining budget blindly. A process timeout does not stop provider billing: collect and checksum evidence, then explicitly stop the pod in the provider console and verify that state. Do writing, plotting and Git publication locally afterward.

Keep one retained volume while frequent sessions need it. Verified local backups permit a later decision to remove storage, but deletion is a separate explicit action. Do not keep duplicate volumes or start extra GPUs speculatively.

## First tranche completed

The `baselines_v4` session completed 39 jobs and 1,440 updates. Its raw archive passed remote and local checks, then the GPU was stopped. Elapsed-time cost at the quoted roughly USD 1/hour rate was about **USD 0.67**, within the USD 5 session cap. The observed balance change was USD 0.58; billing can settle later. These costs belong to the first USD 20 allocation, not an addition to the USD 200 ceiling.

Restart required a provider migration. After explicit user approval, the old stopped pod and its duplicate 20 GB volume were permanently deleted. Only the new stopped workspace remains, reducing quoted storage from roughly USD 0.26/day to USD 0.13/day. All prior archives were verified on the migrated workspace and locally before deletion. The next component-ablation session has a provisional USD 3 cap, released only after its offline preparation is complete.

## Component ablation completed

`prior_ablation_v5` completed all 39 jobs / 1,440 updates in 50.89 experiment minutes. The retained Ada could not restart or migrate due to capacity, so the full study ran on one A40 at a quoted USD 0.59/hour plus approximately USD 0.007/hour container storage. No persistent volume was added. Local tests and a committed manifest preceded the experiments.

Total estimated session cost including setup, collection, validation and shutdown was **USD 0.66**, within the USD 3 cap and charged to the existing **USD 20 prior-quality/simple-baseline allocation**. The observed balance change was also USD 0.66; these observations are not a settled invoice. The complete raw archive passed local validation and checksum verification on two local drives before the A40 was stopped. The new pod now costs **USD 0.00/hour**. The pre-existing stopped 20 GB volume remains approximately USD 0.13/day, so storage charges are unchanged.

Standard wheels were restored directly from PyPI using the original exact file hashes after a slow upload was interrupted; locally built renderer/runtime wheels and system packages came from the verified cache. The one setup failure and recovery logs are retained. None of the 39 experiment jobs failed or was retried. The next work is local repeatability analysis before releasing another bounded cloud batch; MASt3R and controller allocations remain conditional.

## Repeatability/calibration stage authorization

The user authorized a new locally prepared repeatability/calibration stage capped at **USD 3 total**, including setup, collection and unsuccessful attempts. The associated CPU analyses and RTX 3080 integration checks do not consume rental funds. The stage uses a separately frozen manifest; the cap does not reset after a failed deployment. A live restart quote on 2026-11-11 was USD 0.59/hour for the stopped A40, with running container storage additional. No GPU has been restarted for this stage at the time of this preparation entry.

The user explicitly instructed that, if diagnostics do not produce a reliable improvement, **wait for a redesigned method before spending later allocations**. Therefore the USD 25 MASt3R allocation and later held-out/cross-dataset and independent-verification allocations remain unspent until that evidence gate is met. The storage/contingency reserve should remain unused unless needed; it is not an instruction to exhaust the budget.

## Repeatability/calibration GPU stage completed

All 40 jobs / 1,480 updates completed on one A40 at USD 0.597/hour including running container storage. Approximately **USD 0.72** was observed for the session, including setup, one pre-experiment path-resolution recovery, collection and shutdown; this is a rounded balance observation, not a settled invoice. It is within the authorized **USD 3 stage cap** and belongs to the existing diagnostic allocations. No experiment was retried. Raw evidence was verified on two local drives before stopping the pod; its confirmed idle cost is **USD 0/hour**, with no new persistent volume. The older retained 20 GB volume is unchanged at approximately USD 0.13/day.

Neither calibration alternative passes the frozen gate. Later allocations therefore remain withheld. CPU sensitivity analysis and the planned local RTX 3080 precision comparison consume no RunPod allocation. The latter uses an offload path now tested for 40 full-size updates, subject to the documented Windows/WSL memory-residency limitation. These local engineering checks do not release MASt3R funding automatically.

## Weekly usage and experiment shutdown

The current project reserve is 5% of the weekly Codex allowance. Check the signed-in weekly window before starting a rental and at each research checkpoint. At or below the reserve, stop any running TCO experiment pod, preserve the checkpoint and pause the goal for the user to resume. The operator can read the counter locally with `python scripts/weekly_pod_guard.py --reserve 5 status`. The implementation uses the rate limits interface documented in [OpenAI Docs](https://developers.openai.com/codex/app-server).

For a paid session, set an absolute UTC deadline from the launch time, the live hourly quote and the session cap, allowing time to collect results. Run the local guard with `python scripts/weekly_pod_guard.py --reserve 5 watch-pod --host HOST --port PORT --pod-id ID --deadline UNIX_UTC --identity SSH_KEY_PATH` in a separate process. Confirm the printed lease acknowledgement before launching the batch. If arming fails, stop the pod in the provider console. The remote guard checks its own pod ID and uses the pod-scoped credential. It checks a heartbeat every two seconds, requests a stop after 120 seconds without renewal, and has its own hard deadline. The local client checks quota every 30 seconds and requests a stop on reaching the reserve. Keep a persistent or incrementally downloaded results checkpoint. Verify the provider's stopped state and hourly compute rate after shutdown. The stop operation preserves the pod record. [RunPod environment variables](https://docs.runpod.io/pods/templates/environment-variables) · [RunPod stop endpoint](https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop)

On November 11 the signed-in quota reader and the pod lease ran throughout a 12-condition full-model calibration experiment. After two local copies of every completed trajectory were verified, the lease requested shutdown. The RunPod console subsequently showed the RTX 3090 stopped at $0.00/hour. The account balance indicated approximately $0.95 for this calibration stage including the earlier repeatability runs. Charging the full second-session interval at the quoted rate brings the estimate to $0.99 against the $3 cap. The $25 MASt3R allocation remains reserved. [Session timing, rates and raw evidence checksums](../results/canonical_sensor_factorial/session.json)

The subsequent focal-calibration pilot completed 72 full TCO runs on one RTX 3090. RunPod quoted $0.50/hour for compute and $0.007/hour for its temporary 50 GB disk. The provider showed this pod stopped at $0.00/hour after both local drives received all 72 runs. From the initial GPU request through confirmation of shutdown, the conservative cost estimate was **$0.83**. This brings the calibration-stage estimate to **$1.82 of $3**, including earlier attempts. The pilot added zero persistent storage. Scoring uses the local CPU. [Timing, rates and checksums for both copies](../results/dense_focal_pilot/session.json) · [Sealed point-map hashes](../results/dense_focal_pilot/seal.json)


## November 12 local evaluation and verification

MASt3R inference/global alignment, the two-grid sparse-depth development studies, all 260 held-out reference evaluations, all 48 TUM surface evaluations, nine independent checks and the fresh TUM model repeat used the local machine. Their additional RunPod compute spending was zero. The calibration-stage estimate is still USD 1.82 against its USD 3 cap. The USD 25 transfer allocation and later evaluation allocations were not charged for these local runs. The candidate failed the TUM advancement checks, so those rental allocations are reserved.

Full point maps and frozen model outputs are checksum-verified on separate C and E physical disks. Compact results and replay inputs are in the public GitHub repository. Account credit observations are private and are not a measure of total historical spending. The current provider pod state and storage quote await direct user confirmation because Windows Computer Use could not verify the browser URL and no authenticated RunPod CLI was available. The previous stopped-state and storage observations are historical evidence.
