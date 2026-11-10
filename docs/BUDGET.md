# Budget and rental discipline

Preferred ceiling: **USD 200 total**, including prior spending, compute, storage and recovery. USD 500 is a previously discussed stretch ceiling for compelling evidence; it is not automatically released. The latest component-ablation session completed for approximately USD 0.66 within its USD 3 cap. Any next session needs a new frozen plan and live quote; retain a USD 3 maximum for the next bounded diagnostic batch.

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
