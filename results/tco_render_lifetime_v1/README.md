# Native-batch TCO on a 10 GiB RTX 3080

The real rendering loss passed its same-input numerical checks. The full-model Native reference diverged between runs during the first backward pass, before either proposed change was applied. This closed the candidate promotion gate. The study used the local RTX 3080 and recorded **USD 0 rental compute**. [Registered experiment](../../manifests/tco_render_lifetime_v1.json) · [SOL handoff](../../docs/TCO_RENDER_LIFETIME_HANDOFF.md)

## Ten-view failure and memory trace

Both clean camera-checkpoint controls completed one optimizer update with native depth batches of eight and two. Both failed during the second update's RGB Gaussian rasterization on a 24 MiB request, with 8.66 GiB allocated under the 8.75 GiB cap. The clean ordinary Native control failed in the first update's depth head on a 260 MiB request at 8.69 GiB allocated.

The first instrumented ten-view attempt reached backward and hit the 1,200-second watchdog. Its unchanged retry reached the second RGB render and reproduced the OOM. In that retry, allocated memory rose 164.60 MiB across target-normal construction in each update. At the second render's failing allocation the deduplicated inventory identified 102.63 MiB of target-normal tensors, 12 MiB of gradients from update one, and 24 MiB of Adam moments still live. Target-normal loss weight is zero and visualization is disabled. The active RGB loss still needs the source normals, ED rendering, the visibility mask and the opacity denominator. These are measured lifetimes in the control. Savings from the proposed changes remain unmeasured at ten views. [Capacity and trace evidence](screen.json)

The two five-view control traces both finished. During the second RGB render, each still held 12 MiB of gradients from update one and 24 MiB of Adam moments. The subsequent `zero_grad` freed exactly 12 MiB in both traces. Allocated memory also rose 82.30 MiB across target-normal construction in both controls. A deduplicated inventory identified 51.32 MiB of target-normal tensors still referenced at RGB entry. The inventory tracks identified PyTorch storages and leaves extension allocations to the allocator measurements. Full two-update peak allocated memory was 7.822 GiB for ordinary Native and 6.302 GiB with camera checkpointing. [Phase evidence](screen.json)

## Numerical checks

The CPU fixtures verified the guard when normals were unused, when visualization or positive normal loss needed them, and with a near-empty visibility mask. The early-clear optimizer fixture matched the original gradients, moments and parameter updates and removed live spent gradients before the second forward.

On the GPU, the actual gsplat loss and the gradients with respect to its prediction inputs met the registered tolerances on both captured Heads updates and on a separate synthetic scene. Detaching ED opacity changed the derivative and triggered the negative control in all three replays. The renderer tests use common prediction leaves, the same view permutation and the native ED and RGB operators.

All fixed-adjoint checks matched exactly across native batches and across both updates, including all 384 LoRA tensors and the final reconstruction. In the real-loss experiment, both independent Native trajectories started with identical first predictions and loss (0.824327). Their largest per-tensor preclip gradient relative L2 difference was 1.4909% at that update, above the registered 0.01% limit. The largest relative difference in updated factor tensors was 11.9454%. At the next forward the Native repeat hashes differed and the losses were 0.754078 and 0.728837. The exact source of the backward variation remains unresolved.

The matched-state runs restored the same Native adapters, Adam state and RNG before update two. Both forward prediction hashes matched for each comparison in this table.

| Camera checkpoint policy, compared with ordinary Native | Max first-update preclip gradient relative L2 | Max second-update preclip gradient relative L2 |
|---|---:|---:|
| Original camera checkpoint | 1.309% | 1.389% |
| Skip unused target normals | 1.520% | 1.374% |
| Clear old LoRA gradients before the next forward | 1.661% | 1.158% |
| Apply both changes | 1.134% | 1.902% |

The relative comparisons above are over individual gradient tensors. They characterize variation in backward and Adam updates under the small registered fixture. They establish neither a change in the objective nor a 40-step reconstruction effect attributable to a particular intervention. [Qualification receipt](qualification.json) · [Per-attempt evidence](initial_evidence.json)

Two operational details are recorded with the evidence. The reader converts inherited zero-based clipped-gradient receipt labels to the one-based labels used in the comparison while preserving tensor bytes and tolerances. Three initial launcher attempts hit stopped containers with names from the earlier study. Their unchanged second attempts completed. [Receipt reader](../../scripts/qualify_tco_render_lifetime.py)

## Forty-update Native repeatability

All 12 scheduled five-view Native reconstructions and all 12 independent CPU rescores completed. The local comparator uses four retained transformer taps, every LoRA-adapted block, Adam at 0.0003, and the original grouped render. The two runs in each pair used the same inputs, seed and software. They started in separate processes.

| Window | Seed | Accuracy error in mm, replicas 1 / 2 | Completeness error in mm, replicas 1 / 2 |
|---|---:|---:|---:|
| Chess/seq-03 | 7 | 16.865 / 15.841 | 21.653 / 20.073 |
| Chess/seq-03 | 77 | 15.352 / 18.156 | 19.490 / 22.108 |
| Chess/seq-03 | 777 | 16.540 / 18.229 | 20.704 / 22.827 |
| Courtyard | 7 | 262.784 / 739.503 | 221.177 / 655.801 |
| Courtyard | 77 | 598.196 / 361.852 | 451.122 / 300.699 |
| Courtyard | 777 | 367.231 / 733.450 | 367.683 / 507.982 |

All six same-seed pairs differed by more than 1% on both metrics. Accuracy differences were 1.02–2.80 mm on Chess and 236.34–476.72 mm on Courtyard. Across all eight replica-label swaps, the ratio of three-seed median errors ranged from 0.872 to 1.146 on Chess and from 0.493 to 2.027 on Courtyard for accuracy. The corresponding completeness ranges were 0.908–1.101 and 0.592–1.689. This sensitivity is descriptive for these two windows. It gives no precision estimate for the broader fourteen-location benchmark.

The 12 rescores used the archived prediction arrays and the original reference geometry, and reproduced all reported distances within 0.01 mm. The largest measured discrepancy was about 1.2 × 10⁻¹¹ mm. The outputs themselves differed between repeats. [Repeatability replay](decision.json)

## Decision

No candidate passed full-model qualification. The ordered capacity screen, pilot, nine ten-view candidate completions and 84 five-view confirmation runs never opened. Ten-view capacity and the two 1% quality endpoints remain unestablished for the proposed changes. This study recorded no rental compute and supplies no basis for upgrading GPUs.

The next local experiment would replay the **same captured gradients from a real loss** into independent backwards passes through VGGT, then replay the measured variation between renderer gradients at that same boundary. The comparison would quantify how much drift first appears upstream of the renderer and how much the backbone and Adam amplify differences already present in the loss gradients. It would extend the fixed-adjoint test to the gradients that actually drove adaptation. [Independent decision](decision.json) · [Capacity receipt](capacity.json)

The 62 local attempts included 54 completions, four CUDA OOMs, one timeout and three pre-data Docker naming conflicts. Elapsed wall time from first launch to final receipt was about 2 hours 6 minutes. [Per-attempt evidence](initial_evidence.json) · [Second-disk archive receipt](archive_receipt.json)
