# MASt3R joint-prior alignment results

All 21 registered alignments completed on the local RTX 3080, with every prediction sealed before dense evaluation. All 42 reference evaluations completed. This is a three-scene development experiment with supplied benchmark trajectories and neighboring sparse depth, not a held-out or realistic-sensor robustness result.

The primary score is the equal-scene mean of normalized accuracy and completeness errors. Ordinary MASt3R is 1 in each scene. Lower is better.

| Method | Centered grid | Shifted grid |
|---|---:|---:|
| Ordinary MASt3R | 1.000000 | 1.000000 |
| Depth-prior alignment | 0.937104 | 0.925344 |
| Pose-prior alignment | 0.980064 | 0.978304 |
| Joint depth/pose alignment | **0.923714** | **0.913420** |
| Earlier spatial output correction | 1.063843 | 1.039293 |
| Earlier depth-aware output correction | 1.012217 | 1.003752 |

Joint alignment improved over ordinary MASt3R by **7.63% and 8.66%**. All scenes improved against ordinary alignment. Heads gained 12.90% and 13.71%, Chess 0.97% and 3.07%, and Fire 9.02% and 9.19%. The earlier output-substitution Fire regression did not recur under these conditions. This supports coordinating the variables rather than substituting outputs, but does not isolate a unique causal mechanism.

The strongest control was depth-only alignment, already 6.29% and 7.47% better than ordinary MASt3R. The joint candidate added only **1.43% and 1.29%**. Against depth-only, centered scene gains were 2.44%, 0.015% and 1.96% on Heads, Chess and Fire. Shifted gains were 1.78%, minus 0.050% and 2.22%. Component-safety and alternative-reference checks passed on both grids. Both failed the 5% gain criterion; shifted also failed the strict no-scene-regression check. Neither advances under the registered resource criteria. These internal criteria are not statistical significance tests.

On the released reference, joint scores were 0.935540 and 0.927927, versus depth-only 0.947674 and 0.938520. All accuracy and completeness components are in scores.json. No claim relies on normal consistency. Pose-only is an objective ablation using the same sparse-range scale anchor; it is not a no-depth-input setting.

## Implementation and verification

The checkpoint and descriptors remained fixed. Priors inside fine geometric alignment adjust camera, focal and depth variables. This does not implement network-weight or LoRA adaptation. Every run used the released 300-coarse/300-fine schedule and its existing final-output convention. All 5,400 prior backward passes across 18 constrained runs were finite. The real gradient probe passed. The disabled-hook bridge reproduced standard outputs bit-for-bit.

Fresh standard alignments exactly reproduced all nine archived depth/intrinsic/camera arrays across the three scenes, validating reuse of earlier matched-input controls. Both grids, all weights and the 21-job manifest were fixed before execution; completion records preserve the execution commit. There was no hyperparameter sweep or failed experiment retry.

Cached alignment runtimes were 16.56–21.91 seconds, totaling 419.61 seconds. Peak PyTorch allocation was 2,717.24 MiB, about 2.65 GiB. These exclude checkpoint loading and reuse cached predictions/matches. They do not describe cold inference or neural-adaptation memory. Additional rental compute cost was **USD 0**; the USD 25 allocation remains reserved.

All 21 full point maps and 25 native/probe archives are checksum-verified on two distinct physical disks. Public native outputs, completion records, full prior traces, reference identities and replay samples accompany the tables. Run:

```sh
python scripts/replay_mast3r_joint_priors.py
```

The replay independently resizes and projects native outputs, checks sampled coordinates, verifies objective traces and exact baseline repeats, and recomputes both decisions from 42 stored measurements. It does not independently rerun the model, optimizer or dense evaluator. Full reruns additionally need the pinned source/checkpoint and public dataset preparation.

## Next decision

Keep depth-only and joint alignment as strong baselines. Their small incremental pose benefit does not establish novelty or reliability. Prepare network-adaptation feasibility and controlled prior-noise tests locally using [the handoff](../../docs/MAST3R_NEXT_IMPLEMENTATION.md), including appropriate Test3R and Pow3R comparisons. A new backbone or additional priors alone are insufficient novelty. Previously evaluated held-out and TUM labels remain observed evidence and cannot become untouched confirmation for a redesigned method.
