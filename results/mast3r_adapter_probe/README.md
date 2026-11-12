# MASt3R neural-adapter feasibility

One full-resolution decoder-adapter update completed locally on the RTX 3080. This establishes that the registered pairwise backward path fits the machine. It does not evaluate reconstruction accuracy or implement the complete TCO objective.

| Measurement | Result |
|---|---:|
| Image pair | Heads frames 25 and 225 |
| Resolution | 512 by 384 |
| Adapter rank / alpha | 4 / 1 |
| Decoder attention modules | 144 |
| Trainable adapter parameters | 1,032,192 |
| Sparse depth samples | 66 |
| Adam updates | 1 |
| Peak allocated GPU memory | 3,745.72 MiB, about 3.66 GiB |
| Update plus post-update forward | 1.20 seconds |
| Additional rental cost | USD 0 |

All eight pairwise output tensors were exactly equal before injection and with zero-update adapters. Gradients were finite, with 138 of 288 adapter gradient tensors having nonzero norm. Some initial factors have zero gradients with zero-initialized output factors; the objective supervises only the first point map. The point map changed after the update. Cryptographic hashes of every original model-state tensor matched before and after, including buffers.

The probe uses bfloat16 autocast, the released float32 head path, a frozen encoder and heads, and non-reentrant decoder checkpointing. Timing excludes checkpoint loading, baseline checks and state hashing. Five-view training memory, numerical repeatability, dense accuracy and throughput remain unmeasured. The objective is sparse axial-depth Huber fitting with a fixed range-derived scale. It is not the complete TCO rendering/constraint objective or a Test3R reproduction.

Attempt decoder_checkpoint01 failed during direct-model input setup because true_shape was a NumPy array rather than a tensor. No adapter update ran. Its original source, manifest and TypeError record are preserved. The explicitly registered decoder_checkpoint02 corrected that metadata type and passed. No rental or automatic failed-attempt retry was used.

Before/after point maps are public and checksum-verified on two distinct physical disks. Run:

```sh
python scripts/verify_mast3r_adapter_probe.py
```

The verifier checks both frozen source registrations, the preserved failure, module coverage, finite gradient records and output differences. It independently recomputes the sparse fitted loss and scale from public inputs. It does not rerun the network or independently verify the recorded memory measurement or original-state hashes. Sparse fitted loss is not a reconstruction-quality metric.

Next prepare a frozen multi-view adaptation protocol and clean/noisy-prior controls against the strong [alignment baselines](../mast3r_joint_priors/README.md). Neural reconstruction gains and publication novelty remain unestablished. The USD 25 rental allocation is reserved.
