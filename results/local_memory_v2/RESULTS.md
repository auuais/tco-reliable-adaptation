# Local full-resolution memory probe

Removing the unused point head before CUDA transfer and moving the image encoder to CPU after caching its patch tokens allowed **one adaptation step on five 392 x 518 views to complete on the 10 GB RTX 3080**, with **8.54 GiB peak allocated memory**. The full inference/adaptation path took **53.18 seconds**, excluding model load.

Matched zero-step control and offload pointmaps were bitwise identical. All predictions and the one adaptation gradient were finite; tracked base-parameter version counters were unchanged. This verifies the zero-step comparison and one-step feasibility only. Forty-step runs, adapted accuracy/equivalence and controlled speed comparisons remain untested. The shared Windows desktop GPU had varying available memory, so the observed timings are not a speedup benchmark. Allocated tensor memory is not physical VRAM residency.

An earlier attempt failed before model load because a directory scan encountered an Office sequence still being preprocessed. That failed run and its manifest are retained. The replacement study used the previously verified full-sequence metadata cache for the five original development sequences, leaving the selected Heads inputs unchanged.

See [the aggregate record](summary.json), [frozen protocol](../../manifests/local_memory_v2.json) and [wrapper](../../scripts/run_memory_probe.py). This is a memory-management experiment, not a new reconstruction method.
