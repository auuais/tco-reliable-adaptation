# Forty full-resolution updates on the local RTX 3080

The existing inactive-module offloading wrapper completed **40 updates on five 392x518 views** of Heads using the local 10 GB RTX 3080. Peak allocated tensor memory was **8.6269 GiB** and inference/adaptation took **227.90 seconds**, excluding model load and reference evaluation. All 40 gradient records and final predictions were finite; tracked base-parameter version counters were unchanged.

The fresh offloaded zero-step prediction was bitwise identical to the earlier canonical, non-offloaded zero-step control. This validates that control and demonstrates forty-step execution feasibility in this environment. It does **not** establish equivalence of adapted outputs, accuracy, or a speedup. Only one adapted execution and one scene were tested.

Peak reserved memory reached **10.5703 GiB**. Allocated/reserved tensor memory is not physical VRAM residency, and this was a shared Windows/WSL desktop setup. Do not infer guaranteed fit on every 10 GB GPU or a native-Linux memory guarantee from the 8.63 GiB allocation measurement.

The engineering change removes an unused point head before GPU transfer and moves the image encoder to CPU after caching its patch tokens. The adaptation objective is unchanged. Use this path for further local prototype feasibility checks, with a fresh process and explicit evidence record for every run. Keep it a named configuration in accuracy/latency comparisons.

See [validated summary](summary.json), [frozen manifest](../../manifests/local_memory_v3_full40.json), and [validator](../../scripts/aggregate_memory_full40.py). The raw evidence is included in the [two-drive local diagnostic archive](../attention_probe_v6_local/archive.json); no rental GPU cost was incurred by this check.
