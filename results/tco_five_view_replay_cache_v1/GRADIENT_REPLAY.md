# Real-gradient replay on the RTX 3080

The reduced-raster five-view Heads fixture uses two independently restored adapter, Adam and RNG states from a native 40-step TCO configuration. Each replay applies the signed derivatives of the real TCO loss to matching live depth, confidence, camera poses and intrinsics, then runs the native clipping and Adam step. [Registration](../../manifests/tco_five_view_replay_cache_v1.json) · [Compact replay evidence](stage3_gradient_attribution.json)

| Measurement | Update 1 | Update 2 |
|---|---:|---:|
| Production pooled pre-clipping LoRA gradient variation, identical incoming derivatives | 0.6320–0.6453% | 0.6846–0.7368% |
| Production pooled pre-clipping variation, independently calculated derivatives | 0.6188–0.6271% | 0.6593–0.6754% |
| Production relative L2 change among independently calculated renderer derivatives | 1.05e−8–2.25e−8 | 3.25e−8–6.26e−8 |
| Deterministic pooled pre-clipping variation, identical incoming derivatives | Exact equality | Exact equality |
| Deterministic pooled pre-clipping change for the first regenerated derivative versus the captured derivative | 0.6808% | 0.6553% |

Each range covers the three pairwise comparisons of three fresh processes. Within production, the fixed-derivative result establishes variation after the loss boundary. The first deterministic regenerated derivative differed from its captured derivative by 7.05e−9 and 4.20e−8 relative L2 at updates one and two. At update two, all three independently regenerated derivatives produced identical LoRA gradients to one another, and each differed 0.6553% from the result using the captured derivative. This establishes sensitivity at these matched states. A specific CUDA operation and the effects on 40-step reconstruction remain open.

Changing the backend settings at the same first state changed depth and confidence by 0.06776% and 0.19060% relative L2. The six cross-backend attempts therefore failed their live-forward identity check. The prespecified fallback used a deterministic capture for the deterministic runs. Comparisons within each backend use matching states and forwards. The two backends use different anchors, which limits causal comparison across them.

The real sign-flip negative changed pooled gradients by 2.1595% and 3.2622%. Removing the opacity derivative was also detected. Both camera methods reproduced the native signed camera-token gradients at both production states. Whole-model production parity stayed unresolved against the three-repeat envelope. A fourth unchanged native replay also crossed that envelope on a single factor. The supplemented experiment uses the already verified deterministic anchor to test method parity and preserves the original production verdict. [Production qualification](qualification.json) · [Supplement registration](deterministic_extension.json)

Rental compute cost was USD 0.
