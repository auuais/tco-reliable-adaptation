# Five-view LayerNorm study — development

18 of 18 comparison slots scored. Joint verdict: **development**.

Per-room medians across three paired seeds, equal weight per room. Ratios below one favor the candidate.

| Comparator | Endpoint | Change | One-sided 95% lower | One-sided 95% upper |
|---|---|---:|---:|---:|
| L10 | accuracy_mm | +1.56% | -3.18% | +4.11% |
| L10 | completeness_mm | +1.63% | -2.95% | +4.63% |
| L10 | scene_seconds | -1.48% | -4.00% | +5.17% |
| L10 | peak_allocated_bytes | +0.00% | +0.00% | +0.00% |
| L10 | peak_reserved_bytes | +0.03% | +0.03% | +0.31% |
| L40 | accuracy_mm | -2.79% | -15.29% | +10.69% |
| L40 | completeness_mm | -4.12% | -19.22% | +14.51% |
| L40 | scene_seconds | -65.01% | -66.58% | -64.15% |
| L40 | peak_allocated_bytes | +0.00% | +0.00% | +0.00% |
| L40 | peak_reserved_bytes | -0.55% | -1.24% | +0.00% |

| Room | Arm | Accuracy (mm) | Completeness (mm) | Scene (s) | Live (MiB) | Reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|
| chess/seq-03 | L10 | 16.6468 | 21.2376 | 30.81 | 6437.74 | 7174.00 |
| chess/seq-03 | L40 | 16.0756 | 20.0139 | 84.55 | 6437.78 | 7256.00 |
| chess/seq-03 | LN10 | 16.4821 | 20.9474 | 30.31 | 6437.79 | 7176.00 |
| heads/seq-01 | L10 | 12.4447 | 13.2116 | 29.71 | 6437.74 | 7254.00 |
| heads/seq-01 | L40 | 14.1013 | 15.8719 | 85.89 | 6437.78 | 7256.00 |
| heads/seq-01 | LN10 | 12.9565 | 13.8239 | 29.31 | 6437.79 | 7256.00 |

Run coverage including anchors and duplicates: {'completed': 21}. Rental compute: USD 0.
