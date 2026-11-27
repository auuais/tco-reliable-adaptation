# Five-view LayerNorm study — followup

18 of 18 comparison slots scored. Joint verdict: **development**.

Per-room medians across three paired seeds, equal weight per room. Ratios below one favor the candidate.

| Comparator | Endpoint | Change | One-sided 95% lower | One-sided 95% upper |
|---|---|---:|---:|---:|
| L10 | accuracy_mm | -0.83% | -4.13% | +6.22% |
| L10 | completeness_mm | -2.29% | -5.53% | +5.98% |
| L10 | scene_seconds | -4.85% | -6.44% | -0.79% |
| L10 | peak_allocated_bytes | +0.00% | +0.00% | +0.00% |
| L10 | peak_reserved_bytes | +0.03% | +0.03% | +0.45% |
| L40 | accuracy_mm | -4.87% | -18.63% | +13.34% |
| L40 | completeness_mm | -7.46% | -23.53% | +15.86% |
| L40 | scene_seconds | -66.21% | -67.41% | -65.10% |
| L40 | peak_allocated_bytes | +0.00% | +0.00% | +0.00% |
| L40 | peak_reserved_bytes | -0.55% | -1.38% | +0.00% |

| Room | Arm | Accuracy (mm) | Completeness (mm) | Scene (s) | Live (MiB) | Reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|
| chess/seq-03 | L10 | 16.6468 | 21.2376 | 30.81 | 6437.74 | 7174.00 |
| chess/seq-03 | L40 | 16.0756 | 20.0139 | 84.55 | 6437.78 | 7256.00 |
| chess/seq-03 | LN10_low | 16.5771 | 20.8146 | 29.17 | 6437.79 | 7176.00 |
| heads/seq-01 | L10 | 12.4447 | 13.2116 | 29.71 | 6437.74 | 7254.00 |
| heads/seq-01 | L40 | 14.1013 | 15.8719 | 85.89 | 6437.78 | 7256.00 |
| heads/seq-01 | LN10_low | 12.2893 | 12.8700 | 28.41 | 6437.79 | 7256.00 |

Run coverage including anchors and duplicates: {'completed': 20}. Rental compute: USD 0.
