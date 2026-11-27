# Five-view LayerNorm study — primary

54 of 54 comparison slots scored. Joint verdict: **supported_violation**.

Per-room medians across three paired seeds, equal weight per room. Ratios below one favor the candidate.

| Comparator | Endpoint | Change | One-sided 95% lower | One-sided 95% upper |
|---|---|---:|---:|---:|
| L10 | accuracy_mm | +2.83% | -0.15% | +5.17% |
| L10 | completeness_mm | +3.02% | -0.67% | +5.83% |
| L10 | scene_seconds | +0.68% | -2.31% | +3.55% |
| L10 | peak_allocated_bytes | +0.00% | +0.00% | +0.00% |
| L10 | peak_reserved_bytes | +0.00% | -0.46% | +0.09% |
| L40 | accuracy_mm | -1.54% | -7.36% | +3.60% |
| L40 | completeness_mm | +10.36% | +3.61% | +14.41% |
| L40 | scene_seconds | -64.98% | -65.60% | -64.49% |
| L40 | peak_allocated_bytes | +0.00% | +0.00% | +0.00% |
| L40 | peak_reserved_bytes | -0.46% | -1.99% | +0.19% |

| Room | Arm | Accuracy (mm) | Completeness (mm) | Scene (s) | Live (MiB) | Reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|
| apt2/living | L10 | 14.0918 | 15.9669 | 28.92 | 6437.74 | 7256.00 |
| apt2/living | L40 | 13.9155 | 15.0492 | 83.35 | 6437.78 | 7256.00 |
| apt2/living | LN10 | 14.5727 | 16.4487 | 29.33 | 6437.79 | 7256.00 |
| apt2/living | ZERO | 14.5059 | 16.7801 | 3.21 | 5732.66 | 6010.00 |
| apt2/luke | L10 | 8.7300 | 8.3489 | 28.65 | 6437.74 | 7236.00 |
| apt2/luke | L40 | 8.8325 | 7.9066 | 83.83 | 6437.78 | 7296.00 |
| apt2/luke | LN10 | 8.9573 | 8.6321 | 29.91 | 6437.79 | 7236.00 |
| apt2/luke | ZERO | 10.7372 | 12.9237 | 3.24 | 5732.66 | 6010.00 |
| office1/gates362 | L10 | 7.9652 | 8.7182 | 29.56 | 6437.74 | 7176.00 |
| office1/gates362 | L40 | 9.3166 | 7.7290 | 83.68 | 6437.78 | 7256.00 |
| office1/gates362 | LN10 | 8.5811 | 9.6260 | 28.06 | 6437.79 | 7176.00 |
| office1/gates362 | ZERO | 9.0470 | 10.8821 | 3.83 | 5732.66 | 6010.00 |
| office1/gates381 | L10 | 6.5776 | 7.8965 | 29.10 | 6437.74 | 7256.00 |
| office1/gates381 | L40 | 7.3764 | 7.9651 | 83.44 | 6437.78 | 7276.00 |
| office1/gates381 | LN10 | 6.6405 | 7.9381 | 29.50 | 6437.79 | 7256.00 |
| office1/gates381 | ZERO | 6.7805 | 8.7865 | 4.40 | 5732.66 | 6010.00 |
| office1/lounge | L10 | 6.2903 | 10.0685 | 28.63 | 6437.74 | 7276.00 |
| office1/lounge | L40 | 6.9191 | 9.4636 | 84.04 | 6437.78 | 7296.00 |
| office1/lounge | LN10 | 6.3345 | 10.1071 | 29.56 | 6437.79 | 7276.00 |
| office1/lounge | ZERO | 6.7781 | 11.4292 | 4.33 | 5732.66 | 6010.00 |
| office1/manolis | L10 | 13.5820 | 16.6735 | 30.10 | 6437.74 | 7216.00 |
| office1/manolis | L40 | 12.4346 | 14.8568 | 84.49 | 6437.78 | 7236.00 |
| office1/manolis | LN10 | 13.7970 | 16.7422 | 29.72 | 6437.79 | 7216.00 |
| office1/manolis | ZERO | 12.2547 | 15.4489 | 4.38 | 5732.66 | 6010.00 |

Run coverage including anchors and duplicates: {'completed': 60}. Rental compute: USD 0.
