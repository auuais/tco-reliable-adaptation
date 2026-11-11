# Using only proposed changes biases a tile's depth correction

We replayed all five depth rules on nine image batches and recovered the original discrete choice at every pixel in all 225 arm-view combinations. We then compared two decisions that affect which pixels train a tile correction. A source camera must first have a valid observation, and its best depth candidate may then be unchanged. The evaluation used disjoint spatial blocks and only pixels where the first source had proposed no change. These pixels cover 85.40% of the valid reference raster. [Registered design](../../docs/ELIGIBILITY_AND_SELECTION.md) · [Eligibility counts and checksums](support.json)

## Which sites supply the depth labels?

All values are percentage reductions in corresponding-pixel squared 3D error against direct focal substitution at fixed alignment. Positive values indicate improvement. Each of the twelve development conditions receives equal weight. Reference-assisted rows use the depth reference only on their designated training sites. Model-only rows use the source model's estimate.

| Training sites and targets | All test pixels, registered | Eligible identity pixels | Unavailable pixels | All test pixels, released |
|---|---:|---:|---:|---:|
| Reference at nonzero proposals | −10.60% | −21.76% | −12.41% | −8.47% |
| Reference at equally many random eligible sites per tile | +4.47% | +14.45% | −4.06% | +5.91% |
| Reference at eligible identity sites | +4.97% | +16.64% | −3.44% | +6.42% |
| Reference at all eligible sites | +4.98% | +15.00% | −3.54% | +6.56% |
| Reference anywhere, matched to eligible count | +8.61% | +13.97% | +6.19% | +9.86% |
| Reference anywhere, matched to nonzero-proposal count | +8.07% | +13.60% | +5.54% | +9.19% |
| Model targets at nonzero proposals | −10.38% | −12.71% | −9.92% | −9.87% |
| Model targets at all eligible sites | −0.93% | +0.57% | −1.66% | −1.01% |
| Model targets at eligible sites in three predicted-depth bands | −1.98% | +0.23% | −3.78% | −2.08% |

Selecting sites by disagreement produced a 15.07-percentage-point difference from a random draw of the same size among eligible training pixels. The direction held in Heads, Chess and Fire. Visibility also affected transfer. On withheld pixels that had no eligible source, reference labels at the eligible training sites increased error by 3.54%. The same number of labels drawn across the tile reduced it by 6.19%. The subgroup percentages have different condition-specific squared-error denominators and cannot be averaged to recover the combined column. [All condition-normalized results by scene, reference and test population](summary.json)

## Implication for the next experiment

Keeping eligible identity observations reduced the harm from propagating the actual model's nonzero proposals, from a 10.38% error increase to 0.93%. Splitting each tile into three predicted-depth bands failed on all three scenes. None of these rules qualifies for transfer spending.

An earlier reference-assisted surface diagnostic also found substantial depth-correction capacity outside the first source's proposed sites. Choosing among five candidates at every valid pixel gave a registered surface score of 0.8134. Choosing among the same candidates only at proposed sites gave 0.9773, and accepting or rejecting the actual proposal gave 0.9820. Direct focal substitution scored 0.9850. Those figures were measured by the surface evaluator on explicitly constructed maps. [Surface diagnostic](../visibility_depth_evidence/RESULTS.md)

The five original frames in each sequence were 200 frames apart. The next local control should test whether changing the spacing of four source cameras improves coverage of previously unavailable pixels at the same target. Keep the number of sources per target fixed and report total distinct frames and model calls separately. Compare any subsequent deployable method against a frozen baseline given exactly the same images. The three development sequences and the reference-assisted comparisons support a study of this mechanism. They do not establish generalization.

## Reproduction

The public verifier checks packed eligibility masks, matching counts per tile, both training block colors, reference-independent model fits, and agreement with the prior tile moments.

```sh
python scripts/capture_visibility_support.py verify
python scripts/observation_selection.py verify
```

For full replay in a scratch checkout with prepared 7-Scenes inputs, frozen heads and an empty output directory for this study, run `python scripts/capture_visibility_support.py capture` followed by `python scripts/observation_selection.py measure`. The study ran locally with zero additional RunPod cost. [First-source rule and evidence](../visibility_depth_evidence/RESULTS.md)
