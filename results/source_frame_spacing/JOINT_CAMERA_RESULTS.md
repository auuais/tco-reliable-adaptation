# Joint camera control

Positive values mean lower natural squared log-depth error. The primary condition perturbs absolute orientation by one degree and each translation direction by five degrees. The comparison uses three seeds on the same 15 development targets, with identical images, predicted baseline lengths and depth-update weights.

| Scene | Gain over unchanged depth | Gain over frozen cameras | Gain over orientation alone |
|---|---:|---:|---:|
| Overall | -1.31% | -3.26% | 0.06% |
| heads/seq-01 | -1.72% | -4.98% | -0.56% |
| chess/seq-03 | -0.78% | -0.28% | 0.36% |
| fire/seq-03 | -1.41% | -4.52% | 0.40% |

Supplying both exact components increased error by 4.64% against frozen cameras. Using exact directions alone increased it by 1.71%. The registered 2% improvement threshold failed. This diagnostic gives no basis for a surface evaluation or transfer spending. Future redesigns should address how translation magnitude and noisy image correspondences affect the depth update.

[Protocol](../../docs/JOINT_CAMERA_CONTROL.md) and [full results](joint_camera_control.json). Cloud spend was $0.
