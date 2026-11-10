# How much do identical-seed reruns change the reference comparison?

For every observed substitution of same-seed reruns, switching from the released reference to the registered RGB reference reduced TCO's apparent advantage over frozen VGGT. The difference was **5.94 to 7.50 percentage points** in relative error. Switching to the RGB SfM reference reduced it by **4.30 to 5.17 points**. This is a paired comparison of the same model outputs across references. [All paired contrasts](summary.json)

| Reference | Unchanged TCO, observed range | Half of TCO's relative depth change, observed range |
|---|---:|---:|
| Released | 0.9117–0.9368 | 0.9048–0.9146 |
| Registered RGB | 0.9715–1.0112 | 1.0096–1.0237 |
| RGB SfM | 0.9548–0.9885 | 0.9914–1.0045 |

A score of 1.0 is the scene-normalized initial VGGT result. On the released reference, TCO improved the three-scene mean for all nine rerun combinations. On the registered RGB reference, the sign changed with the choice of repeat. The RGB SfM results also depended on which scenes were averaged. With the original runs, omitting Heads gave a two-scene score of 0.877, omitting Chess gave 1.071, and omitting Fire gave 1.007.

The original study captured two additional seed-42 executions on each of Heads and Fire. We scored their four TCO point maps and four half-depth counterparts against all three references, adding 24 CPU evaluations. Each of the nine possible substitutions replaces just the seed-42 result on the two repeated scenes. Chess and the other two seeds per scene stay fixed. The endpoints describe observed reruns on three development scenes. They do not provide a population confidence interval. [Protocol frozen before rescoring](../../manifests/reference_repeat_sensitivity.json) · [Original 108 evaluations](../sensor_extrinsic_v15/RESULTS.md)
