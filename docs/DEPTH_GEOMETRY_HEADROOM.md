# Where can camera and depth corrections reduce error?

Before fitting another method, measure what different families of geometric corrections can achieve when given the registered reference at evaluation time. Use the sealed local frozen FP32 and direct-focal point maps, their fixed valid pixel masks, both depth references and the same twelve development conditions as the depth-scale test. Reuse identical tensors and report scene-wise results. This diagnostic spends no cloud budget.

Compute the best attainable root-mean-square 3D error on corresponding pixels for these nested transformations, with all valid pixels carrying equal weight within a condition.

1. One global similarity transform for all five views.
2. One common rotation and scale with independent translations for the five views.
3. Independent rotations and translations with one common scale.
4. Independent similarities, including one scale per view.
5. Independent full affine transforms for the five views.

The first four fits use proper rotations and positive scales. Centering the points per view removes each independent translation. At fixed shared scale the rotations maximize the trace of cross-covariance and follow from a three-dimensional SVD. For a shared scale, sum the optimal traces over the views and divide by the summed centered prediction variance. For independent scales, use this ratio within each view. The affine fit uses ordinary least squares after centering. Report weighted residual sums of squares and RMS errors. Publish per-view sufficient statistics to replay the optimizations without distributing RGB or depth images. Verify monotone training error and known synthetic transformations.

The drop between successive levels measures retrospective capacity conditional on the preceding transformations. Even the view-independent fits can absorb systematic reference errors or depth distortions. The residual after an affine fit includes shape errors and disagreement with the depth reference. These optima provide a lower bound on corresponding-pixel squared error for more restricted transforms. They neither bound nearest-neighbor reconstruction error nor provide an inference rule or a spending decision. Use their magnitude and variation across scenes to choose which local correction is worth testing next.
