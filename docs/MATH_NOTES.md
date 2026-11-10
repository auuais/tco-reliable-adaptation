# Why a lower adaptation loss can produce worse geometry

These are analytical illustrations and design constraints, not experimental results or a novelty claim. They motivate the ablations in RESEARCH_PLAN.md.

## A scalar example with a biased prior

Let the unknown geometry coordinate be x*, a visual estimate be v = x* + e_v, and an external prior be z = x* + b + e_p. Assume independent zero-mean errors with variances s_v^2 and s_p^2; b is systematic prior bias. A weighted estimate is

`x(q) = (1-q) v + q z`, with `0 <= q <= 1`.

Its expected squared error is

`R(q) = (1-q)^2 s_v^2 + q^2 (s_p^2 + b^2)`.

Differentiating gives the optimal mixing coefficient

`q* = s_v^2 / (s_v^2 + s_p^2 + b^2)`.

A covariance-only rule that assumes b=0 can overweight a systematically biased prior. Relative to visual-only risk R(0)=s_v^2, the change is

`R(q)-R(0) = -2 q s_v^2 + q^2 (s_v^2+s_p^2+b^2)`.

Thus a positive prior weight helps only while `q < 2 s_v^2 / (s_v^2+s_p^2+b^2)` under these assumptions. A prior may have low random noise but high systematic error; calling it confident does not make it useful. Correlated or biased visual errors change this calculation, which is why real-data validation is necessary.

## Why fewer steps can help even if the objective keeps decreasing

Consider the quadratic objective

`L(x) = A/2 (x-v)^2 + B/2 (x-z)^2`, with A,B>0.

Its minimizer has prior fraction `q_inf = B/(A+B)`. Starting at x_0=v, gradient descent with `0 < eta*(A+B) < 1` produces

`x_t = v + q_t (z-v)`,

`q_t = [1-(1-eta*(A+B))^t] q_inf`.

The objective decreases monotonically, but q_t moves toward a potentially overconfident prior mixture. The expected reconstruction risk may first decrease and then increase if q_inf exceeds the risk-optimal region. A smaller learning rate or fewer steps can therefore improve geometry without changing the loss at all. This is a simple regularization explanation that the proposed method must beat; it is not evidence of a new algorithm.

## Testing a constraint's marginal benefit

For one small SGD step, write the without-group update as `d_without = -eta*g_v` and the with-group update as `d_with = -eta*(g_v + w*g_p)`. For a separate validation objective V, the first-order difference is

`V(a+d_without) - V(a+d_with) ~= eta*w*grad(V)^T g_p`.

A positive inner product suggests the prior's gradient locally helps the chosen validation objective. It does not show that V matches true geometry, and the approximation may fail for large steps, clipping or nonlinear behavior. With Adam, the two updates must be computed from identical optimizer moments and counters; the simple SGD expression is only intuition. Common RNG states reduce one source of comparison noise but do not eliminate the observed numerical execution variability.

The actual experiment must compare with-group and without-group updates from the same snapshot. Comparing with-group only against the original model mixes together the effects of visual adaptation and prior adaptation. Include probe cost in every efficiency comparison.

## What images cannot certify

If validation uses only image reprojection and is invariant under a global similarity transform T, then `V(T G)=V(G)`. It cannot distinguish two reconstructions differing solely in an unobservable scale or world-frame gauge. No acceptance rule based only on this V can certify that an absolute scale prior is correct. Low parallax creates additional weakly constrained directions.

This limits the proposed contribution: first study observable relative geometry and shape. Handle global gauge conventions explicitly, and require independent metric information and suitable metrics before claiming scale improvement. Sim(3)-aligned error removes scale information from evaluation as well.

## Trust regions are controls, not correctness guarantees

A bound on geometry displacement can limit how far one update changes a prediction, but does not prove that the change is beneficial. Under a local linearization, `delta_G ~= J_G d`, a geometry trust region uses `d^T J_G^T W J_G d <= epsilon^2`. Use Jacobian-vector products or a small projected subspace; a dense matrix over roughly 3.1 million adapter parameters is infeasible.

Rotation, translation, focal and pointmap residuals have different units. Whitening and robust penalties must use documented scales. Camera constraints sharing a reference frame are correlated, so independent scalar residual weights are an approximation. Keep both a simple diagonal baseline and an explicit constraint-group design before introducing a more elaborate uncertainty model.

The practical evidence needed is a better accuracy/runtime tradeoff and fewer harmful updates on frozen held-out data, beyond lower learning rates, fewer steps, robust loss and elementary gating. None of the algebra above establishes that result in advance.

## Numerical precision is part of the camera constraint

The [known-rotation audit](../results/rotation_precision_v6/RESULTS.md) found zero gradients under the released autocast path at all tested 0.1-3 degree errors, while float32 pose arithmetic retained the expected gradient. This is a controlled numerical result, not yet a reconstruction improvement.

For a proper relative rotation, `cos(theta) = (trace(R)-1)/2`. Near identity, `1-cos(theta) ~= theta^2/2`. If the matrix-product rounding perturbs this cosine by order `u`, angles of order `sqrt(u)` can become difficult to distinguish through this trace formulation. Computing the final inverse cosine in float32 cannot restore already rounded matrix products. The actual threshold is orientation/operator dependent; the tested 3-degree observation is not a universal constant.

Small angular errors can still matter geometrically. In an ideal rectified two-view model, disparity is `d=f*B/Z`. For a disparity error `delta_d`,

`Z_hat/Z = 1 / (1 + delta_d/d)`.

A small image-center orientation error approximately induces `delta_d ~= f*delta_theta`, giving the first-order sensitivity `delta_Z/Z ~= -(Z/B)*delta_theta` when the relative disparity error is small. Thus low parallax amplifies an angular error. As an illustrative calculation, with `Z/B=10` and a pixel-disparity perturbation corresponding to one degree, the exact disparity formula gives roughly 15% lower or 21% higher depth depending on the sign. These numbers are a hypothetical local model, not measurements on our sequences. Similarity-aligned evaluation can remove a uniform scale error; depth-dependent distortion is a separate issue.

The immediate control is inexpensive float32 pose-loss arithmetic, tested separately from loss redesign. Only after that baseline is evaluated should we consider whether an uncertainty-aware or observability-aware update rule adds value. Otherwise an apparent reliability-method gain might merely compensate for a numerical defect.

## Shared-gradient proposals

Repeated renderer/backbone evaluations can produce different gradients even with matched RNG states. Capture weighted component gradients once, then construct each group subset from the same stored arrays and original optimizer moments. This removes independent recomputation of shared components from the within-step comparison; it does not remove error in the captured gradients or ensure either update is beneficial. Recompute clipping and the complete Adam step for each subset. Parameter-update subtraction is not equivalent because Adam and clipping are nonlinear.

The [transaction primitive](../scripts/shared_gradient_proposals.py) implements state rollback and has CPU/CUDA tests. It does not yet implement a model controller. The fixed-match validation baseline also demonstrates why sufficient image support must be checked explicitly: its matches do not connect every sampled view in Heads and Chess. An unsupported comparison should abstain rather than manufacture a confident decision.
