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
