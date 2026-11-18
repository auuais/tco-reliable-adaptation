# LoRA with calibration compensation regularization

The verified D1 probe found that refitting the five scales reduced the norm of the training-path query-residual change by 28.86% and 31.23% in the two original spatial folds. The FP32 inference contrasts retained 98.84% and 97.76% of the fixed-scale changes. Both paths reduced query loss in both folds. In training fold 1 the final-state full derivative along the observed output change was −0.001407, compared with −0.080198 when the scales were detached at the same forward state. The training query observations were available to the adapter, so held-source scores are still needed. [D1 result](../results/lora_profiled_d1_v1/README.md)

Test whether regularizing the component that requires a calibration response improves held-source reconstruction, relative to ordinary damping with the same local curvature trace. This is the single proposed scientific redesign. D1 measured the response in one case and the screen rejected the original accuracy hypothesis. Those observations motivate a controlled test without asserting that attenuation caused the failed screen.

For each fold retain the original positive ridge `lambda`, support weights, gauges and query normalizers prepared from pristine `z0`. Write `A=A(z)`, `Q=Q(z)`, `b,d` for fixed targets and `s(z)=solve(A.T A+lambda I,A.T b+lambda 1)`. Use the graph-zero P3 method. At every step use the fixed pristine fit `s0=s(z0)` for that fold. Define the least-squares query loss and the penalty as

```text
J(z) = 0.5 ||Q(z)s(z)-d||²
C(z) = Q(z)(s(z)-s0)
L(z) = J(z) + 0.5 ||C(z)||²
```

`C` is exactly the finite compensation term measured by D1. With fixed `s0`, its derivative is `dot(Q)(s-s0)+Q dot(s)`. Include both terms. At every state the same-state detached control evaluates identical scales, residuals and penalty. It detaches `s` in both terms of the objective and retains the direct dependence on `Q`. Both penalties and their gradients vanish at pristine initialization. Record state and gradient equality through the first shared full-gradient step for P3, ordinary damping and the compensation arm. The detached arm can differ on the first step.

For the ordinary damping control let `a_i,q_i,b_i` be the active support-column, query-column and support targets for camera `i` at pristine initialization. Define `r_i=a_i*s0_i-b_i` and `h_i=a_i.T a_i+lambda`. Work in the whitened support perturbation coordinates `x_i=(a_i(z)-a_i)*s0_i`. Differentiating the fitted scale gives `dot(s_i)=-(a_i+r_i/s0_i).T dot(x_i)/h_i`. Therefore the squared Frobenius norm of the query compensation Jacobian for camera `i` is

```text
trace_i = ||q_i||² ||a_i + r_i/s0_i||² / h_i²
kappa_i = trace_i / number_of_support_rows_i
L_ordinary(z) = J(z) + 0.5 sum_i kappa_i ||x_i||²
```

This fixes `kappa_i` with pristine inputs before updates. Its isotropic support penalty matches the per-camera trace of the compensation penalty Hessian at pristine initialization. The structured penalty concentrates that curvature along the scale-response direction. Equal trace does not imply equal achieved penalty or equal step length. Record both and retain actual per-fold `||C||` as a mechanism readout. The unit penalty weight is preselected because both losses are in the same whitened query-residual units. No tuning sweep is planned. [Math and fixtures](../scripts/profiled_compensation_penalty.py)

| Arm | Training objective or control |
|---|---|
| P0 | Fresh frozen model with the common endpoint calibration |
| P1 | Fresh matched D with the original schedule |
| P3 | Fresh full-gradient graph-zero profiled query loss |
| R0 | P3 plus trace-matched ordinary damping |
| R1 | P3 plus compensation regularization with full gradients |
| R2 | Same forward objective as R1 with scales detached |

Use every original scene and direction, Heads, Chess and Fire with both source packets assigned in separate runs. This is 30 fresh trained states plus six frozen controls. Retain the original rank 4, alpha 1, eight macro-steps, Adam learning rate, 20 directed pairs per step, BF16 decoder with FP32 heads during training, pristine confidence fusion and alternating support folds. Reuse the two-pass replay. The regularizer uses only the original allowed support/query rows, and needs no extra model forward at a step. All arms use the common fresh FP32 native KDP plus calibration endpoint. Seal all outputs before same-protocol held-source evaluation. Budget USD 0 rental. Estimate local compute from the original screen during registration and log actual wall time and memory.

Only R1 may be nominated. Require the original 1% held-source MAE gate over P1 in **each** direction with at least two of the same scenes improving, plus no numerical failures. Also require lower directional means than P3, R0 and R2 in both directions. Report all pair effects and per-view penalties, including cases with reduced compensation but worse accuracy. A missed gate ends the redesign screen and triggers the bounded negative evidence package. A passing screen releases matched densification and repeatability checks under the original rules. A mechanism claim additionally needs the matched controls and repeat stability. The five-percent surface gate still controls rental authorization.

The differentiable graph-zero bridge is implemented and passed four synthetic fixtures, including the curvature trace, exact initial P3 objective and gradient, and same-state detachment. Integrate it into the versioned two-pass runner. Use the no-gradient D1 bridge for reference arithmetic. Verify saved replay equivalence, same-state forward equality, original-weight integrity, and parameter finite differences under the original engineering runtime before full training. Then commit the versioned execution and evaluation manifests with the named arms, exact inventory and scoring rules. Count the redesign when scientific optimization starts. The math fixtures are prepared and the scientific run is still pending.
