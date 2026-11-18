# Mathematical validation for the next LoRA study

The design trains scene adapters using sensor predictions after a small calibration solve. The proposed model experiments are specified in the [mathematical plan](../../docs/FROZEN_LORA_ACCURACY_PLAN.md) and [experiment manifest](../../manifests/lora_profiled_design_v1.json).

The executed check uses five synthetic cameras, ten noncoplanar landmarks, thirty support depths, twenty separate query depths and thirty matched landmarks. Its prescribed network-output direction stands in for a LoRA output derivative.

| Quantity | Observed value |
|---|---:|
| Full derivative through fitted scales | 3.666001177 |
| Central finite difference | 3.666001196 |
| Derivative with fitted scales detached | 7.094145105 |
| Absolute full-gradient discrepancy | 1.89e-8 |
| Maximum scale derivative discrepancy | 4.14e-10 |
| Normal-equation residual | 2.27e-13 |

Checks also verify the ridge Schur-complement identity, its positive semidefinite residual operator, its non-idempotence under nonzero regularization, the unregularized gauge reparameterization, and detection of unsupported view scales despite a solvable ridge system. A disconnected match graph is reported separately from sensor-scale identifiability.

Run the CPU checks from the repository root.

```sh
python scripts/check_lora_profiled_gradient.py --verify
```

NumPy is the only mathematical dependency. The script verifies a prescribed output derivative. A production LoRA gradient bridge and real reconstruction-accuracy experiments remain the next steps. This design work used USD 0 of rental compute.
