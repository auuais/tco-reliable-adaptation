# Can a softer camera weight recover the clean update?

**With the existing detector, even the best weight for each individual case could reduce gradient error by only 18.58%.** This falls below the 20% target. A weight learned from the other two development scenes increased error by 27.14%. These results rule out further tuning of a scalar weight on this detector's whole-camera omission for the tested cases. [Frozen protocol](../../manifests/soft_camera_gradient_bounds.json) · [Case-level results](summary.json)

| Camera-loss policy | Mean normalized gradient error | Improvement over unchanged |
|---|---:|---:|
| Unchanged supplied cameras | 0.5304 | 0.00% |
| Weight learned on two scenes, applied to the flagged camera | 0.6744 | −27.14% |
| Weight learned on two scenes, random camera on the same alerts | 0.5399 | −1.80% |
| Weight learned on two scenes, all camera terms on the same alerts | 0.5696 | −7.39% |
| Best weight using the clean gradient, existing alerts and suspect | 0.4318 | 18.58% |
| Best weight using the clean gradient and true faulty camera | 0.4080 | 23.08% |
| Best weight and best omitted camera using the clean gradient | 0.4031 | 23.99% |

The mean covers 180 synthetic faults on three five-frame development windows. The final three rows use information unavailable during deployment. All three clean windows were also evaluated. On clean Heads, the detector raised a false alert. The weight fitted on Chess and Fire moved the gradient by 0.696 times its clean norm, exceeding the 0.05 clean-distortion limit. On corrupted Heads the error rose from 0.322 to 0.775. On Chess and Fire the fitted rule reduced error by less than 2% each. [Weights, scene breakdown and clean controls](summary.json)

## How the upper bound works

Let $g$ be the gradient from all supplied cameras, $g_{-i}$ the gradient with camera $i$ omitted, and $g_*$ the gradient with clean priors. The mixed objective has gradient $g+\alpha(g_{-i}-g)$ for $0\le\alpha\le1$. With $b=g-g_*$ and $d=g_{-i}-g$, the best weight is $\operatorname{clip}(-b^\top d/(d^\top d),0,1)$. When $d=0$ we use zero. We evaluate this minimum on every case. Repeating it for all five camera choices gives the ceiling for any method that selects one whole camera and one weight on this grid. The 23.99% ceiling leaves open improvements that change which camera is selected. It also leaves open different losses that modify individual measurement components. [Analysis code](../../scripts/soft_camera_gradient_bounds.py)

This secondary analysis reuses 1,098 saved FP32 derivatives with respect to the frozen camera-head outputs. It measures the direction and magnitude of the external camera update. A reconstruction claim requires evidence from full adaptation and dense scoring. No cloud compute or held-out data was used. Replay with `python scripts/soft_camera_gradient_bounds.py verify`. [Parent gradient experiment](../masked_camera_gradients/RESULTS.md)
