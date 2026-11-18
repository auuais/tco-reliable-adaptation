# MASt3R LoRA step selection on fresh rooms

The result missed the registered threshold for a consistent gain over the fixed full step.
Selection improved held depth error by more than 0.05 mm in 4 cases and worsened it by more than 0.05 mm in 0. The remaining 2 valid paired changes fell within that range.

## Paired held depth errors

Every value is mean absolute axial depth error in millimeters after fresh native alignment and a shared metric gauge convention. The final column subtracts the fixed full step from the selected result.

| Room | Adaptation source | Chosen multiplier | Fixed 0 | Fixed 0.25 | Fixed 0.5 | Fixed 1 | Selected | Change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| apt1/living | earlier | 0.5 | 86.868 | 84.002 | 82.431 | 85.765 | 82.431 | -3.335 |
| apt1/living | later | 1 | 84.915 | 86.800 | 83.181 | 82.997 | 82.997 | +0.000 |
| apt2/kitchen | earlier | 0.5 | 58.966 | 61.575 | 54.166 | 55.336 | 54.166 | -1.170 |
| apt2/kitchen | later | 1 | 59.442 | 57.553 | 61.531 | 53.751 | 53.751 | +0.000 |
| office2/5b | earlier | 0 | 76.384 | 76.611 | 77.531 | 77.190 | 76.384 | -0.806 |
| office2/5b | later | 0.5 | 60.756 | 59.000 | 59.030 | 59.276 | 59.030 | -0.246 |

## Equal-room means

| Adaptation source | Fixed 0 | Fixed 0.25 | Fixed 0.5 | Fixed 1 | Selected | Gain vs fixed 1 |
|---|---:|---:|---:|---:|---:|---:|
| earlier | 74.073 | 74.063 | 71.376 | 72.764 | 70.994 | +2.433% |
| later | 68.371 | 67.784 | 67.914 | 65.341 | 65.259 | +0.126% |

The selected multiplier matched the best of the four tested held-out endpoints in 5 of 6 valid cases. Its largest gap to that retrospective benchmark was 0.029 mm.

## Coverage and verification

- All 30 fixed held views retained 53–64 unique pixels after overlap removal. The reserved adaptation selector used 13–16 observations per view.
- The distribution of selected multipliers was 1 at 0, 0 at 0.25, 3 at 0.5, 2 at 1. Rooms improving in both directions by more than 0.05 mm were office2/5b.
- The registered consistency criterion required at least 1% mean gain in both directions, two rooms with gains in both directions, and no case with a regression greater than 1%.
- The independent replay verified 120 public held-view rows and 240 local held and selector view scores, including the metric gauge and fixed overlap masks.
- All 24 native captures finished on the local RTX 3080. GPU jobs used 29.9 summed minutes with 4.49 GiB peak allocated memory. RunPod compute was USD 0.
- The three rooms each contribute two paired adaptation directions over the same RGB views. The results measure first-step sparse depth accuracy.

[Prospective design](../../docs/NATIVE_LORA_STEP_SELECTION.md) · [Execution manifest](../../manifests/native_lora_step_v1.json) · [Pre-score seal](selection_seal.json) · [Scores](held_scores.json) · [Replay receipt](independent_replay.json)

## Reproduce

```powershell
python scripts/native_lora_step_inputs.py intake
python scripts/run_native_lora_step_local.py
python scripts/score_native_lora_step.py seal
python scripts/score_native_lora_step.py score
python scripts/replay_native_lora_step.py --private
```

Run the commands in a fresh working copy of preregistration commit `52a64f1` with the three publisher ZIP archives. The launch uses the registered checkpoint and Docker image. To replay the aggregate decision from Git alone, run `python scripts/replay_native_lora_step.py`.
