# FallVision results summary

All rows: GMDCSA24, rules-based detector, video-level scoring.
Tuning on Subjects 1-2, reported on held-out Subjects 3-4 (80 videos: 38 Fall, 42 ADL).
Add new rows at the END only.

## Day (original GMDCSA24 videos)

| Step | Pose model | Rules | Hold | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|---|---|---|
| 7 | YOLO11n-pose | v2 | 2.0 s | 0.719 | 0.885 | 0.605 | 3 | 15 |
| 9 | YOLO11m-pose | v1 | 1.5 s | 0.835 | 0.805 | 0.868 | 8 | 5 |
| 9 | YOLO11m-pose | v2 | 2.0 s | 0.870 | 0.968 | 0.789 | 1 | 8 |
| 9 | YOLO11m-pose | v2 | 1.5 s | 0.880 | 0.892 | 0.868 | 4 | 5 |

Fall-start error (temporal localization): 0.3-0.4 s. Remaining false alarm: S4 ADL/06 (push-up).
Most misses: falls onto or from a bed or chair.

## Night (simulated: gamma 2.2, gain 0.45, noise 4; mean brightness 29/255)

| Step | Pose model | Night setting | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|---|---|
| 10b | YOLO11m-pose | none (conf 0.25) | 0.582 | 0.941 | 0.421 | 1 | 22 |
| 10b | YOLO11m-pose | conf 0.10 | 0.556 | 0.938 | 0.395 | 1 | 23 |
| 10b | YOLO11m-pose | CLAHE + gamma | 0.348 | 1.000 | 0.211 | 0 | 30 |
| 10b | YOLO11m-pose | denoise + gamma | 0.417 | 1.000 | 0.263 | 0 | 28 |

Finding: darkness causes missed falls, not false alarms. Brightening makes it worse; not used.
