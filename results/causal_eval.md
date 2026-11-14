# Submitted (causal) system: evaluation (Step 28)

infer.py engine on whole videos, frame by frame, current and past frames only. One alarm per video is counted if the highest collapse score reaches the threshold.

Threshold chosen on VALIDATION only (highest F1, ties -> fewer false alarms): **1.01**. Earlier threshold (whole-video evaluation): 0.824.

| Split | Threshold | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Median alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | chosen 1.01 | 18 | 0.000 | 0.000 | 0.000 | 1.000 | 0 | 0 | 18 | 0 | n/a |
| val | earlier 0.824 | 18 | 0.000 | 0.000 | 0.000 | 0.944 | 0 | 1 | 17 | 0 | n/a |
