# Submitted (causal) system: evaluation (Step 28)

infer.py engine on whole videos, frame by frame, current and past frames only. One alarm per video is counted if the highest collapse score reaches the threshold.

Threshold chosen on VALIDATION only (highest F1, ties -> fewer false alarms): **0.8573**. Earlier threshold (whole-video evaluation): 0.824.

| Split | Threshold | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Median alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | chosen 0.8573 | 76 | 0.965 | 0.932 | 1.000 | 0.961 | 41 | 3 | 32 | 0 | 1.6 s |
| val | earlier 0.824 | 76 | 0.965 | 0.932 | 1.000 | 0.961 | 41 | 3 | 32 | 0 | 1.6 s |
| test_gmdcsa24 | chosen 0.8573 | 80 | 0.875 | 0.833 | 0.921 | 0.875 | 35 | 7 | 35 | 3 | 1.9 s |
| test_gmdcsa24 | earlier 0.824 | 80 | 0.875 | 0.833 | 0.921 | 0.875 | 35 | 7 | 35 | 3 | 1.9 s |
| test_staged | chosen 0.8573 | 57 | 0.968 | 0.968 | 0.968 | 0.965 | 30 | 1 | 25 | 1 | 1.5 s |
| test_staged | earlier 0.824 | 57 | 0.968 | 0.968 | 0.968 | 0.965 | 30 | 1 | 25 | 1 | 1.5 s |
| test_oops | chosen 0.8573 | 818 | 0.855 | 0.958 | 0.772 | 0.763 | 572 | 25 | 52 | 169 | 1.1 s |
| test_oops | earlier 0.824 | 818 | 0.859 | 0.955 | 0.780 | 0.768 | 578 | 27 | 50 | 163 | 1.1 s |
| test_own | chosen 0.8573 | 20 | 0.857 | 1.000 | 0.750 | 0.750 | 15 | 0 | 0 | 5 | 1.0 s |
| test_own | earlier 0.824 | 20 | 0.857 | 1.000 | 0.750 | 0.750 | 15 | 0 | 0 | 5 | 1.0 s |
