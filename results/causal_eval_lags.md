# Submitted (causal) system: evaluation with decision lag (Step 28b)

infer.py engine on whole videos, frame by frame, current and past frames only. One alarm per video is counted if the highest collapse score reaches the threshold.

Choice on VALIDATION only, rule fixed in advance: decision lag with the best average precision (ties -> shorter lag), then threshold with the best F1 (ties -> fewer false alarms).

| Decision lag | Validation average precision |
|---|---|
| 0.0 s | 0.9734 |
| 0.5 s | 0.9661 |
| 1.0 s | 0.9509 |
| 1.5 s | 0.9229 |

Chosen: decision lag **0.0 s**, threshold **0.499**.

| Split | Setting | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Median alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val | chosen: lag 0.0 s, thr 0.499 | 76 | 0.953 | 0.911 | 1.000 | 0.947 | 41 | 4 | 31 | 0 | 1.6 s |
| val | config: lag 0.0 s, thr 0.824 | 76 | 0.952 | 0.930 | 0.976 | 0.947 | 40 | 3 | 32 | 1 | 1.7 s |
| test_gmdcsa24 | chosen: lag 0.0 s, thr 0.499 | 80 | 0.842 | 0.842 | 0.842 | 0.850 | 32 | 6 | 36 | 6 | 2.3 s |
| test_gmdcsa24 | config: lag 0.0 s, thr 0.824 | 80 | 0.827 | 0.838 | 0.816 | 0.838 | 31 | 6 | 36 | 7 | 2.6 s |
| test_staged | chosen: lag 0.0 s, thr 0.499 | 57 | 0.968 | 0.968 | 0.968 | 0.965 | 30 | 1 | 25 | 1 | 1.3 s |
| test_staged | config: lag 0.0 s, thr 0.824 | 57 | 0.968 | 0.968 | 0.968 | 0.965 | 30 | 1 | 25 | 1 | 1.6 s |
| test_oops | chosen: lag 0.0 s, thr 0.499 | 818 | 0.827 | 0.945 | 0.735 | 0.721 | 545 | 32 | 45 | 196 | 1.1 s |
| test_oops | config: lag 0.0 s, thr 0.824 | 818 | 0.775 | 0.956 | 0.652 | 0.658 | 483 | 22 | 55 | 258 | 1.3 s |
| test_own | chosen: lag 0.0 s, thr 0.499 | 20 | 0.667 | 1.000 | 0.500 | 0.500 | 10 | 0 | 0 | 10 | 2.1 s |
| test_own | config: lag 0.0 s, thr 0.824 | 20 | 0.519 | 1.000 | 0.350 | 0.350 | 7 | 0 | 0 | 13 | 2.1 s |
