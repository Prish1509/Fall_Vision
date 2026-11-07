# GRU with augmentation (Step 21)

2-layer GRU, hidden 128, best epoch 14 of 40 (val window macro-F1 0.862). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.992
- best-F1 threshold (val): 0.835

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.817 | 0.967 | 0.707 | 0.829 | 29 | 1 | 34 | 12 |
| val | best-F1 | 0.929 | 0.907 | 0.951 | 0.921 | 39 | 4 | 31 | 2 |
| val | window macro-F1 | 0.862 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.712 | 1.000 | 0.553 | 0.787 | 21 | 0 | 42 | 17 |
| test_gmdcsa24 | best-F1 | 0.914 | 0.860 | 0.974 | 0.912 | 37 | 6 | 36 | 1 |
| test_gmdcsa24 | window macro-F1 | 0.794 | | | | | | | |
| test_staged | precision-first | 0.681 | 1.000 | 0.516 | 0.737 | 16 | 0 | 26 | 15 |
| test_staged | best-F1 | 0.967 | 1.000 | 0.935 | 0.965 | 29 | 0 | 26 | 2 |
| test_staged | window macro-F1 | 0.863 | | | | | | | |
| test_oops | precision-first | 0.302 | 0.971 | 0.179 | 0.251 | 132 | 4 | 73 | 607 |
| test_oops | best-F1 | 0.794 | 0.954 | 0.679 | 0.680 | 502 | 24 | 53 | 237 |
| test_oops | window macro-F1 | 0.558 | | | | | | | |
