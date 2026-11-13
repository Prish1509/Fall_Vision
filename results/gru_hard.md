# GRU with augmentation, hard-negative weight 3.0 (Step 21)

2-layer GRU, hidden 128, best epoch 16 of 40 (val window macro-F1 0.864). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.998
- best-F1 threshold (val): 0.824

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.746 | 0.962 | 0.610 | 0.776 | 25 | 1 | 34 | 16 |
| val | best-F1 | 0.940 | 0.929 | 0.951 | 0.934 | 39 | 3 | 32 | 2 |
| val | window macro-F1 | 0.864 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.593 | 1.000 | 0.421 | 0.725 | 16 | 0 | 42 | 22 |
| test_gmdcsa24 | best-F1 | 0.900 | 0.857 | 0.947 | 0.900 | 36 | 6 | 36 | 2 |
| test_gmdcsa24 | window macro-F1 | 0.803 | | | | | | | |
| test_staged | precision-first | 0.708 | 1.000 | 0.548 | 0.754 | 17 | 0 | 26 | 14 |
| test_staged | best-F1 | 0.951 | 0.967 | 0.935 | 0.947 | 29 | 1 | 25 | 2 |
| test_staged | window macro-F1 | 0.884 | | | | | | | |
| test_oops | precision-first | 0.179 | 0.973 | 0.099 | 0.181 | 73 | 2 | 75 | 666 |
| test_oops | best-F1 | 0.810 | 0.952 | 0.705 | 0.701 | 521 | 26 | 51 | 218 |
| test_oops | window macro-F1 | 0.565 | | | | | | | |
