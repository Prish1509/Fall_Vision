# GRU without augmentation (Step 21)

2-layer GRU, hidden 128, best epoch 7 of 40 (val window macro-F1 0.851). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.994
- best-F1 threshold (val): 0.985

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.909 | 0.972 | 0.854 | 0.908 | 35 | 1 | 34 | 6 |
| val | best-F1 | 0.911 | 0.947 | 0.878 | 0.908 | 36 | 2 | 33 | 5 |
| val | window macro-F1 | 0.851 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.899 | 1.000 | 0.816 | 0.912 | 31 | 0 | 42 | 7 |
| test_gmdcsa24 | best-F1 | 0.960 | 0.973 | 0.947 | 0.963 | 36 | 1 | 41 | 2 |
| test_gmdcsa24 | window macro-F1 | 0.779 | | | | | | | |
| test_staged | precision-first | 0.784 | 1.000 | 0.645 | 0.807 | 20 | 0 | 26 | 11 |
| test_staged | best-F1 | 0.893 | 1.000 | 0.806 | 0.895 | 25 | 0 | 26 | 6 |
| test_staged | window macro-F1 | 0.878 | | | | | | | |
| test_oops | precision-first | 0.441 | 0.968 | 0.286 | 0.344 | 211 | 7 | 70 | 528 |
| test_oops | best-F1 | 0.603 | 0.964 | 0.438 | 0.477 | 324 | 12 | 65 | 415 |
| test_oops | window macro-F1 | 0.548 | | | | | | | |
