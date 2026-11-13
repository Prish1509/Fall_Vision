# GRU with augmentation (Step 21)

2-layer GRU, hidden 128, best epoch 12 of 40 (val window macro-F1 0.855). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.992
- best-F1 threshold (val): 0.849

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.800 | 0.966 | 0.683 | 0.816 | 28 | 1 | 34 | 13 |
| val | best-F1 | 0.940 | 0.929 | 0.951 | 0.934 | 39 | 3 | 32 | 2 |
| val | window macro-F1 | 0.855 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.712 | 1.000 | 0.553 | 0.787 | 21 | 0 | 42 | 17 |
| test_gmdcsa24 | best-F1 | 0.914 | 0.860 | 0.974 | 0.912 | 37 | 6 | 36 | 1 |
| test_gmdcsa24 | window macro-F1 | 0.785 | | | | | | | |
| test_staged | precision-first | 0.760 | 1.000 | 0.613 | 0.789 | 19 | 0 | 26 | 12 |
| test_staged | best-F1 | 0.933 | 0.966 | 0.903 | 0.930 | 28 | 1 | 25 | 3 |
| test_staged | window macro-F1 | 0.856 | | | | | | | |
| test_oops | precision-first | 0.260 | 0.957 | 0.150 | 0.224 | 111 | 5 | 72 | 628 |
| test_oops | best-F1 | 0.798 | 0.953 | 0.686 | 0.685 | 507 | 25 | 52 | 232 |
| test_oops | window macro-F1 | 0.557 | | | | | | | |
