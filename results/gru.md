# GRU with augmentation (Step 21)

2-layer GRU, hidden 128, best epoch 10 of 40 (val window macro-F1 0.861). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.994
- best-F1 threshold (val): 0.895

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.895 | 0.971 | 0.829 | 0.895 | 34 | 1 | 34 | 7 |
| val | best-F1 | 0.918 | 0.886 | 0.951 | 0.908 | 39 | 5 | 30 | 2 |
| val | window macro-F1 | 0.861 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.848 | 1.000 | 0.737 | 0.875 | 28 | 0 | 42 | 10 |
| test_gmdcsa24 | best-F1 | 0.914 | 0.860 | 0.974 | 0.912 | 37 | 6 | 36 | 1 |
| test_gmdcsa24 | window macro-F1 | 0.787 | | | | | | | |
| test_staged | precision-first | 0.760 | 1.000 | 0.613 | 0.789 | 19 | 0 | 26 | 12 |
| test_staged | best-F1 | 0.967 | 1.000 | 0.935 | 0.965 | 29 | 0 | 26 | 2 |
| test_staged | window macro-F1 | 0.875 | | | | | | | |
| test_oops | precision-first | 0.455 | 0.969 | 0.298 | 0.355 | 220 | 7 | 70 | 519 |
| test_oops | best-F1 | 0.822 | 0.952 | 0.724 | 0.717 | 535 | 27 | 50 | 204 |
| test_oops | window macro-F1 | 0.557 | | | | | | | |
