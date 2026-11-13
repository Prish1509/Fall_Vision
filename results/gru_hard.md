# GRU with augmentation, hard-negative weight 3.0 (Step 21)

2-layer GRU, hidden 128, best epoch 8 of 40 (val window macro-F1 0.865). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.996
- best-F1 threshold (val): 0.973

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.849 | 0.969 | 0.756 | 0.855 | 31 | 1 | 34 | 10 |
| val | best-F1 | 0.927 | 0.927 | 0.927 | 0.921 | 38 | 3 | 32 | 3 |
| val | window macro-F1 | 0.865 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.794 | 1.000 | 0.658 | 0.838 | 25 | 0 | 42 | 13 |
| test_gmdcsa24 | best-F1 | 0.947 | 0.947 | 0.947 | 0.950 | 36 | 2 | 40 | 2 |
| test_gmdcsa24 | window macro-F1 | 0.795 | | | | | | | |
| test_staged | precision-first | 0.735 | 1.000 | 0.581 | 0.772 | 18 | 0 | 26 | 13 |
| test_staged | best-F1 | 0.912 | 1.000 | 0.839 | 0.912 | 26 | 0 | 26 | 5 |
| test_staged | window macro-F1 | 0.862 | | | | | | | |
| test_oops | precision-first | 0.397 | 0.974 | 0.249 | 0.314 | 184 | 5 | 72 | 555 |
| test_oops | best-F1 | 0.763 | 0.955 | 0.635 | 0.642 | 469 | 22 | 55 | 270 |
| test_oops | window macro-F1 | 0.558 | | | | | | | |
