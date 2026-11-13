# GRU without augmentation (Step 21)

2-layer GRU, hidden 128, best epoch 8 of 40 (val window macro-F1 0.862). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.990
- best-F1 threshold (val): 0.990

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.909 | 0.972 | 0.854 | 0.908 | 35 | 1 | 34 | 6 |
| val | best-F1 | 0.909 | 0.972 | 0.854 | 0.908 | 35 | 1 | 34 | 6 |
| val | window macro-F1 | 0.862 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.944 | 1.000 | 0.895 | 0.950 | 34 | 0 | 42 | 4 |
| test_gmdcsa24 | best-F1 | 0.944 | 1.000 | 0.895 | 0.950 | 34 | 0 | 42 | 4 |
| test_gmdcsa24 | window macro-F1 | 0.790 | | | | | | | |
| test_staged | precision-first | 0.830 | 1.000 | 0.710 | 0.842 | 22 | 0 | 26 | 9 |
| test_staged | best-F1 | 0.830 | 1.000 | 0.710 | 0.842 | 22 | 0 | 26 | 9 |
| test_staged | window macro-F1 | 0.875 | | | | | | | |
| test_oops | precision-first | 0.566 | 0.970 | 0.399 | 0.445 | 295 | 9 | 68 | 444 |
| test_oops | best-F1 | 0.566 | 0.970 | 0.399 | 0.445 | 295 | 9 | 68 | 444 |
| test_oops | window macro-F1 | 0.531 | | | | | | | |
