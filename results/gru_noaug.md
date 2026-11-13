# GRU without augmentation (Step 21)

2-layer GRU, hidden 128, best epoch 9 of 40 (val window macro-F1 0.853). Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.991
- best-F1 threshold (val): 0.983

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.923 | 0.973 | 0.878 | 0.921 | 36 | 1 | 34 | 5 |
| val | best-F1 | 0.925 | 0.949 | 0.902 | 0.921 | 37 | 2 | 33 | 4 |
| val | window macro-F1 | 0.853 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.944 | 1.000 | 0.895 | 0.950 | 34 | 0 | 42 | 4 |
| test_gmdcsa24 | best-F1 | 0.944 | 1.000 | 0.895 | 0.950 | 34 | 0 | 42 | 4 |
| test_gmdcsa24 | window macro-F1 | 0.791 | | | | | | | |
| test_staged | precision-first | 0.877 | 0.962 | 0.806 | 0.877 | 25 | 1 | 25 | 6 |
| test_staged | best-F1 | 0.915 | 0.964 | 0.871 | 0.912 | 27 | 1 | 25 | 4 |
| test_staged | window macro-F1 | 0.884 | | | | | | | |
| test_oops | precision-first | 0.547 | 0.976 | 0.380 | 0.430 | 281 | 7 | 70 | 458 |
| test_oops | best-F1 | 0.620 | 0.968 | 0.456 | 0.494 | 337 | 11 | 66 | 402 |
| test_oops | window macro-F1 | 0.552 | | | | | | | |
