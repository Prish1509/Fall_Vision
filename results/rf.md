# Random Forest (Step 19)

300 trees, 460 window-summary features, class-balanced. Thresholds chosen on validation videos only.

- precision-first threshold (val precision >= 0.95): 0.770
- best-F1 threshold (val): 0.618

## Video level (one alarm per video or none)

| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | precision-first | 0.880 | 0.971 | 0.805 | 0.882 | 33 | 1 | 34 | 8 |
| val | best-F1 | 0.916 | 0.905 | 0.927 | 0.908 | 38 | 4 | 31 | 3 |
| val | window macro-F1 | 0.845 | | | | | | | |
| test_gmdcsa24 | precision-first | 0.754 | 1.000 | 0.605 | 0.812 | 23 | 0 | 42 | 15 |
| test_gmdcsa24 | best-F1 | 0.909 | 0.897 | 0.921 | 0.912 | 35 | 4 | 38 | 3 |
| test_gmdcsa24 | window macro-F1 | 0.834 | | | | | | | |
| test_staged | precision-first | 0.873 | 1.000 | 0.774 | 0.877 | 24 | 0 | 26 | 7 |
| test_staged | best-F1 | 0.949 | 1.000 | 0.903 | 0.947 | 28 | 0 | 26 | 3 |
| test_staged | window macro-F1 | 0.848 | | | | | | | |
| test_oops | precision-first | 0.505 | 0.984 | 0.340 | 0.397 | 251 | 4 | 73 | 488 |
| test_oops | best-F1 | 0.691 | 0.978 | 0.535 | 0.567 | 395 | 9 | 68 | 344 |
| test_oops | window macro-F1 | 0.513 | | | | | | | |

## Most useful features

- min:top_speed: 0.026
- mean:top_speed: 0.020
- min:pos_y2: 0.019
- min:pos_y0: 0.019
- min:pos_y4: 0.019
- std:top_height: 0.018
- std:top_speed: 0.018
- min:pos_y5: 0.018
- min:pos_y6: 0.018
- min:pos_y3: 0.017
- min:pos_y1: 0.017
- delta:top_height: 0.017
- delta:pos_y6: 0.015
- max:top_height: 0.013
- delta:pos_y3: 0.012
