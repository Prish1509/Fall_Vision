# Rules v2 on every split

Rule baseline (v2, hold 2.0 s) on the cached YOLO11m-pose skeletons; video-level scoring identical to the learned models.

| Split | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| train | 0.759 | 0.946 | 0.633 | 0.688 | 209 | 12 | 84 | 121 |
| val | 0.816 | 0.886 | 0.756 | 0.816 | 31 | 4 | 31 | 10 |
| test_staged | 0.741 | 0.870 | 0.645 | 0.754 | 20 | 3 | 23 | 11 |
| test_gmdcsa24 | 0.870 | 0.968 | 0.789 | 0.887 | 30 | 1 | 41 | 8 |
| test_oops | 0.384 | 0.973 | 0.239 | 0.304 | 177 | 5 | 72 | 564 |
