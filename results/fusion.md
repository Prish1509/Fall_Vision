# Model combination chosen on validation (Step 22 preparation)

Every setting below is scored on VALIDATION videos only; the best one (validation F1, ties ->
fewer false alarms) is then reported once on the test splits.

| Models | Score | Val F1 (best-F1 thr) | Val P | Val R | Val FP | Val F1 (precision-first thr) | Val FP |
|---|---|---|---|---|---|---|---|
| gru | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.817 | 1 |
| rf+gru | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.880 | 1 |
| rf+gru+gru_noaug | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.895 | 1 |
| rf+gru | falling | 0.925 | 0.949 | 0.902 | 2 | 0.895 | 1 |
| gru+gru_noaug | fall+fallen | 0.918 | 0.886 | 0.951 | 5 | 0.865 | 1 |
| rf | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.880 | 1 |
| gru | falling | 0.916 | 0.905 | 0.927 | 4 | 0.895 | 1 |
| rf+gru_noaug | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.880 | 1 |
| gru_noaug | fall+fallen | 0.911 | 0.947 | 0.878 | 2 | 0.909 | 1 |
| gru+gru_noaug | falling | 0.909 | 0.972 | 0.854 | 1 | 0.909 | 1 |
| rf+gru_noaug | falling | 0.902 | 0.902 | 0.902 | 4 | 0.833 | 1 |
| rf+gru+gru_noaug | falling | 0.900 | 0.923 | 0.878 | 3 | 0.849 | 1 |
| gru_noaug | falling | 0.897 | 0.946 | 0.854 | 2 | 0.865 | 1 |
| rf | falling | 0.892 | 0.881 | 0.902 | 5 | 0.849 | 1 |

**Chosen:** models `gru`, score `fall+fallen`, best-F1 threshold 0.835, precision-first threshold 0.992

## Chosen setting on every split

| Split | Threshold | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | best-F1 | 0.929 | 0.907 | 0.951 | 0.921 | 39 | 4 | 31 | 2 |
| val | precision-first | 0.817 | 0.967 | 0.707 | 0.829 | 29 | 1 | 34 | 12 |
| test_gmdcsa24 | best-F1 | 0.914 | 0.860 | 0.974 | 0.912 | 37 | 6 | 36 | 1 |
| test_gmdcsa24 | precision-first | 0.712 | 1.000 | 0.553 | 0.787 | 21 | 0 | 42 | 17 |
| test_staged | best-F1 | 0.967 | 1.000 | 0.935 | 0.965 | 29 | 0 | 26 | 2 |
| test_staged | precision-first | 0.681 | 1.000 | 0.516 | 0.737 | 16 | 0 | 26 | 15 |
| test_oops | best-F1 | 0.794 | 0.954 | 0.679 | 0.680 | 502 | 24 | 53 | 237 |
| test_oops | precision-first | 0.302 | 0.971 | 0.179 | 0.251 | 132 | 4 | 73 | 607 |
