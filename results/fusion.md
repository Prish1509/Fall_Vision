# Model combination chosen on validation (Step 22 preparation)

Every setting below is scored on VALIDATION videos only; the best one (validation F1, ties ->
fewer false alarms) is then reported once on the test splits.

| Models | Score | Val F1 (best-F1 thr) | Val P | Val R | Val FP | Val F1 (precision-first thr) | Val FP |
|---|---|---|---|---|---|---|---|
| gru_hard | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.746 | 1 |
| gru+gru_hard | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.833 | 1 |
| rf+gru_hard | falling | 0.937 | 0.974 | 0.902 | 1 | 0.937 | 1 |
| rf+gru | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.880 | 1 |
| rf+gru_hard | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.880 | 1 |
| gru_noaug+gru_hard | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.909 | 1 |
| rf+gru+gru_hard | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.880 | 1 |
| rf+gru_noaug+gru_hard | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.895 | 1 |
| rf+gru+gru_noaug+gru_hard | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.895 | 1 |
| gru_hard | falling | 0.927 | 0.927 | 0.927 | 3 | 0.909 | 1 |
| gru_noaug | fall+fallen | 0.925 | 0.949 | 0.902 | 2 | 0.923 | 1 |
| rf+gru+gru_hard | falling | 0.923 | 0.973 | 0.878 | 1 | 0.923 | 1 |
| gru | fall+fallen | 0.918 | 0.886 | 0.951 | 5 | 0.895 | 1 |
| gru+gru_noaug | fall+fallen | 0.918 | 0.886 | 0.951 | 5 | 0.895 | 1 |
| rf+gru+gru_noaug | fall+fallen | 0.918 | 0.886 | 0.951 | 5 | 0.895 | 1 |
| gru+gru_noaug+gru_hard | fall+fallen | 0.918 | 0.886 | 0.951 | 5 | 0.880 | 1 |
| rf | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.880 | 1 |
| rf+gru_noaug | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.909 | 1 |
| gru+gru_hard | falling | 0.916 | 0.905 | 0.927 | 4 | 0.895 | 1 |
| rf+gru_noaug+gru_hard | falling | 0.916 | 0.905 | 0.927 | 4 | 0.909 | 1 |
| rf+gru_noaug | falling | 0.911 | 0.947 | 0.878 | 2 | 0.833 | 1 |
| gru_noaug+gru_hard | falling | 0.911 | 0.947 | 0.878 | 2 | 0.895 | 1 |
| rf+gru+gru_noaug+gru_hard | falling | 0.909 | 0.972 | 0.854 | 1 | 0.909 | 1 |
| rf+gru+gru_noaug | falling | 0.905 | 0.884 | 0.927 | 5 | 0.895 | 1 |
| gru+gru_noaug+gru_hard | falling | 0.905 | 0.884 | 0.927 | 5 | 0.895 | 1 |
| rf+gru | falling | 0.902 | 0.902 | 0.902 | 4 | 0.880 | 1 |
| gru_noaug | falling | 0.895 | 0.971 | 0.829 | 1 | 0.895 | 1 |
| gru+gru_noaug | falling | 0.895 | 0.971 | 0.829 | 1 | 0.895 | 1 |
| gru | falling | 0.894 | 0.864 | 0.927 | 6 | 0.880 | 1 |
| rf | falling | 0.892 | 0.881 | 0.902 | 5 | 0.849 | 1 |

**Chosen:** models `gru_hard`, score `fall+fallen`, best-F1 threshold 0.824, precision-first threshold 0.998

## Chosen setting on every split

| Split | Threshold | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | best-F1 | 0.940 | 0.929 | 0.951 | 0.934 | 39 | 3 | 32 | 2 |
| val | precision-first | 0.746 | 0.962 | 0.610 | 0.776 | 25 | 1 | 34 | 16 |
| test_gmdcsa24 | best-F1 | 0.900 | 0.857 | 0.947 | 0.900 | 36 | 6 | 36 | 2 |
| test_gmdcsa24 | precision-first | 0.593 | 1.000 | 0.421 | 0.725 | 16 | 0 | 42 | 22 |
| test_staged | best-F1 | 0.951 | 0.967 | 0.935 | 0.947 | 29 | 1 | 25 | 2 |
| test_staged | precision-first | 0.708 | 1.000 | 0.548 | 0.754 | 17 | 0 | 26 | 14 |
| test_oops | best-F1 | 0.810 | 0.952 | 0.705 | 0.701 | 521 | 26 | 51 | 218 |
| test_oops | precision-first | 0.179 | 0.973 | 0.099 | 0.181 | 73 | 2 | 75 | 666 |
