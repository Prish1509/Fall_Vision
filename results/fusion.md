# Model combination chosen on validation (Step 22 preparation)

Every setting below is scored on VALIDATION videos only; the best one (validation F1, ties ->
fewer false alarms) is then reported once on the test splits.

| Models | Score | Val F1 (best-F1 thr) | Val P | Val R | Val FP | Val F1 (precision-first thr) | Val FP |
|---|---|---|---|---|---|---|---|
| gru | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.800 | 1 |
| rf+gru_hard | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.880 | 1 |
| gru+gru_hard | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.833 | 1 |
| rf+gru+gru_hard | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.895 | 1 |
| gru+gru_noaug+gru_hard | fall+fallen | 0.940 | 0.929 | 0.951 | 3 | 0.849 | 1 |
| rf+gru | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.895 | 1 |
| gru+gru_noaug | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.849 | 1 |
| rf+gru+gru_noaug | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.895 | 1 |
| rf+gru+gru_noaug+gru_hard | fall+fallen | 0.929 | 0.907 | 0.951 | 4 | 0.895 | 1 |
| gru_hard | fall+fallen | 0.927 | 0.927 | 0.927 | 3 | 0.849 | 1 |
| rf+gru_noaug+gru_hard | fall+fallen | 0.927 | 0.927 | 0.927 | 3 | 0.895 | 1 |
| gru | falling | 0.925 | 0.949 | 0.902 | 2 | 0.895 | 1 |
| rf+gru | falling | 0.925 | 0.949 | 0.902 | 2 | 0.909 | 1 |
| rf+gru+gru_hard | falling | 0.923 | 0.973 | 0.878 | 1 | 0.923 | 1 |
| rf | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.880 | 1 |
| rf+gru_noaug | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.880 | 1 |
| rf+gru_hard | falling | 0.916 | 0.905 | 0.927 | 4 | 0.909 | 1 |
| gru+gru_hard | falling | 0.916 | 0.905 | 0.927 | 4 | 0.895 | 1 |
| gru_noaug+gru_hard | fall+fallen | 0.916 | 0.905 | 0.927 | 4 | 0.880 | 1 |
| gru_hard | falling | 0.911 | 0.947 | 0.878 | 2 | 0.895 | 1 |
| rf+gru_noaug+gru_hard | falling | 0.911 | 0.947 | 0.878 | 2 | 0.909 | 1 |
| rf+gru+gru_noaug+gru_hard | falling | 0.911 | 0.947 | 0.878 | 2 | 0.909 | 1 |
| gru_noaug | fall+fallen | 0.909 | 0.972 | 0.854 | 1 | 0.909 | 1 |
| rf+gru+gru_noaug | falling | 0.909 | 0.972 | 0.854 | 1 | 0.909 | 1 |
| gru_noaug | falling | 0.907 | 1.000 | 0.829 | 0 | 0.895 | 1 |
| rf+gru_noaug | falling | 0.905 | 0.884 | 0.927 | 5 | 0.849 | 1 |
| gru+gru_noaug+gru_hard | falling | 0.900 | 0.923 | 0.878 | 3 | 0.895 | 1 |
| gru+gru_noaug | falling | 0.897 | 0.946 | 0.854 | 2 | 0.895 | 1 |
| gru_noaug+gru_hard | falling | 0.897 | 0.946 | 0.854 | 2 | 0.880 | 1 |
| rf | falling | 0.892 | 0.881 | 0.902 | 5 | 0.849 | 1 |

**Chosen:** models `gru`, score `fall+fallen`, best-F1 threshold 0.849, precision-first threshold 0.992

## Chosen setting on every split

| Split | Threshold | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| val | best-F1 | 0.940 | 0.929 | 0.951 | 0.934 | 39 | 3 | 32 | 2 |
| val | precision-first | 0.800 | 0.966 | 0.683 | 0.816 | 28 | 1 | 34 | 13 |
| test_gmdcsa24 | best-F1 | 0.914 | 0.860 | 0.974 | 0.912 | 37 | 6 | 36 | 1 |
| test_gmdcsa24 | precision-first | 0.712 | 1.000 | 0.553 | 0.787 | 21 | 0 | 42 | 17 |
| test_staged | best-F1 | 0.933 | 0.966 | 0.903 | 0.930 | 28 | 1 | 25 | 3 |
| test_staged | precision-first | 0.760 | 1.000 | 0.613 | 0.789 | 19 | 0 | 26 | 12 |
| test_oops | best-F1 | 0.798 | 0.953 | 0.686 | 0.685 | 507 | 25 | 52 | 232 |
| test_oops | precision-first | 0.260 | 0.957 | 0.150 | 0.224 | 111 | 5 | 72 | 628 |
