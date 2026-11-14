# Complete system on our own videos (test only)

120 videos recorded by the team, never used for training or tuning. Same code as detect_falls.py (input 640, re-run at 960 if people are small), threshold 0.824 (chosen on validation).

| Group | Videos | Falls | No-fall | Falls caught | False alarms | Precision | Recall | F1 | Accuracy | Fall-start error | Alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 120 | 60 | 60 | 55/60 | 21 | 0.72 | 0.92 | 0.81 | 0.78 | 0.6 s | 1.7 s |
| light: day | 120 | 60 | 60 | 55/60 | 21 | 0.72 | 0.92 | 0.81 | 0.78 | 0.6 s | 1.7 s |
| distance: far | 40 | 20 | 20 | 17/20 | 4 | 0.81 | 0.85 | 0.83 | 0.82 | 0.7 s | 1.9 s |
| distance: near | 80 | 40 | 40 | 38/40 | 17 | 0.69 | 0.95 | 0.80 | 0.76 | 0.5 s | 1.5 s |
| angle: eye | 120 | 60 | 60 | 55/60 | 21 | 0.72 | 0.92 | 0.81 | 0.78 | 0.6 s | 1.7 s |
| set: crowd | 40 | 20 | 20 | 19/20 | 14 | 0.58 | 0.95 | 0.72 | 0.62 | 0.5 s | 1.5 s |
| set: far | 40 | 20 | 20 | 17/20 | 4 | 0.81 | 0.85 | 0.83 | 0.82 | 0.7 s | 1.9 s |
| set: normal | 40 | 20 | 20 | 19/20 | 3 | 0.86 | 0.95 | 0.90 | 0.90 | 0.6 s | 1.5 s |
| 1 person | 80 | 40 | 40 | 36/40 | 7 | 0.84 | 0.90 | 0.87 | 0.86 | 0.7 s | 1.7 s |
| 2+ people | 40 | 20 | 20 | 19/20 | 14 | 0.58 | 0.95 | 0.72 | 0.62 | 0.5 s | 1.5 s |

## Falls by where they started

| Fell from | Falls | Caught |
|---|---|---|
| standing | 60 | 55/60 |

## Every video

| Video | Light | Truth | Alarm | Score | Fall start (true / est.) | Alarm at | Input | Person seen | Notes |
|---|---|---|---|---|---|---|---|---|---|
| normal_000 | day | fall | yes | 1.00 | 0.86 / 2.5 | 3.95 | 640 | 94% | normal |
| far_000 | day | fall | yes | 0.85 | 0.86 / 10.8 | 11.95 | 960 | 67% | far |
| normal_001 | day | fall | yes | 1.00 | 2.27 / 3.0 | 4.45 | 640 | 83% | normal |
| far_001 | day | fall | yes | 1.00 | 2.27 / 3.0 | 4.45 | 960 | 67% | far |
| normal_002 | day | fall | yes | 0.99 | 2.88 / 4.22 | 6.16 | 640 | 100% | normal |
| far_002 | day | fall | yes | 0.99 | 2.88 / 4.55 | 6.16 | 640 | 97% | far |
| normal_003 | day | fall | yes | 0.99 | 9.46 / 10.38 | 11.32 | 640 | 100% | normal |
| far_003 | day | fall | yes | 0.99 | 9.46 / 10.38 | 11.32 | 640 | 100% | far |
| normal_004 | day | fall | yes | 1.00 | 6.13 / 6.4 | 7.6 | 640 | 98% | normal |
| far_004 | day | fall | yes | 1.00 | 6.13 / 6.24 | 7.6 | 640 | 99% | far |
| normal_005 | day | fall | yes | 1.00 | 3.47 / 3.84 | 4.72 | 640 | 100% | normal |
| far_005 | day | fall | yes | 1.00 | 3.47 / 3.76 | 5.2 | 640 | 93% | far |
| normal_006 | day | fall | yes | 1.00 | 0.97 / 1.15 | 2.45 | 640 | 82% | normal |
| far_006 | day | fall | yes | 0.99 | 0.97 / 1.15 | 1.95 | 960 | 98% | far |
| normal_007 | day | fall | yes | 1.00 | 6.2 / 6.37 | 7.58 | 640 | 100% | normal |
| far_007 | day | fall | yes | 1.00 | 6.2 / 6.91 | 7.58 | 640 | 100% | far |
| normal_008 | day | fall | yes | 1.00 | 1.43 / 1.92 | 2.8 | 640 | 97% | normal |
| far_008 | day | fall | yes | 0.99 | 1.43 / 2.0 | 2.8 | 640 | 99% | far |
| normal_009 | day | fall | yes | 1.00 | 3.3 / 3.82 | 4.76 | 640 | 100% | normal |
| far_009 | day | fall | yes | 1.00 | 3.3 / 3.82 | 4.76 | 960 | 100% | far |
| normal_010 | day | fall | yes | 1.00 | 6.96 / 8.24 | 8.98 | 640 | 100% | normal |
| far_010 | day | fall | yes | 1.00 | 6.96 / 8.17 | 8.98 | 640 | 100% | far |
| normal_011 | day | fall | yes | 1.00 | 3.9 / 4.4 | 5.2 | 640 | 99% | normal |
| far_011 | day | fall | yes | 0.92 | 3.9 / 4.48 | 5.2 | 640 | 74% | far |
| normal_012 | day | fall | yes | 1.00 | 2.62 / 3.22 | 4.29 | 640 | 100% | normal |
| far_012 | day | fall | yes | 0.99 | 2.62 / 3.22 | 4.29 | 640 | 100% | far |
| normal_013 | day | fall | yes | 1.00 | 6.53 / 6.83 | 7.92 | 640 | 93% | normal |
| far_013 | day | fall | no | 0.00 | 6.53 /  |  | 960 | 24% | far |
| normal_014 | day | fall | yes | 0.99 | 1.9 / 3.48 | 3.82 | 640 | 99% | normal |
| far_014 | day | fall | yes | 0.99 | 1.9 / 2.75 | 3.82 | 640 | 100% | far |
| normal_015 | day | fall | yes | 1.00 | 5.26 / 3.66 | 5.13 | 640 | 94% | normal |
| far_015 | day | fall | yes | 0.99 | 5.26 / 6.86 | 7.79 | 640 | 94% | far |
| normal_016 | day | fall | yes | 1.00 | 4.84 / 4.75 | 5.92 | 640 | 98% | normal |
| far_016 | day | fall | no | 0.00 | 4.84 /  |  | 960 | 23% | far |
| normal_017 | day | fall | yes | 1.00 | 0.78 / 1.88 | 3.35 | 640 | 100% | normal |
| far_017 | day | fall | yes | 1.00 | 0.78 / 1.88 | 3.35 | 640 | 76% | far |
| normal_018 | day | fall | no | 0.00 | 5.11 /  |  | 640 | 57% | normal |
| far_018 | day | fall | no | 0.00 | 5.11 /  |  | 640 | 59% | far |
| normal_019 | day | fall | yes | 1.00 | 1.7 / 2.33 | 4.27 | 640 | 77% | normal |
| far_019 | day | fall | yes | 1.00 | 1.7 / 3.27 | 4.27 | 640 | 41% | far |
| normal_020 | day | no fall | no | 0.00 |  |  | 640 | 97% | normal |
| far_020 | day | no fall | no | 0.00 |  |  | 640 | 94% | far |
| normal_021 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_021 | day | no fall | no | 0.00 |  |  | 640 | 100% | far |
| normal_022 | day | no fall | no | 0.00 |  |  | 640 | 98% | normal |
| far_022 | day | no fall | no | 0.00 |  |  | 960 | 100% | far |
| normal_023 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_023 | day | no fall | no | 0.00 |  |  | 640 | 100% | far |
| normal_024 | day | no fall | yes | 1.00 |  | 2.41 | 640 | 100% | normal |
| far_024 | day | no fall | yes | 1.00 |  | 2.41 | 640 | 100% | far |
| normal_025 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_025 | day | no fall | yes | 0.89 |  | 10.86 | 640 | 100% | far |
| normal_026 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_026 | day | no fall | no | 0.00 |  |  | 640 | 91% | far |
| normal_027 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_027 | day | no fall | no | 0.00 |  |  | 960 | 79% | far |
| normal_028 | day | no fall | yes | 1.00 |  | 3.0 | 640 | 93% | normal |
| far_028 | day | no fall | yes | 0.99 |  | 2.47 | 960 | 98% | far |
| normal_029 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_029 | day | no fall | no | 0.00 |  |  | 640 | 100% | far |
| normal_030 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_030 | day | no fall | no | 0.00 |  |  | 960 | 100% | far |
| normal_031 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_031 | day | no fall | no | 0.00 |  |  | 640 | 100% | far |
| normal_032 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_032 | day | no fall | no | 0.00 |  |  | 640 | 100% | far |
| normal_033 | day | no fall | yes | 1.00 |  | 3.35 | 640 | 100% | normal |
| far_033 | day | no fall | yes | 0.99 |  | 3.82 | 640 | 100% | far |
| normal_034 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_034 | day | no fall | no | 0.00 |  |  | 640 | 100% | far |
| normal_035 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_035 | day | no fall | no | 0.00 |  |  | 960 | 52% | far |
| normal_036 | day | no fall | no | 0.00 |  |  | 640 | 97% | normal |
| far_036 | day | no fall | no | 0.00 |  |  | 640 | 81% | far |
| normal_037 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_037 | day | no fall | no | 0.00 |  |  | 640 | 78% | far |
| normal_038 | day | no fall | no | 0.00 |  |  | 640 | 100% | normal |
| far_038 | day | no fall | no | 0.00 |  |  | 640 | 97% | far |
| normal_039 | day | no fall | no | 0.00 |  |  | 640 | 99% | normal |
| far_039 | day | no fall | no | 0.00 |  |  | 960 | 100% | far |
| crowd_fall_000 | day | fall | yes | 0.96 | 0.86 / 4.9 | 5.95 | 640 | 100% | crowd |
| crowd_fall_001 | day | fall | yes | 1.00 | 2.27 / 1.8 | 2.45 | 640 | 100% | crowd |
| crowd_fall_002 | day | fall | yes | 0.99 | 2.88 / 5.82 | 6.63 | 640 | 100% | crowd |
| crowd_fall_003 | day | fall | yes | 0.99 | 9.46 / 10.38 | 11.32 | 640 | 100% | crowd |
| crowd_fall_004 | day | fall | yes | 0.98 | 6.13 / 6.32 | 7.6 | 640 | 100% | crowd |
| crowd_fall_005 | day | fall | yes | 1.00 | 3.47 / 3.84 | 4.72 | 640 | 100% | crowd |
| crowd_fall_006 | day | fall | yes | 0.99 | 0.97 / 1.45 | 2.45 | 640 | 100% | crowd |
| crowd_fall_007 | day | fall | yes | 1.00 | 6.2 / 6.37 | 7.11 | 640 | 100% | crowd |
| crowd_fall_008 | day | fall | yes | 1.00 | 1.43 / 3.68 | 3.76 | 640 | 100% | crowd |
| crowd_fall_009 | day | fall | yes | 1.00 | 3.3 / 3.82 | 4.76 | 640 | 100% | crowd |
| crowd_fall_010 | day | fall | no | 0.00 | 6.96 /  |  | 640 | 100% | crowd |
| crowd_fall_011 | day | fall | yes | 1.00 | 3.9 / 4.4 | 5.2 | 640 | 100% | crowd |
| crowd_fall_012 | day | fall | yes | 0.99 | 2.62 / 3.15 | 4.29 | 640 | 100% | crowd |
| crowd_fall_013 | day | fall | yes | 1.00 | 6.53 / 6.83 | 7.92 | 640 | 100% | crowd |
| crowd_fall_014 | day | fall | yes | 1.00 | 1.9 / 2.75 | 3.82 | 640 | 100% | crowd |
| crowd_fall_015 | day | fall | yes | 0.99 | 5.26 / 3.13 | 4.6 | 640 | 100% | crowd |
| crowd_fall_016 | day | fall | yes | 1.00 | 4.84 / 4.75 | 5.92 | 640 | 100% | crowd |
| crowd_fall_017 | day | fall | yes | 1.00 | 0.78 / 1.95 | 3.35 | 640 | 100% | crowd |
| crowd_fall_018 | day | fall | yes | 1.00 | 5.11 / 5.08 | 6.42 | 640 | 100% | crowd |
| crowd_fall_019 | day | fall | yes | 1.00 | 1.7 / 3.2 | 3.8 | 640 | 100% | crowd |
| crowd_adl_000 | day | no fall | yes | 0.99 |  | 1.95 | 640 | 100% | crowd |
| crowd_adl_001 | day | no fall | yes | 0.95 |  | 8.45 | 640 | 100% | crowd |
| crowd_adl_002 | day | no fall | yes | 0.95 |  | 8.45 | 640 | 100% | crowd |
| crowd_adl_003 | day | no fall | yes | 0.84 |  | 10.86 | 640 | 100% | crowd |
| crowd_adl_004 | day | no fall | yes | 0.98 |  | 3.53 | 640 | 100% | crowd |
| crowd_adl_005 | day | no fall | yes | 0.88 |  | 10.86 | 640 | 100% | crowd |
| crowd_adl_006 | day | no fall | yes | 1.00 |  | 3.82 | 640 | 100% | crowd |
| crowd_adl_007 | day | no fall | no | 0.00 |  |  | 640 | 100% | crowd |
| crowd_adl_008 | day | no fall | yes | 1.00 |  | 2.47 | 640 | 100% | crowd |
| crowd_adl_009 | day | no fall | no | 0.00 |  |  | 640 | 100% | crowd |
| crowd_adl_010 | day | no fall | no | 0.00 |  |  | 640 | 100% | crowd |
| crowd_adl_011 | day | no fall | yes | 1.00 |  | 2.47 | 640 | 100% | crowd |
| crowd_adl_012 | day | no fall | no | 0.00 |  |  | 640 | 100% | crowd |
| crowd_adl_013 | day | no fall | yes | 1.00 |  | 2.47 | 640 | 100% | crowd |
| crowd_adl_014 | day | no fall | yes | 0.99 |  | 3.35 | 640 | 100% | crowd |
| crowd_adl_015 | day | no fall | no | 0.00 |  |  | 640 | 100% | crowd |
| crowd_adl_016 | day | no fall | yes | 0.84 |  | 10.8 | 640 | 100% | crowd |
| crowd_adl_017 | day | no fall | no | 0.00 |  |  | 640 | 100% | crowd |
| crowd_adl_018 | day | no fall | yes | 1.00 |  | 2.95 | 640 | 100% | crowd |
| crowd_adl_019 | day | no fall | yes | 0.99 |  | 3.35 | 640 | 100% | crowd |
