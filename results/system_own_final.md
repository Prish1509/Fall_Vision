# Complete system on our own videos (test only)

20 videos recorded by the team, never used for training or tuning. Same code as detect_falls.py (input 640, re-run at 960 if people are small), threshold 0.824 (chosen on validation).

| Group | Videos | Falls | No-fall | Falls caught | False alarms | Precision | Recall | F1 | Accuracy | Fall-start error | Alarm delay |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 20 | 20 | 0 | 17/20 | 0 | 1.00 | 0.85 | 0.92 | 0.85 | 0.4 s | 1.0 s |
| light: day | 19 | 19 | 0 | 16/19 | 0 | 1.00 | 0.84 | 0.91 | 0.84 | 0.4 s | 1.0 s |
| light: night | 1 | 1 | 0 | 1/1 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.3 s | 1.5 s |
| distance: far | 4 | 4 | 0 | 2/4 | 0 | 1.00 | 0.50 | 0.67 | 0.50 | 0.8 s | 1.8 s |
| distance: near | 16 | 16 | 0 | 15/16 | 0 | 1.00 | 0.94 | 0.97 | 0.94 | 0.4 s | 1.0 s |
| angle: eye | 20 | 20 | 0 | 17/20 | 0 | 1.00 | 0.85 | 0.92 | 0.85 | 0.4 s | 1.0 s |
| 1 person | 12 | 12 | 0 | 10/12 | 0 | 1.00 | 0.83 | 0.91 | 0.83 | 0.3 s | 1.0 s |
| 2+ people | 8 | 8 | 0 | 7/8 | 0 | 1.00 | 0.88 | 0.93 | 0.88 | 0.5 s | 1.0 s |

## Falls by where they started

| Fell from | Falls | Caught |
|---|---|---|
| hanging | 1 | 1/1 |
| jumping | 5 | 5/5 |
| riding | 2 | 2/2 |
| running | 4 | 4/4 |
| sitting | 1 | 1/1 |
| sleeping | 1 | 1/1 |
| standing | 4 | 1/4 |
| walking | 2 | 2/2 |

## Every video

| Video | Light | Truth | Alarm | Score | Fall start (true / est.) | Alarm at | Input | Person seen | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 001 | day | fall | yes | 0.99 | 2.0 / 0.53 | 2.47 | 640 | 100% |  |
| 002 | day | fall | no | 0.00 | 9.5 /  |  | 640 | 96% |  |
| 003 | day | fall | yes | 0.98 | 2.0 / 2.07 | 2.47 | 640 | 87% |  |
| 004 | day | fall | yes | 0.98 | 3.0 / 3.0 | 4.07 | 640 | 100% |  |
| 005 | day | fall | yes | 0.94 | 2.0 / 2.2 | 4.07 | 640 | 97% |  |
| 006 | day | fall | yes | 0.96 | 2.0 / 1.6 | 3.0 | 640 | 100% |  |
| 007 | day | fall | yes | 1.00 | 2.0 / 1.67 | 3.0 | 640 | 62% |  |
| 008 | day | fall | yes | 0.97 | 3.0 / 4.5 | 5.92 | 640 | 84% |  |
| 009 | day | fall | no | 0.00 | 4.0 /  |  | 960 | 54% |  |
| 010 | day | fall | yes | 0.99 | 2.0 / 1.73 | 2.47 | 640 | 88% |  |
| 011 | day | fall | no | 0.00 | 3.0 /  |  | 640 | 100% |  |
| 012 | day | fall | yes | 0.99 | 3.0 / 3.53 | 4.07 | 640 | 99% |  |
| 013 | day | fall | yes | 0.83 | 2.0 / 1.47 | 2.47 | 640 | 100% |  |
| 014 | night | fall | yes | 0.99 | 1.0 / 0.73 | 2.47 | 640 | 67% |  |
| 015 | day | fall | yes | 0.99 | 2.0 / 3.4 | 5.13 | 640 | 100% |  |
| 016 | day | fall | yes | 0.99 | 2.0 / 2.47 | 3.0 | 640 | 100% |  |
| 017 | day | fall | yes | 0.98 | 2.0 / 3.2 | 4.07 | 640 | 96% |  |
| 018 | day | fall | yes | 0.99 | 3.0 / 2.53 | 3.53 | 640 | 100% |  |
| 019 | day | fall | yes | 0.99 | 2.0 / 1.87 | 2.47 | 640 | 100% |  |
| 020 | day | fall | yes | 0.99 | 2.0 / 1.67 | 3.0 | 640 | 80% |  |
