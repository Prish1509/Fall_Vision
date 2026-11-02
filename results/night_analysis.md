# Night analysis (Step 16)

Night copy: gamma 2.2, gain 0.45, noise 4 (simulate_night.py); night detection threshold 0.25.

## A. What the pose model sees (largest person)

| Videos | Window | Light | Person detected % | Torso visible % | Mean keypoint conf |
|---|---|---|---|---|---|
| Fall | whole clip | day | 97 | 91 | 0.75 |
| Fall | whole clip | night | 83 | 73 | 0.69 |
| Fall | fall -1 s to +4 s | day | 96 | 90 | 0.74 |
| Fall | fall -1 s to +4 s | night | 81 | 69 | 0.68 |
| ADL | whole clip | day | 100 | 99 | 0.84 |
| ADL | whole clip | night | 95 | 93 | 0.78 |

Falls caught by day but missed at night: 22

| Video | Day: detected / torso % | Night: detected / torso % | Description |
|---|---|---|---|
| Subject 1/Fall/01.mp4 | 82 / 56 | 68 / 41 | Falling from sitting on the chair to the ground. Left side F |
| Subject 1/Fall/09.mp4 | 94 / 94 | 33 / 33 | Standing position to falling. Forward Fall. Full body (head) |
| Subject 1/Fall/10.mp4 | 100 / 100 | 46 / 46 | Standing position to falling. Forward Fall. Full body (right |
| Subject 2/Fall/02.mp4 | 99 / 97 | 73 / 70 | Right side fall on the ground after walking |
| Subject 2/Fall/09.mp4 | 96 / 79 | 100 / 40 | Falling backward after walking |
| Subject 2/Fall/10.mp4 | 73 / 65 | 32 / 32 | Right side fall then backward |
| Subject 2/Fall/12.mp4 | 97 / 56 | 40 / 33 | Drinking water from the water bottle then falling backwards |
| Subject 2/Fall/13.mp4 | 95 / 92 | 59 / 45 | Falling backward (side view) |
| Subject 3/Fall/02.mp4 | 97 / 97 | 66 / 34 | Falling on the bed (forward) |
| Subject 3/Fall/04.mp4 | 99 / 99 | 77 / 45 | Walking and then falling backward on the floor |
| Subject 3/Fall/06.mp4 | 100 / 100 | 92 / 92 | Falling (backward) on the floor from the sitting position |
| Subject 3/Fall/13.mp4 | 100 / 100 | 98 / 78 | Writing something in a notebook while sitting position on th |
| Subject 3/Fall/14.mp4 | 99 / 99 | 100 / 25 | Falling backward on the floor while removing the shirt and t |
| Subject 3/Fall/17.mp4 | 100 / 89 | 28 / 28 | Reading newspaper then falling Backward on the bed |
| Subject 3/Fall/18.mp4 | 79 / 79 | 31 / 31 | Sitting on the bed; then falling on the floor |
| Subject 3/Fall/20.mp4 | 97 / 96 | 46 / 36 | Walking then falling forward on the floor |
| Subject 4/Fall/01.mp4 | 100 / 98 | 95 / 90 | Standing on the bed to falling (LS) on the bed |
| Subject 4/Fall/04.mp4 | 81 / 81 | 22 / 12 | Walking then falling (forward) view 2 |
| Subject 4/Fall/07.mp4 | 100 / 100 | 30 / 27 | Walking then falling (LS) on the floor |
| Subject 4/Fall/09.mp4 | n/a | n/a | Falling backward on the bed from sitting on the bed |
| Subject 4/Fall/12.mp4 | 100 / 98 | 69 / 66 | Falling (FW)to the floor from sitting on the bed |
| Subject 4/Fall/13.mp4 | 98 / 98 | 98 / 98 | Falling (sideways) on the bed from sitting on the bed |

## B. Rule settings (chosen on Subjects 1-2, day + night, no extra false alarms)

Searched: {'hold': [2.0, 1.5, 1.0], 'kp_conf': [0.3, 0.2, 0.15], 'drop_ratio': [0.25, 0.2]}
Chosen: {'hold': 1.5, 'kp_conf': 0.3, 'drop_ratio': 0.25}

| Rules | Light | Held-out Subjects 3-4 |
|---|---|---|
| current v2 | day | F1=0.870  P=0.968  R=0.789  [TP=30 FP=1 TN=41 FN=8] |
| current v2 | night | F1=0.582  P=0.941  R=0.421  [TP=16 FP=1 TN=41 FN=22] |
| chosen | day | F1=0.880  P=0.892  R=0.868  [TP=33 FP=4 TN=38 FN=5] |
| chosen | night | F1=0.667  P=0.840  R=0.553  [TP=21 FP=4 TN=38 FN=17] |
