# Night analysis (Step 16)

Night copy: gamma 2.2, gain 0.45, noise 4 (simulate_night.py); night detection threshold 0.25.

## A. What the pose model sees (largest person)

| Videos | Window | Light | Person detected % | Torso visible % | Mean keypoint conf |
|---|---|---|---|---|---|
| Fall | whole clip | day | 97 | 89 | 0.74 |
| Fall | whole clip | night | 94 | 90 | 0.74 |
| Fall | fall -1 s to +4 s | day | 96 | 88 | 0.73 |
| Fall | fall -1 s to +4 s | night | 93 | 89 | 0.73 |
| ADL | whole clip | day | 100 | 98 | 0.84 |
| ADL | whole clip | night | 98 | 97 | 0.81 |

Falls caught by day but missed at night: 15

| Video | Day: detected / torso % | Night: detected / torso % | Description |
|---|---|---|---|
| Subject 1/Fall/01.mp4 | 64 / 42 | 70 / 65 | Falling from sitting on the chair to the ground. Left side F |
| Subject 1/Fall/09.mp4 | 83 / 82 | 39 / 39 | Standing position to falling. Forward Fall. Full body (head) |
| Subject 2/Fall/02.mp4 | 99 / 95 | 86 / 85 | Right side fall on the ground after walking |
| Subject 2/Fall/08.mp4 | 100 / 100 | 83 / 83 | Reading a newspaper in the sitting position on the ground th |
| Subject 2/Fall/09.mp4 | 99 / 74 | 100 / 100 | Falling backward after walking |
| Subject 2/Fall/12.mp4 | 96 / 57 | 87 / 56 | Drinking water from the water bottle then falling backwards |
| Subject 2/Fall/14.mp4 | 95 / 88 | 49 / 49 | Falling backward on the bed |
| Subject 2/Fall/16.mp4 | 100 / 100 | 98 / 98 | Walking then falling forward |
| Subject 3/Fall/01.mp4 | 100 / 98 | 100 / 100 | falling forward |
| Subject 3/Fall/04.mp4 | 96 / 96 | 91 / 81 | Walking and then falling backward on the floor |
| Subject 3/Fall/14.mp4 | 95 / 83 | 92 / 79 | Falling backward on the floor while removing the shirt and t |
| Subject 3/Fall/17.mp4 | 99 / 68 | 97 / 92 | Reading newspaper then falling Backward on the bed |
| Subject 3/Fall/20.mp4 | 96 / 84 | 78 / 78 | Walking then falling forward on the floor |
| Subject 4/Fall/01.mp4 | 98 / 98 | 76 / 76 | Standing on the bed to falling (LS) on the bed |
| Subject 4/Fall/12.mp4 | 100 / 100 | 98 / 97 | Falling (FW)to the floor from sitting on the bed |

## B. Rule settings (chosen on Subjects 1-2, day + night, no extra false alarms)

Searched: {'hold': [2.0, 1.5, 1.0], 'kp_conf': [0.3, 0.2, 0.15], 'drop_ratio': [0.25, 0.2]}
Chosen: {'hold': 1.5, 'kp_conf': 0.3, 'drop_ratio': 0.25}

| Rules | Light | Held-out Subjects 3-4 |
|---|---|---|
| current v2 | day | F1=0.853  P=0.967  R=0.763  [TP=29 FP=1 TN=41 FN=9] |
| current v2 | night | F1=0.742  P=0.958  R=0.605  [TP=23 FP=1 TN=41 FN=15] |
| chosen | day | F1=0.849  P=0.886  R=0.816  [TP=31 FP=4 TN=38 FN=7] |
| chosen | night | F1=0.812  P=0.903  R=0.737  [TP=28 FP=3 TN=39 FN=10] |
