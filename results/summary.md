# FallVision results summary

All rows: GMDCSA24, rules-based detector, video-level scoring.
Tuning on Subjects 1-2, reported on held-out Subjects 3-4 (80 videos: 38 Fall, 42 ADL).
Add new rows at the END only.

## Day (original GMDCSA24 videos)

| Step | Pose model | Rules | Hold | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|---|---|---|
| 7 | YOLO11n-pose | v2 | 2.0 s | 0.719 | 0.885 | 0.605 | 3 | 15 |
| 9 | YOLO11m-pose | v1 | 1.5 s | 0.835 | 0.805 | 0.868 | 8 | 5 |
| 9 | YOLO11m-pose | v2 | 2.0 s | 0.870 | 0.968 | 0.789 | 1 | 8 |
| 9 | YOLO11m-pose | v2 | 1.5 s | 0.880 | 0.892 | 0.868 | 4 | 5 |

Fall-start error (temporal localization): 0.3-0.4 s. Remaining false alarm: S4 ADL/06 (push-up).
Most misses: falls onto or from a bed or chair.

## Night (simulated: gamma 2.2, gain 0.45, noise 4; mean brightness 29/255)

| Step | Pose model | Night setting | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|---|---|
| 10b | YOLO11m-pose | none (conf 0.25) | 0.582 | 0.941 | 0.421 | 1 | 22 |
| 10b | YOLO11m-pose | conf 0.10 | 0.556 | 0.938 | 0.395 | 1 | 23 |
| 10b | YOLO11m-pose | CLAHE + gamma | 0.348 | 1.000 | 0.211 | 0 | 30 |
| 10b | YOLO11m-pose | denoise + gamma | 0.417 | 1.000 | 0.263 | 0 | 28 |

Finding: darkness causes missed falls, not false alarms. Brightening makes it worse; not used.

## Far away (simulated: frames shrunk to 20 %, people about 60-130 px tall)

| Step | Pose model | Small-person setting | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|---|---|
| 15b | YOLO11m-pose | none (input 640) | 0.632 | 0.947 | 0.474 | 1 | 20 |
| 15b | YOLO11m-pose | zoom re-check below 160 px | 0.632 | 0.947 | 0.474 | 1 | 20 |
| 15b | YOLO11m-pose | input 960 | 0.742 | 0.958 | 0.605 | 1 | 15 |
| 15b | YOLO11m-pose | input 1280 | 0.719 | 0.885 | 0.605 | 3 | 15 |

Finding: input 960 helps far-away people with no extra false alarms; 1280 adds false alarms;
the zoom re-check changed nothing (misses come from lost detections, not weak joints).

## Input size 960 vs 640 (Step 15c, held-out Subjects 3-4, rules v2 hold 2.0)

| Step | Video | Input | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|---|---|
| 15c | normal | 960 | 0.853 | 0.967 | 0.763 | 1 | 9 |
| 15c | night (simulated) | 960 | 0.519 | 0.875 | 0.368 | 2 | 24 |

Decision: input 640 is the default (960 is worse for normal and night video);
960 is used only when people are small (it helps far-away people: F1 0.632 -> 0.742).

## Why falls are missed at night and far away (Steps 16 and 15d; largest person, 1 s before to 4 s after the fall)

| Step | Video | Input | Person detected % | Torso readable % |
|---|---|---|---|---|
| 16 | normal | 640 | 96 | 90 |
| 16 | night (simulated) | 640 | 81 | 69 |
| 15d | normal | 960 | 96 | 88 |
| 15d | far away (20 % size) | 960 | 93 | 89 |

Night: the person is lost in about 1 of 5 frames and the torso is unreadable in about 1 of 3.
Far away at input 960: people are seen almost as well as at normal size, so the misses come from the rules.
Rule search (chosen on Subjects 1-2): hold 1.5 s raises recall (night 0.42 -> 0.55) but adds 3 false
alarms on held-out Subjects 3-4, so rules v2 keeps hold 2.0. A lower joint threshold did not help.
Plan: fix night and far-away in the learned model (night-darkened and shrunk training skeletons) and the motion cue.

## Learned models vs rules (Steps 19, 21, 22; video level, threshold chosen on validation)

Same cached YOLO11m-pose skeletons and the same test videos for every model. FP = false alarms on
non-fall videos, FN = missed falls. None of these test sets was used for training or for any choice.

| Model | GMDCSA24 S3-4: F1 / P / R | FP | FN | Staged test: F1 / P / R | FP | FN | OOPS: F1 / P / R | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Rules v2 (hold 2.0 s) | 0.870 / 0.968 / 0.789 | 1 | 8 | 0.741 / 0.870 / 0.645 | 3 | 11 | 0.384 / 0.973 / 0.239 | 5 | 564 |
| Random Forest | 0.909 / 0.897 / 0.921 | 4 | 3 | 0.949 / 1.000 / 0.903 | 0 | 3 | 0.691 / 0.978 / 0.535 | 9 | 344 |
| GRU, no augmentation | 0.944 / 1.000 / 0.895 | 0 | 4 | 0.830 / 1.000 / 0.710 | 0 | 9 | 0.566 / 0.970 / 0.399 | 9 | 444 |
| **GRU with augmentation (chosen)** | 0.914 / 0.860 / 0.974 | 6 | 1 | 0.933 / 0.966 / 0.903 | 1 | 3 | **0.798** / 0.953 / 0.686 | 25 | 232 |
| GRU with augmentation + hard negatives x3 | 0.947 / 0.947 / 0.947 | 2 | 2 | 0.912 / 1.000 / 0.839 | 0 | 5 | 0.763 / 0.955 / 0.635 | 22 | 270 |

Findings:
- Every learned model beats the rules on every test set; on real-world OOPS the chosen GRU catches
  507 falls against the rules' 177.
- Augmentation (mirror, rotation, scale, joint noise, night-style hidden joints, dropped frames)
  matters most for unseen real-world video: OOPS recall 0.399 -> 0.686.
- Extra weight on "looks like a fall" windows (lying down, exercise, sitting down...) cuts false
  alarms on GMDCSA24 from 6 to 2, at some cost in OOPS recall. Validation (76 videos, 35 non-falls)
  could not separate the two models (both F1 0.940), so the simpler one was kept, as decided
  before looking at the test sets.
- Error analysis: 9 of the 12 worst missed OOPS falls scored 0 because the person is detected in
  only 6-21 % of frames - a pose-detection failure, not a classifier failure. 77 of 818 OOPS clips
  carry no "fall" label although they come from the falls folder, so OOPS precision is understated.

## Final system: complete pipeline on whole videos (Steps 23 and 25)

detect_falls.py end to end (YOLO11m-pose + ByteTrack, every tracked person scored by the GRU,
alarm at threshold 0.849 chosen on validation). A video alarms if any person does.

| Test set | Videos | F1 | Precision | Recall | False alarms | Missed falls | Fall-start error | Alarm delay |
|---|---|---|---|---|---|---|---|---|
| GMDCSA24 Subjects 3-4 | 80 | 0.916 | 0.844 | **1.000** | 7 | 0 | 0.4 s | 1.4 s |
| Staged (Le2i, CAUCAFall, MCFD) | 57 | **0.968** | 0.968 | 0.968 | 1 | 1 | 0.2 s | 1.4 s |
| OOPS (real-world) | 818 | 0.875 | 0.945 | 0.815 | 35 | 137 | 0.5 s | 1.0 s |
| **Our own videos** | 20 | 0.92 | 1.00 | 0.85 | 0 | 3 | 0.4 s | 0.5 s |

Our own videos (results/system_own.md): 20 falls recorded by the team (19 day, 1 night), never used
for training or tuning. 17 of 20 caught, including the night fall, the fall from sitting, and 7 of 8
falls with two or more people in view. Near: 15/16 caught; far away: 2/4. The three misses are two
far-away falls (one with the person seen in only 46 % of frames) and one fall with three people in
view. The set has no no-fall clips, so false alarms are measured on GMDCSA24 and the staged set.

Localization (results/localization.md): for 36 sampled alarms (12 per test set) the alarm frame
with the system's box was checked by eye. The box is on the person who fell in 33/36 (92 %):
GMDCSA24 12/12, staged 12/12, OOPS 9/12. Temporal localization: the estimated fall start is within
0.2-0.5 s of the labelled start (median), and the alarm comes 0.5-1.4 s after the fall begins.

Error analysis (results/error_analysis.md; 12 missed falls and 6 false alarms watched by hand):
misses are mostly small or far-away people (4), camera motion (3) and very fast falls (2), plus a
backward fall from sitting and a fall while mounting a horse; false alarms are lying down on purpose
(2), exercise, a non-person "fall", one very fast movement and one label error.

Small-person check fixed (after the error analysis): the standing height is now measured in a
1280-wide frame (the frame size the 150-px limit was chosen on), so a far person in a 1920x1080
or portrait phone video is recognised as small and re-run at input 960. Unchanged for 1280x720
video; low-resolution video no longer re-runs needlessly. Re-run once on the 20 own videos
(results/system_own_fix.md): the far video 009 now runs at 960 (person seen in 54 % of frames
instead of 46 %) but is still missed; all other videos are unchanged, so the own-video result
stays 17/20 falls caught, 0 false alarms. Kept because it is correct for any video resolution.

Threshold check on the hard validation set (Step 27): F2 (missed falls count twice) chose 0.864
(hard validation: same 56/60 falls caught, false alarms 22 -> 19). Run once on the test sets
(results/system_final864.md, system_own_final864.md) it lost recall without reducing false alarms:
GMDCSA24 36 -> 35 caught (FP 6 -> 6), staged 30 -> 29 (FP 1 -> 1), OOPS 580 -> 570 (FP 28 -> 27),
own videos 17 -> 16 (the fall from sitting, score 0.83, was lost). The validation gain came mostly from
the synthetic crowds and did not carry over to real video, so the final system keeps 0.824
(chosen on validation by fuse.py). This choice was made after seeing the test results and is
reported as such; both rows are shown. No further values were tried on test data.

## FINAL system (after the model retrain was found; replaces the numbers in "Final system" above)

The final evaluation run retrained the GRUs (Save & Run All), so the model and threshold were
re-chosen on validation for those weights: fuse.py picked `gru_hard.pt`, threshold 0.824
(results/fusion.md). The numbers in the "Final system" section above came from `gru.pt` with a
threshold chosen for earlier weights and are kept only as a record. Final results
(results/system_final.md, results/system_own_final.md):

| Test set | Videos | F1 | Precision | Recall | False alarms | Missed falls | Fall-start error | Alarm delay |
|---|---|---|---|---|---|---|---|---|
| GMDCSA24 Subjects 3-4 | 80 | 0.900 | 0.857 | 0.947 | 6 | 2 | 0.5 s | 1.7 s |
| Staged (Le2i, CAUCAFall, MCFD) | 57 | 0.968 | 0.968 | 0.968 | 1 | 1 | 0.3 s | 1.3 s |
| OOPS (real-world) | 818 | 0.860 | 0.954 | 0.783 | 28 | 161 | 0.4 s | 1.1 s |
| Our own videos | 20 | 0.92 | 1.00 | 0.85 | 0 | 3 | 0.4 s | 1.0 s |

Own videos: 17/20 caught (night fall and fall from sitting caught); misses: 002 (tracked 96 % of the
time but the model never scores it as a fall), 009 (small person lost by the detector at the moment
of the fall), 011 (three people; the faller's track breaks at the fall). Localization (92 % correct
person) was checked on the earlier model's alarms with the same box logic.

Step 27, ideas tested on a hard validation set (validation videos made far away and put into
crowds; results/system_hv_*.md) and not kept: low-confidence tracking (far 18 -> 17 caught,
false alarms 22 -> 21) and fall-aware track joining (no change). The hard set showed that false
alarms rise when several people are in view (crowd no-fall videos: 14/20 alarmed, partly because
the same no-fall clips are reused in several crowds).

Live mode (live_fall.py): the same model, features, windows, smoothing and threshold, applied
while the video runs, for any number of people; tested to give the same alarm as run_fall.py on the
same clip. All settings are in configs/fallvision.json.
