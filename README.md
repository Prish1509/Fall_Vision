# Fall_Vision: fall detection and localization in video (DS5602)

Detects when a person who is walking, standing or sitting falls, by day or at night, and
localizes them: **who** fell (track ID), **where** (bounding box) and **when** (fall start, alarm time).

## How it works
1. **Pose:** Ultralytics YOLO11m-pose (17 COCO keypoints) + ByteTrack, at about 15 fps.
   If people are small (standing height < 150 px) the video is re-run at input 960.
2. **Tracks:** broken tracks are joined so a person keeps one ID through the fall.
3. **Features:** 92 numbers per frame per person (joint positions and speeds scaled by body
   height, joint confidences, head height and speed, box shape, torso angle).
4. **Model:** a 2-layer GRU reads 2-second windows and outputs normal / falling / fallen.
   Trained on Le2i, CAUCAFall, MCFD and GMDCSA24 with augmentation (mirror, rotation, scale,
   joint noise, night-style hidden joints, dropped frames).
5. **Alarm:** when the fall score passes the threshold chosen on validation; every person is scored.
6. **Second opinion:** a rule-based detector runs on the same skeletons.

## Run it on a video

    python src/detect_falls.py --video clip.mp4 --gru models/gru.pt --threshold 0.849 --out clip_annotated.mp4 --csv clip_falls.csv

Output: printed alarms (person, fall start, alarm time, box), a CSV, and an annotated video.

## Data and splits
| Use | Data |
|---|---|
| Train | Le2i, CAUCAFall, MCFD (OmniFall cross-subject splits), GMDCSA24 Subject 1 |
| Validation (all choices) | Le2i + CAUCAFall val, GMDCSA24 Subject 2 |
| Test (never trained or tuned on) | GMDCSA24 Subjects 3-4, staged test, **OOPS** (818 real-world falls), **our own videos** |

## Results
Final system (threshold chosen on validation, never tuned on test):

| Test set | F1 | Precision | Recall |
|---|---|---|---|
| GMDCSA24 Subjects 3-4 (80 videos) | 0.916 | 0.844 | 1.000 |
| Staged: Le2i, CAUCAFall, MCFD (57) | 0.968 | 0.968 | 0.968 |
| OOPS real-world (818) | 0.875 | 0.945 | 0.815 |
| Our own videos (20 falls) | 0.92 | 1.00 | 0.85 |

The alarm box is on the person who fell in 92 % of checked alarms; the fall start is found to
within about 0.5 s.

See `results/summary.md` for every experiment, including the ideas we tested and rejected
(night brightening, zoom re-check for small people, input 1280, shorter hold time).

## Code (`src/`)
| Step | File |
|---|---|
| Data and splits | `make_splits.py`, `cache_poses.py` |
| Rule baseline | `baseline_fall.py`, `eval_gmdcsa24.py`, `eval_rules_splits.py` |
| Night and far-away tests | `simulate_night.py`, `simulate_far.py`, `night_analysis.py`, `small_person.py` |
| Learned models | `preprocess.py`, `make_windows.py`, `train_rf.py`, `train_gru.py`, `fuse.py` |
| Complete system | `detect_falls.py`, `eval_system.py` |

Videos, pose caches and model weights stay on Kaggle and are never committed.
Pose model: Ultralytics YOLO11 (AGPL-3.0). Datasets: see their own licences (OOPS and Le2i are
for non-commercial research).

## Team
| Member | Main work |
|---|---|
| Member 1 | System, rules, small-person handling, localization, demo |
| Member 2 | Data, splits, pose caches, evaluation, model runs |
| Member 3 | Night handling, learned models, own-video recording, report |
