# Fall_Vision: fall detection and localization in video (DS5602)

Detects when a person who is walking, standing or sitting falls, by day or at night, and
localizes them: **who** fell (track ID), **where** (bounding box) and **when** (fall start, alarm time).

## How it works
1. **Pose:** Ultralytics YOLO11m-pose (17 COCO keypoints) + ByteTrack, at about 15 fps.
   If people are small (standing height < 150 px in a 1280-wide frame) the video is re-run at input 960.
2. **Tracks:** broken tracks are joined so a person keeps one ID through the fall.
3. **Features:** 92 numbers per frame per person (joint positions and speeds scaled by body
   height, joint confidences, head height and speed, box shape, torso angle).
4. **Model:** a 2-layer GRU reads 2-second windows and outputs normal / falling / fallen.
   Trained on Le2i, CAUCAFall, MCFD and GMDCSA24 with augmentation (mirror, rotation, scale,
   joint noise, night-style hidden joints, dropped frames).
5. **Alarm:** when the fall score passes the threshold chosen on validation; every person is scored.
6. **Second opinion:** a rule-based detector runs on the same skeletons.

## Run it
Folders: `app/` (the two programs), `configs/fallvision.json` (every setting; nothing is set inside the
code), `models/` (put `gru_hard.pt` from the Kaggle output here; the YOLO11 pose weights are
downloaded here automatically), `test_videos/` (your videos), `outputs/` (results).
`models/`, `test_videos/` and `outputs/` stay on your computer and are never committed.

    pip install -r requirements.txt
    python app/run_fall.py                             # asks for videos one after another; models load only once
    python app/run_fall.py my_clip.mp4                 # a video in test_videos/: FALL / NO FALL + annotated video in outputs/
    python app/run_fall.py test_videos                 # every video in a folder
    python app/live_fall.py                            # live from the webcam (q to quit)
    python app/live_fall.py my_clip.mp4                # a video shown as it plays, alarms in real time
    python app/live_fall.py http://192.168.1.20:8080/video --serve   # phone camera; watch on a phone browser
    python app/live_fall.py rtsp://camera-address      # CCTV / IP camera

Output: who fell (person number), when (fall start and alarm time) and where (box), printed in the
terminal and drawn on the video (red FALL box); `--save` keeps the live video and alarms in `outputs/`.
Without an NVIDIA GPU the pose model is converted once to OpenVINO, which runs about 4x faster on a CPU.

## Data and splits
| Use | Data |
|---|---|
| Train | Le2i, CAUCAFall, MCFD (OmniFall cross-subject splits), GMDCSA24 Subject 1 |
| Validation (all choices) | Le2i + CAUCAFall val, GMDCSA24 Subject 2 |
| Test (never trained or tuned on) | GMDCSA24 Subjects 3-4, staged test, **OOPS** (818 real-world falls), **our own videos** |

## Results
Final system: `gru_hard.pt`, threshold 0.824, both chosen on validation. One alternative threshold
(0.864, from a harder validation set) was run once on test, lost recall and was not kept; both are
reported in `results/summary.md`.

| Test set | F1 | Precision | Recall | False alarms | Missed falls |
|---|---|---|---|---|---|
| GMDCSA24 Subjects 3-4 (80 videos) | 0.900 | 0.857 | 0.947 | 6 | 2 |
| Staged: Le2i, CAUCAFall, MCFD (57) | 0.968 | 0.968 | 0.968 | 1 | 1 |
| OOPS real-world (818) | 0.860 | 0.954 | 0.783 | 28 | 161 |
| Our own videos (20 falls) | 0.92 | 1.00 | 0.85 | 0 | 3 |

- Own videos: the night fall and the fall from sitting are caught; misses are two far-away people
  and one fall in a group of three.
- Localization: the alarm box is on the person who fell in 92 % of checked alarms; the fall start
  is found to within about 0.4 s and the alarm comes about 1-1.7 s after the fall begins.
- Speed: 29.6 frames/s on a T4 GPU (2x real time); about 10-13 frames/s on a CPU with OpenVINO.

See `results/summary.md` for every experiment, including the ideas we tested and rejected
(night brightening, zoom re-check for small people, input 1280, shorter hold time, low-confidence
tracking, fall-aware track joining, a higher threshold chosen on a harder validation set).

## Code (`src/`)
| Step | File |
|---|---|
| Data and splits | `make_splits.py`, `cache_poses.py` |
| Rule baseline | `baseline_fall.py`, `eval_gmdcsa24.py`, `eval_rules_splits.py` |
| Night and far-away tests | `simulate_night.py`, `simulate_far.py`, `night_analysis.py`, `small_person.py` |
| Learned models | `preprocess.py`, `make_windows.py`, `train_rf.py`, `train_gru.py`, `fuse.py` |
| Complete system | `detect_falls.py`, `eval_system.py`, `eval_own.py`, `speed_test.py` |
| Hard validation set (far / crowd) | `make_hard_val.py` |
| Localization check, own-video check | `loc_sheets.py`, `check_own_videos.py` |
| Run it (`app/`) | `run_fall.py`, `live_fall.py`, `common.py`; settings in `configs/fallvision.json` |

Videos, pose caches and model weights stay on Kaggle and are never committed.
Pose model: Ultralytics YOLO11 (AGPL-3.0). Datasets: see their own licences (OOPS and Le2i are
for non-commercial research).

## Team
| Member | Main work |
|---|---|
| Member 1 | System, rules, small-person handling, localization, demo |
| Member 2 | Data, splits, pose caches, evaluation, model runs |
| Member 3 | Night handling, learned models, own-video recording, report |
