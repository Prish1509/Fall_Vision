# Fall_Vision: fall detection and localization in video (DS5602)

Detects when a person who is walking, running, standing or sitting falls, by day or at night,
and localizes them: who fell (track ID), where (bounding box) and when (fall start, alert time).
Pose model: Ultralytics YOLO11-pose (AGPL-3.0) with ByteTrack tracking.

## Folders
| Folder | Contents |
|---|---|
| `src/` | All Python code |
| `notebooks/` | Kaggle notebooks we ran |
| `configs/` | Settings and thresholds |
| `splits/` | Train / validation / test lists (test = OOPS + own videos, never trained on) |
| `results/` | Small CSVs of scores and `summary.md` |
| `team/` | Setup check-in files |

Videos, pose caches and model weights stay on Kaggle and are never committed.

## Team
| Member | Owns |
|---|---|
| Member 1 | System: pose + tracking, rules, small-person handling, motion cue, localization, demo |
| Member 2 | Data and evaluation: datasets, splits, pose caches, preprocessing, all scoring |
| Member 3 | Night handling and learned models: night tests, augmentation, Random Forest, GRU |

## Workflow
1. Run the start-of-session cell in Kaggle (clones or pulls this repo).
2. Edit only the files you own; run on the Kaggle GPU.
3. Pull, then commit and push your own files.
