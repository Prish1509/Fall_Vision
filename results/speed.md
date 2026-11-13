# Speed of the complete system (Step 26)

20 videos (109 s of video, 1609 frames sampled at 15 fps) on Tesla T4, YOLO11m-pose at input 640. Times include video decoding; the first run was a warm-up.

| Stage | Time | Speed |
|---|---|---|
| Pose + tracking (YOLO11m-pose + ByteTrack) | 54.1 s | 29.7 frames/s |
| Fall stage (track joining, features, GRU) | 0.16 s | 0.10 ms per frame |
| **Whole system** | 54.3 s | **29.6 frames/s** |

Real-time factor: 2.0x (video length / processing time; the system needs 15 frames/s to keep up with live video). Pose estimation takes 100 % of the time; the GRU itself is negligible.

Median per video: 29.4 frames/s.
