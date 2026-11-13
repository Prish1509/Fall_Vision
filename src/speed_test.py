"""
Speed of the complete system (Step 26): can it run in real time?

For each video: time the pose + tracking stage (YOLO11m-pose + ByteTrack on frames sampled at
15 fps, including video decoding) and the fall stage (track joining, features, GRU on every person)
separately. Reported: processed frames per second for each stage and for the whole system, and the
real-time factor = video length / processing time (above 1 means faster than real time).
The first video is run once beforehand to warm up the GPU and is not counted.

Output: results/speed.md

Usage:
  python src/speed_test.py --videos /kaggle/input/.../fallvision-own-videos --gru models/gru.pt --n 10
"""

import argparse
import glob
import os
import time

import numpy as np
import torch

from baseline_fall import extract_tracks
from detect_falls import alarms_for_video, load_gru

EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp")


def main():
    ap = argparse.ArgumentParser(description="Time the complete fall detector")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--gru", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.849)
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=10, help="number of videos to time")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU only"
    grus = [load_gru(g, device) for g in a.gru]
    files = sorted(p for p in glob.glob(os.path.join(a.videos, "**", "*"), recursive=True)
                   if p.lower().endswith(EXTS))[:a.n]
    print(f"{len(files)} videos on {gpu}, input {a.imgsz}", flush=True)

    data, _ = extract_tracks(files[0], a.model, 15.0, a.imgsz)          # warm-up, not counted
    alarms_for_video(data, grus, None, device, a.threshold)

    rows = []
    for p in files:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        data, _ = extract_tracks(p, a.model, 15.0, a.imgsz)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        alarms_for_video(data, grus, None, device, a.threshold)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        n = len(data["frames"])
        secs = n / data["fps"] if data["fps"] else 0.0
        rows.append((os.path.basename(p), n, secs, t1 - t0, t2 - t1))
        print(f"  {os.path.basename(p):<12} {n:>4} frames  pose+track {n / (t1 - t0):6.1f} fps  "
              f"fall stage {(t2 - t1) * 1000:6.0f} ms  real-time factor {secs / (t2 - t0):.1f}x", flush=True)

    frames = sum(r[1] for r in rows)
    video_s = sum(r[2] for r in rows)
    pose_s = sum(r[3] for r in rows)
    fall_s = sum(r[4] for r in rows)
    lines = ["# Speed of the complete system (Step 26)", "",
             f"{len(rows)} videos ({video_s:.0f} s of video, {frames} frames sampled at 15 fps) on {gpu}, "
             f"YOLO11m-pose at input {a.imgsz}. Times include video decoding; the first run was a warm-up.", "",
             "| Stage | Time | Speed |", "|---|---|---|",
             f"| Pose + tracking (YOLO11m-pose + ByteTrack) | {pose_s:.1f} s | {frames / pose_s:.1f} frames/s |",
             f"| Fall stage (track joining, features, GRU) | {fall_s:.2f} s | "
             f"{1000 * fall_s / max(frames, 1):.2f} ms per frame |",
             f"| **Whole system** | {pose_s + fall_s:.1f} s | **{frames / (pose_s + fall_s):.1f} frames/s** |", "",
             f"Real-time factor: {video_s / (pose_s + fall_s):.1f}x (video length / processing time; the system "
             f"needs 15 frames/s to keep up with live video). Pose estimation takes "
             f"{100 * pose_s / (pose_s + fall_s):.0f} % of the time; the GRU itself is negligible.",
             "", f"Median per video: {np.median([r[1] / (r[3] + r[4]) for r in rows]):.1f} frames/s."]
    os.makedirs("results", exist_ok=True)
    open("results/speed.md", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[4:]))
    print("\nSaved results/speed.md")


if __name__ == "__main__":
    main()
