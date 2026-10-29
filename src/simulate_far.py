"""
Make "far-away" copies of a video dataset (Member 1, Step 15).

Each frame is shrunk (e.g. to 20 % of its size) and placed in the middle of a plain grey
frame of the original size, so every person looks as small as someone far from the camera
(about 60-130 px tall instead of 300-650 px). Label files are copied, so the evaluation
script runs unchanged on the far copy.

Usage:
  python src/simulate_far.py --src GMDCSA24 --dst GMDCSA24_far --scale 0.2
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def main():
    ap = argparse.ArgumentParser(description="Make far-away (small-person) copies of videos")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scale", type=float, default=0.2, help="shrink factor (0.2 = 20 %% size)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (0 = all)")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    for f in src.rglob("*"):                          # copy labels and other non-video files
        if f.is_file() and f.suffix.lower() not in VIDEO_EXT:
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)

    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if a.limit:
        videos = videos[:a.limit]
    for i, v in enumerate(videos, 1):
        out = dst / v.relative_to(src).with_suffix(".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            canvas = np.full_like(frame, 110)
            y0, x0 = (H - small.shape[0]) // 2, (W - small.shape[1]) // 2
            canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
            vw.write(canvas)
        vw.release()
        cap.release()
        if i % 10 == 0 or i == len(videos):
            print(f"{i}/{len(videos)} videos shrunk", flush=True)


if __name__ == "__main__":
    main()
