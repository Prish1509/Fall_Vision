"""
Make night-simulated copies of a video dataset (Member 3, Step 10).

Every frame is darkened with a gamma curve, scaled down and given sensor noise, which is
roughly what a phone or CCTV camera records in a dark room or street. The folder structure
is kept and label files (CSV etc.) are copied, so the evaluation scripts run unchanged on
the dark copy.

Usage:
  python src/simulate_night.py --src GMDCSA24 --dst GMDCSA24_night
  python src/simulate_night.py --src GMDCSA24 --dst GMDCSA24_night --gain 0.3   # darker
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")


def darken(frame, rng, gamma=2.2, gain=0.45, noise=4.0):
    """gamma > 1 crushes shadows, gain < 1 lowers overall brightness, noise = sensor noise."""
    x = (frame.astype(np.float32) / 255.0) ** gamma * gain * 255.0
    x += rng.normal(0.0, noise, frame.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Make night-simulated copies of videos")
    ap.add_argument("--src", required=True, help="folder with the original videos")
    ap.add_argument("--dst", required=True, help="output folder (same structure)")
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--gain", type=float, default=0.45)
    ap.add_argument("--noise", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (0 = all)")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    rng = np.random.default_rng(0)

    # copy label files and anything else that is not a video
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() not in VIDEO_EXT:
            out = dst / f.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)

    videos = sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if a.limit:
        videos = videos[:a.limit]
    means = []
    for i, v in enumerate(videos, 1):
        out = dst / v.relative_to(src).with_suffix(".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            d = darken(frame, rng, a.gamma, a.gain, a.noise)
            means.append(float(d.mean()))
            vw.write(d)
        vw.release()
        cap.release()
        if i % 10 == 0 or i == len(videos):
            print(f"{i}/{len(videos)} videos darkened", flush=True)
    print(f"Mean brightness of darkened frames: {np.mean(means):.1f} / 255 "
          f"(daylight video is usually 100-130)")


if __name__ == "__main__":
    main()
