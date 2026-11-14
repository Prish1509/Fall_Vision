"""
Hard validation videos: far-away people and crowds, made ONLY from validation videos (Step 27).

Our own test videos showed two weak spots: very small (far) people and one person falling among
others. Fixing them by looking at those test videos would only fit those videos, so every fix is
judged on these validation-only sets instead, and the test sets are run once at the end.

  normal the validation videos unchanged (a change must not make ordinary video worse)
  far    each validation video shrunk to --far-scale of its size and placed at a random spot
         on a grey canvas of the original size (people about 50-150 px tall)
  crowd  each validation FALL video placed next to --crowd-k other validation NO-FALL videos
         (2x2 grid on one 1920x1080 frame), so one person falls while others walk, sit or lie down;
         plus as many no-fall crowds (only no-fall videos in the grid) to count false alarms

Output: <out>/<name>.mp4 and <out>/labels.csv in the same format as splits/own_videos_labels.csv,
so eval_own.py scores them unchanged.

Usage:
  python src/make_hard_val.py --gmd-root GMD --omnifall-root OMNI --out /kaggle/working/hard_val
  python src/eval_own.py --videos /kaggle/working/hard_val --labels /kaggle/working/hard_val/labels.csv \
                         --gru GRU --threshold 0.824 --name val_hard
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import cv2
import numpy as np

from cache_poses import video_file

FALL = 1


def fall_starts(label_dir):
    start = {}
    for fname in os.listdir(label_dir):
        if fname.endswith(".csv"):
            for r in csv.DictReader(open(os.path.join(label_dir, fname), newline="")):
                if int(r["label"]) == FALL:
                    key = (r["dataset"], r["path"])
                    start[key] = min(start.get(key, 1e9), float(r["start"]))
    return start


def read_all(path, fps_out=15.0, max_w=None):
    """Frames resampled to about fps_out (the system samples at 15 fps anyway); optionally shrunk
    to at most max_w pixels wide to keep memory low."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / fps_out)))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if max_w and f.shape[1] > max_w:
                f = cv2.resize(f, (max_w, int(f.shape[0] * max_w / f.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(f)
        i += 1
    cap.release()
    return frames, fps / step


def write(path, frames, fps):
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def make_far(frames, scale, rng):
    h, w = frames[0].shape[:2]
    sw, sh = max(2, int(w * scale)), max(2, int(h * scale))
    x, y = rng.randint(0, w - sw), rng.randint(0, h - sh)
    out = []
    for f in frames:
        canvas = np.full((h, w, 3), 110, np.uint8)
        canvas[y:y + sh, x:x + sw] = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        out.append(canvas)
    return out


def make_crowd(clips, size=(1920, 1080)):
    """Clips in a grid on one 1920x1080 frame (2 or 3 clips -> 2x2 grid, 960x540 per clip), like
    a wide camera view with several people; shorter clips are looped."""
    n = len(clips[0])
    cols = int(np.ceil(np.sqrt(len(clips))))
    rows = int(np.ceil(len(clips) / cols))
    tw, th = size[0] // cols, size[1] // rows
    out = []
    for i in range(n):
        canvas = np.full((size[1], size[0], 3), 110, np.uint8)
        for j, c in enumerate(clips):
            r, q = divmod(j, cols)
            canvas[r * th:(r + 1) * th, q * tw:(q + 1) * tw] = cv2.resize(c[i % len(c)], (tw, th))
        out.append(canvas)
    return out


def main():
    ap = argparse.ArgumentParser(description="Far-away and crowd versions of the validation videos")
    ap.add_argument("--splits", default="splits/videos.csv")
    ap.add_argument("--labels", default="labels/omnifall")
    ap.add_argument("--gmd-root", required=True)
    ap.add_argument("--omnifall-root", required=True)
    ap.add_argument("--out", default="hard_val")
    ap.add_argument("--far-scale", type=float, default=0.3)
    ap.add_argument("--crowd-k", type=int, default=2, help="no-fall videos placed next to each fall video")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-fall", type=int, default=0, help="use only N validation fall videos (0 = all)")
    ap.add_argument("--n-adl", type=int, default=0, help="use only N validation no-fall videos (0 = all)")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    starts = fall_starts(a.labels)
    rows = [r for r in csv.DictReader(open(a.splits, newline="")) if r["split"] == "val"]
    vids = []
    for r in rows:
        key = (r["dataset"], r["path"])
        p = video_file(r["dataset"], r["path"], a.omnifall_root, a.gmd_root)
        if os.path.exists(p):
            vids.append(dict(key=key, path=p, fall=key in starts, start=starts.get(key)))
    falls = [v for v in vids if v["fall"]]
    adls = [v for v in vids if not v["fall"]]
    if a.n_fall:
        falls = rng.sample(falls, min(a.n_fall, len(falls)))
    if a.n_adl:
        adls = rng.sample(adls, min(a.n_adl, len(adls)))
    vids = falls + adls
    print(f"{len(vids)} validation videos found ({len(falls)} falls, {len(adls)} no-fall)", flush=True)

    os.makedirs(a.out, exist_ok=True)
    out_rows = []

    def frames_of(v, max_w=None):                       # read on demand: no big cache in memory
        return read_all(v["path"], max_w=max_w)

    def add(name, frames, fps, v_fall, people, kind):
        write(os.path.join(a.out, name + ".mp4"), frames, fps)
        out_rows.append(dict(file=name + ".mp4", member="val", light="day",
                             type="fall" if v_fall else "adl", fall_from="standing" if v_fall else "none",
                             fall_start_s=f"{v_fall['start']:.2f}" if v_fall else "", people=people,
                             distance="far" if kind == "far" else "near", angle="eye", notes=kind))

    for i, v in enumerate(vids):                                              # unchanged + far-away copies
        fr, fps = frames_of(v)
        if fr:
            add(f"normal_{i:03d}", fr, fps, v if v["fall"] else None, 1, "normal")
            add(f"far_{i:03d}", make_far(fr, a.far_scale, rng), fps, v if v["fall"] else None, 1, "far")
    for i, v in enumerate(falls):                                             # one faller in a crowd
        others = rng.sample(adls, min(a.crowd_k, len(adls)))
        fr, fps = frames_of(v, 960)
        clips = [fr] + [frames_of(o, 960)[0] for o in others]
        if all(clips):
            order = list(range(len(clips)))
            rng.shuffle(order)                                                # faller not always top-left
            add(f"crowd_fall_{i:03d}", make_crowd([clips[j] for j in order]), fps, v, len(clips), "crowd")
    for i in range(len(falls)):                                               # crowds where nobody falls
        group = rng.sample(adls, min(a.crowd_k + 1, len(adls)))
        read = [frames_of(o, 960) for o in group]
        clips = [r[0] for r in read]
        if all(clips):
            add(f"crowd_adl_{i:03d}", make_crowd(clips), read[0][1], None, len(clips), "crowd")

    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    kinds = defaultdict(lambda: [0, 0])
    for r in out_rows:
        kinds[r["notes"]][r["type"] == "fall"] += 1
    for k, (n_adl, n_fall) in kinds.items():
        print(f"  {k:<6} {n_fall} fall videos, {n_adl} no-fall videos")
    print(f"Saved {len(out_rows)} videos and labels.csv in {a.out}")


if __name__ == "__main__":
    main()
