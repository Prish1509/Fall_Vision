"""
2-second training windows (Member 3 / Member 2, Step 18).

Cuts every preprocessed video (preprocess.py) into overlapping 2-second windows of the main
person's 92 features and gives each window one of three labels, following OmniFall's two-stage
definition of a fall:
    0 normal    everything else (walk, stand, sit down, sitting, lie down, lying, kneel, ...)
    1 falling   the window contains the fall movement
    2 fallen    the person is on the ground after a fall (window ends inside a "fallen" segment)
"lie_down" and "lying" stay normal on purpose: lying down on purpose is a hard negative.

Rules for a window ending at time t (length 2 s):
    falling  if it overlaps a "fall" segment by at least min(0.5 s, half that segment)
    fallen   else if the time t lies inside a "fallen" segment
    normal   otherwise
Windows where the person is seen in less than half of the frames are dropped.

Output: <out>/<split>.npz with X (N x 30 x 92, float16), y (N), video index, end time, and
<out>/<split>_videos.csv (one row per video: dataset, path, has_fall, n_windows) for video-level
scoring. A summary of window counts per split and class is printed and saved to <out>/summary.md.

Usage:
  python src/make_windows.py --features /kaggle/input/.../features --labels labels/omnifall --out /kaggle/working/windows
"""

import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np

WIN_S, STRIDE_S = 2.0, 0.5
FALL, FALLEN = 1, 2                       # OmniFall class ids
CLASSES = ["normal", "falling", "fallen"]


def load_segments(label_dir):
    seg = defaultdict(list)               # (dataset, path) -> [(label, start, end)]
    for fname in os.listdir(label_dir):
        if fname.endswith(".csv"):
            with open(os.path.join(label_dir, fname), newline="") as f:
                for r in csv.DictReader(f):
                    seg[(r["dataset"], r["path"])].append((int(r["label"]), float(r["start"]), float(r["end"])))
    return seg


def window_label(segs, t0, t1):
    for lab, s, e in segs:
        if lab == FALL:
            overlap = min(t1, e) - max(t0, s)
            if overlap >= min(0.5, 0.5 * (e - s)):
                return 1
    for lab, s, e in segs:
        if lab == FALLEN and s <= t1 <= e:
            return 2
    return 0


def end_label(segs, t1):
    """OmniFall class of the segment that contains time t1 (-1 if none)."""
    for lab, s, e in segs:
        if s <= t1 <= e:
            return lab
    return -1


def main():
    ap = argparse.ArgumentParser(description="Cut labelled 2-second windows from preprocessed videos")
    ap.add_argument("--features", required=True, help="output folder of preprocess.py (has manifest.csv)")
    ap.add_argument("--labels", default="labels/omnifall")
    ap.add_argument("--out", default="windows")
    ap.add_argument("--min-seen", type=float, default=0.5)
    a = ap.parse_args()

    segs = load_segments(a.labels)
    with open(os.path.join(a.features, "manifest.csv"), newline="") as f:
        videos = [r for r in csv.DictReader(f) if r["status"] == "ok"]

    per_split = defaultdict(lambda: dict(X=[], y=[], vid=[], t=[], src=[], videos=[]))
    no_label = 0
    for r in videos:
        key = (r["dataset"], r["path"])
        if key not in segs:
            no_label += 1
            continue
        d = np.load(os.path.join(a.features, r["dataset"], r["path"] + ".npz"))
        feats, present, t, fps = d["feats"], d["present"], d["t"], float(d["fps"])
        L, step = 30, max(1, int(round(STRIDE_S * fps)))   # 30 frames = 2 s at 15 fps
        bucket = per_split[r["split"]]
        vid = len(bucket["videos"])
        n_win = 0
        for end in range(L, len(t) + 1, step):
            if present[end - L:end].mean() < a.min_seen:
                continue
            t1 = float(t[end - 1])
            bucket["X"].append(feats[end - L:end].astype(np.float16))
            bucket["y"].append(window_label(segs[key], t1 - WIN_S, t1))
            bucket["vid"].append(vid)
            bucket["t"].append(t1)
            bucket["src"].append(end_label(segs[key], t1))
            n_win += 1
        has_fall = int(any(lab == FALL for lab, _, _ in segs[key]))
        bucket["videos"].append(dict(dataset=r["dataset"], path=r["path"], has_fall=has_fall, n_windows=n_win))

    os.makedirs(a.out, exist_ok=True)
    lines = ["# Windows (Step 18)", "", f"2-s windows, stride {STRIDE_S} s, 30 frames x 92 features.", "",
             "| Split | videos | windows | normal | falling | fallen |", "|---|---|---|---|---|---|"]
    order = ["train", "val", "test_staged", "test_gmdcsa24", "test_oops", "test_own"]
    for split in sorted(per_split, key=lambda s: order.index(s) if s in order else 99):
        b = per_split[split]
        if not b["X"]:
            continue
        X, y = np.stack(b["X"]), np.array(b["y"], np.int8)
        np.savez_compressed(os.path.join(a.out, f"{split}.npz"), X=X, y=y,
                            vid=np.array(b["vid"], np.int32), t=np.array(b["t"], np.float32),
                            src=np.array(b["src"], np.int8))
        with open(os.path.join(a.out, f"{split}_videos.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(b["videos"][0].keys()))
            w.writeheader()
            w.writerows(b["videos"])
        c = Counter(y.tolist())
        lines.append(f"| {split} | {len(b['videos'])} | {len(y)} | {c[0]} | {c[1]} | {c[2]} |")
    if no_label:
        lines += ["", f"Videos without labels (skipped): {no_label}"]
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved one .npz per split in {a.out}")


if __name__ == "__main__":
    main()
