"""
Check our own recorded videos before they are used (Member 3, Step 13).

Reads every video in --videos (size, frame rate, length) and compares the folder with
splits/own_videos_labels.csv: videos without a label row, rows without a video, wrong values,
falls without fall_start_s or with a start after the end, and videos that are too short (< 4 s),
too small (< 480 px on the short side) or unreadable. Any file name works (01.mp4, 02.mp4, ...).
Counts the clips against the 20-video recording plan and writes <out>/own_labels_draft.csv
(one row per video) to copy into the labels sheet.

Usage:
  python src/check_own_videos.py --videos /kaggle/input/fallvision-own-videos --labels splits/own_videos_labels.csv
"""

import argparse
import csv
import glob
import os
from collections import Counter

import cv2

EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp")
ALLOWED = dict(light={"day", "night"}, type={"fall", "adl"},
               distance={"near", "far"}, angle={"eye", "high"})
PLAN = [("day falls", 6), ("night clips", 8), ("night falls", 4), ("no-fall clips", 9),
        ("2+ people", 2), ("far away", 4), ("high angle", 4)]


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = img.shape[:2]                       # size after the phone's rotation is applied
    return dict(w=w, h=h, fps=round(fps, 1), seconds=round(n / fps, 1) if fps else 0.0)


def main():
    ap = argparse.ArgumentParser(description="Check own recorded videos against the labels sheet")
    ap.add_argument("--videos", required=True, help="folder with our own videos (searched recursively)")
    ap.add_argument("--labels", default="splits/own_videos_labels.csv")
    ap.add_argument("--out", default="/kaggle/working")
    a = ap.parse_args()

    files = sorted(p for p in glob.glob(os.path.join(a.videos, "**", "*"), recursive=True)
                   if p.lower().endswith(EXTS))
    vids = {os.path.splitext(os.path.basename(p))[0]: p for p in files}
    labels = {}
    if os.path.exists(a.labels):
        for r in csv.DictReader(open(a.labels, newline="")):
            if r.get("file") and "example" not in (r.get("notes") or ""):
                labels[os.path.splitext(r["file"].strip())[0]] = r

    problems, info, draft = [], {}, []
    print(f"{len(vids)} videos in {a.videos}, {len(labels)} rows in {a.labels}\n")
    print(f"{'video':<14} {'size':>10} {'fps':>5} {'sec':>6}  label")
    for key, path in sorted(vids.items()):
        d = video_info(path)
        info[key] = d
        if d is None:
            problems.append(f"{key}: cannot be read (try converting to .mp4)")
        else:
            if min(d["w"], d["h"]) < 480:
                problems.append(f"{key}: small video {d['w']}x{d['h']} (want 720p or more)")
            if d["seconds"] < 4:
                problems.append(f"{key}: only {d['seconds']} s long (want at least 4 s)")
        if key not in labels:
            problems.append(f"{key}: no row in the labels sheet")
        size = f"{d['w']}x{d['h']}" if d else "unreadable"
        print(f"{key:<14} {size:>10} {d['fps'] if d else '':>5} {d['seconds'] if d else '':>6}  "
              f"{'yes' if key in labels else 'MISSING'}")
        draft.append(dict(file=os.path.basename(path), member="M3", light="", type="", fall_from="",
                          fall_start_s="", people="1", distance="near", angle="eye", notes=""))

    for key, r in labels.items():
        if key not in vids:
            problems.append(f"{key}: in the labels sheet but no video file")
        for col, ok in ALLOWED.items():
            v = (r.get(col) or "").strip().lower()
            if v not in ok:
                problems.append(f"{key}: {col} = '{v}' (allowed: {', '.join(sorted(x for x in ok if x))})")
        if not (r.get("people") or "").strip().isdigit() or int(r["people"]) < 1:
            problems.append(f"{key}: people = '{r.get('people')}' (write a number: 1, 2, 3, ...)")
        is_fall = (r.get("type") or "").strip().lower() == "fall"
        st = (r.get("fall_start_s") or "").strip()
        if is_fall and not st:
            problems.append(f"{key}: fall without fall_start_s")
        if is_fall and (r.get("fall_from") or "").strip().lower() in ("", "none"):
            problems.append(f"{key}: fall without fall_from (standing / walking / chair / bed)")
        if st:
            try:
                s = float(st)
                d = info.get(key)
                if d and d["seconds"] and s > d["seconds"]:
                    problems.append(f"{key}: fall_start_s {s} is after the end of the video ({d['seconds']} s)")
            except ValueError:
                problems.append(f"{key}: fall_start_s '{st}' is not a number")

    rs = list(labels.values())
    low = lambda r, c: (r.get(c) or "").strip().lower()
    have = {"day falls": sum(low(r, "light") == "day" and low(r, "type") == "fall" for r in rs),
            "night clips": sum(low(r, "light") == "night" for r in rs),
            "night falls": sum(low(r, "light") == "night" and low(r, "type") == "fall" for r in rs),
            "no-fall clips": sum(low(r, "type") == "adl" for r in rs),
            "2+ people": sum(low(r, "people").isdigit() and int(low(r, "people")) >= 2 for r in rs),
            "far away": sum(low(r, "distance") == "far" for r in rs),
            "high angle": sum(low(r, "angle") == "high" for r in rs)}
    print("\nRecording plan (from the labels sheet)")
    for name, want in PLAN:
        print(f"  {name:<14} {have[name]:>3} / {want}   {'ok' if have[name] >= want else 'still needed'}")
    print("  falls from a chair:", sum(low(r, "fall_from") == "chair" for r in rs))

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, "own_labels_draft.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "member", "light", "type", "fall_from", "fall_start_s",
                                          "people", "distance", "angle", "notes"])
        w.writeheader()
        w.writerows(draft)

    print(f"\nProblems: {len(problems)}")
    for p in problems:
        print("  -", p)
    print(f"\nDraft rows for every video (copy into the labels sheet): {out}")


if __name__ == "__main__":
    main()
