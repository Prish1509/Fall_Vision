"""
Localization check sheets (Step 23): is the alarm box on the person who fell?
Draws the system's box on the frame at the alarm time, tiles 12 frames per sheet, and writes
loc_check.csv with a 'verdict' column (ok / part / wrong / none). --score turns it into numbers.

Usage:
  python src/loc_sheets.py --results results/system_all_videos.csv --gmd-root GMD --omnifall-root OMNI --out loc
  python src/loc_sheets.py --score results/loc_check.csv
"""

import argparse
import csv
import os
import random
from collections import Counter

import cv2
import numpy as np

from cache_poses import video_file

TILE_W, COLS, PER_SHEET = 360, 4, 12


def frame_at(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def tile(img, box, label):
    if img is None:
        img = np.zeros((270, 360, 3), np.uint8)
        cv2.putText(img, "frame not readable", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif box:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, img.shape[1] // 200))
    s = TILE_W / img.shape[1]
    img = cv2.resize(img, (TILE_W, int(img.shape[0] * s)))
    img = cv2.copyMakeBorder(img, 34, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(img, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def make_sheets(a):
    rows = [r for r in csv.DictReader(open(a.results, newline="")) if r["alarm"] == "1" and r["box"]]
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], []).append(r)
    random.seed(0)
    picked = []
    for split, rs in sorted(by_split.items()):
        random.shuffle(rs)
        picked += rs[:a.n]
    os.makedirs(a.out, exist_ok=True)
    tiles, index = [], []
    for i, r in enumerate(picked, 1):
        path = video_file(r["dataset"], r["path"], a.omnifall_root, a.gmd_root, a.own_root)
        box = [float(v) for v in r["box"].split()]
        tiles.append(tile(frame_at(path, float(r["alarm_time_s"])), box, f"#{i}  {r['split']}"))
        index.append(dict(id=i, split=r["split"], dataset=r["dataset"], path=r["path"],
                          has_fall=r["has_fall"], alarm_time_s=r["alarm_time_s"], box=r["box"],
                          people_flagged=r.get("people_flagged", ""), verdict=""))
    h = max(t.shape[0] for t in tiles) if tiles else 0
    for p in range(0, len(tiles), PER_SHEET):
        page = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                for t in tiles[p:p + PER_SHEET]]
        while len(page) % COLS:
            page.append(np.full_like(page[0], 255))
        grid = np.vstack([np.hstack(page[r:r + COLS]) for r in range(0, len(page), COLS)])
        cv2.imwrite(os.path.join(a.out, f"sheet_{p // PER_SHEET + 1:02d}.jpg"), grid)
    with open(os.path.join(a.out, "loc_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    print(f"{len(picked)} alarms on {(len(tiles) + PER_SHEET - 1) // PER_SHEET} sheets in {a.out}/; "
          f"fill the 'verdict' column of loc_check.csv with ok / part / wrong / none")


def score(path):
    rows = [r for r in csv.DictReader(open(path, newline="")) if r["verdict"].strip()]
    lines = ["# Localization check (Step 23)", "",
             "Alarm frames with the system's box, checked by eye: is the box on the person who fell?", "",
             "| Set | Checked | ok | part | wrong | none | Correct person |", "|---|---|---|---|---|---|---|"]
    for name, rs in [("all", rows)] + sorted(
            ((s, [r for r in rows if r["split"] == s]) for s in {r["split"] for r in rows})):
        c = Counter(r["verdict"].strip().lower() for r in rs)
        n = len(rs)
        good = c["ok"] + c["part"]
        lines.append(f"| {name} | {n} | {c['ok']} | {c['part']} | {c['wrong']} | {c['none']} | "
                     f"{good}/{n} = {100 * good / max(n, 1):.0f} % |")
    out = os.path.join(os.path.dirname(path) or ".", "localization.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


def main():
    ap = argparse.ArgumentParser(description="Contact sheets for checking alarm boxes by eye")
    ap.add_argument("--results", default="results/system_all_videos.csv")
    ap.add_argument("--gmd-root", default=".")
    ap.add_argument("--omnifall-root", default=".")
    ap.add_argument("--own-root", default=None)
    ap.add_argument("--n", type=int, default=12, help="alarms sampled per test split")
    ap.add_argument("--out", default="loc")
    ap.add_argument("--score", default=None, help="filled loc_check.csv -> localization numbers")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    else:
        make_sheets(a)


if __name__ == "__main__":
    main()
