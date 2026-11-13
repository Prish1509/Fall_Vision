"""
The complete system on our own recorded videos (Member 2, Step 25 / Step 13).

Own videos are TEST ONLY. Each video goes through exactly what detect_falls.py does on a new video:
YOLO11m-pose + ByteTrack at input 640, a re-run at input 960 when people are small (standing height
under --small-px), then the GRU on every tracked person (alarms_for_video). The threshold is the one
chosen on validation (0.849); nothing is tuned here.

Labels come from splits/own_videos_labels.csv (light, type, fall_from, fall_start_s, people,
distance, angle). Reported:
  all videos, and per group: day / night, near / far, eye / high angle, 1 person / 2+ people,
  falls by where they started (standing, walking, chair = from sitting, bed, ...)
  timing: median error of the estimated fall start, median alarm delay after the true start
Optional --video-out writes annotated videos (boxes, skeletons, FALL label) for the demo.

Output: results/system_own.md and results/system_own_videos.csv

Usage:
  python src/eval_own.py --videos /kaggle/input/.../fallvision-own-videos --gru models/gru.pt --threshold 0.849
"""

import argparse
import csv
import glob
import os
import subprocess
import sys

import numpy as np
import torch

from baseline_fall import extract_tracks
from detect_falls import alarms_for_video, load_gru
from train_rf import video_metrics

EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp")


def run_video(path, grus, device, a):
    imgsz = 640
    data, _ = extract_tracks(path, a.model, 15.0, imgsz, conf=a.conf)
    heights = [b[3] - b[1] for f in data["frames"] for b in f["boxes"]]
    h90 = float(np.percentile(heights, 90)) if heights else 0.0
    if heights and h90 < a.small_px:
        imgsz = 960
        data, _ = extract_tracks(path, a.model, 15.0, imgsz, conf=a.conf)
    seen = np.mean([bool(f["boxes"]) for f in data["frames"]]) if data["frames"] else 0.0
    return alarms_for_video(data, grus, None, device, a.threshold, "fall+fallen"), imgsz, h90, seen


def table_row(name, rs):
    truth = np.array([x["has_fall"] for x in rs], bool)
    pred = np.array([x["alarm"] for x in rs], float)
    m = video_metrics(pred, truth, 0.5)
    timed = [x for x in rs if x["alarm"] and x["has_fall"] and x["true_fall_start_s"] != ""]
    errs = [abs(x["fall_start_est_s"] - x["true_fall_start_s"]) for x in timed]
    dels = [x["alarm_time_s"] - x["true_fall_start_s"] for x in timed]
    med = lambda v: f"{np.median(v):.1f} s" if v else "n/a"
    nf = int(truth.sum())
    return (f"| {name} | {len(rs)} | {nf} | {len(rs) - nf} | {m['tp']}/{nf} | {m['fp']} | "
            f"{m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['accuracy']:.2f} | "
            f"{med(errs)} | {med(dels)} |")


def main():
    ap = argparse.ArgumentParser(description="Complete system on our own recorded test videos")
    ap.add_argument("--videos", required=True, help="folder with our own videos")
    ap.add_argument("--labels", default="splits/own_videos_labels.csv")
    ap.add_argument("--gru", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, required=True, help="chosen on validation by fuse.py")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--small-px", type=int, default=150)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--video-out", default=None, help="folder for annotated videos (demo); slower")
    ap.add_argument("--name", default="own")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grus = [load_gru(g, device) for g in a.gru]
    files = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(a.videos, "**", "*"), recursive=True) if p.lower().endswith(EXTS)}
    labels = [r for r in csv.DictReader(open(a.labels, newline=""))
              if r.get("file") and "example" not in (r.get("notes") or "")]
    print(f"{len(labels)} labelled videos, {len(files)} video files, threshold {a.threshold}", flush=True)

    out = []
    for n, r in enumerate(labels, 1):
        key = os.path.splitext(r["file"].strip())[0]
        if key not in files:
            print(f"  {key}: video file not found, skipped")
            continue
        alarms, imgsz, h90, seen = run_video(files[key], grus, device, a)
        best = max(alarms, key=lambda x: x[4]) if alarms else None
        low = lambda c: (r.get(c) or "").strip().lower()
        st = low("fall_start_s")
        row = dict(video=key, light=low("light"), type=low("type"), fall_from=low("fall_from"),
                   people=low("people"), distance=low("distance"), angle=low("angle"),
                   has_fall=low("type") == "fall", alarm=bool(alarms), people_flagged=len(alarms),
                   score=best[4] if best else 0.0, alarm_time_s=best[1] if best else "",
                   fall_start_est_s=best[2] if best else "", true_fall_start_s=float(st) if st else "",
                   person_id=best[0] if best else "", box=" ".join(map(str, best[3])) if best and best[3] else "",
                   imgsz=imgsz, standing_px=round(h90), person_seen=round(float(seen), 2),
                   notes=r.get("notes") or "")
        out.append(row)
        verdict = ("ok" if row["alarm"] == row["has_fall"] else
                   "MISSED FALL" if row["has_fall"] else "FALSE ALARM")
        print(f"  {n:>2}/{len(labels)} {key:<6} {row['light']:<5} {row['type']:<4} "
              f"alarm={'yes' if row['alarm'] else 'no ':<3} score={row['score']:.2f} "
              f"input={imgsz} seen={seen:.0%}  {verdict}", flush=True)
        if a.video_out:
            os.makedirs(a.video_out, exist_ok=True)
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "detect_falls.py"),
                            "--video", files[key], "--gru", *a.gru, "--threshold", str(a.threshold),
                            "--small-px", str(a.small_px), "--out", os.path.join(a.video_out, key + "_falls.mp4")],
                           check=False, stdout=subprocess.DEVNULL)

    head = ["| Group | Videos | Falls | No-fall | Falls caught | False alarms | Precision | Recall | "
            "F1 | Accuracy | Fall-start error | Alarm delay |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines = ["# Complete system on our own videos (test only)", "",
             f"{len(out)} videos recorded by the team, never used for training or tuning. Same code as "
             f"detect_falls.py (input 640, re-run at 960 if people are small), threshold {a.threshold} "
             f"(chosen on validation).", "", *head, table_row("all", out)]
    for col, title in [("light", "light"), ("distance", "distance"), ("angle", "angle")]:
        for g in sorted({x[col] for x in out}):
            lines.append(table_row(f"{title}: {g}", [x for x in out if x[col] == g]))
    one = [x for x in out if x["people"] == "1"]
    many = [x for x in out if x["people"].isdigit() and int(x["people"]) >= 2]
    if one and many:
        lines += [table_row("1 person", one), table_row("2+ people", many)]

    falls = [x for x in out if x["has_fall"]]
    lines += ["", "## Falls by where they started", "", "| Fell from | Falls | Caught |", "|---|---|---|"]
    for g in sorted({x["fall_from"] for x in falls}):
        rs = [x for x in falls if x["fall_from"] == g]
        lines.append(f"| {g} | {len(rs)} | {sum(x['alarm'] for x in rs)}/{len(rs)} |")

    lines += ["", "## Every video", "",
              "| Video | Light | Truth | Alarm | Score | Fall start (true / est.) | Alarm at | Input | Person seen | Notes |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for x in out:
        ts = f"{x['true_fall_start_s']} / {x['fall_start_est_s']}" if x["has_fall"] else ""
        lines.append(f"| {x['video']} | {x['light']} | {'fall' if x['has_fall'] else 'no fall'} | "
                     f"{'yes' if x['alarm'] else 'no'} | {x['score']:.2f} | {ts} | {x['alarm_time_s']} | "
                     f"{x['imgsz']} | {x['person_seen']:.0%} | {x['notes']} |")

    os.makedirs("results", exist_ok=True)
    md = f"results/system_{a.name}.md"
    open(md, "w").write("\n".join(lines) + "\n")
    with open(md.replace(".md", "_videos.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("\n" + "\n".join(lines[4:lines.index("## Every video")]))
    print(f"\nSaved {md} and {md.replace('.md', '_videos.csv')}")


if __name__ == "__main__":
    main()
