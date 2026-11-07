"""
Rule baseline on every split (Member 2, Step 25 preparation).

Runs the rule-based detector (baseline_fall.run_rules, rules v2, hold 2.0 s) on the cached skeletons
of every video in the split file, and scores it exactly like the learned models (train_rf.py,
train_gru.py): one alarm per video or none, a video counts as a fall video if its OmniFall labels
contain a "fall" segment. This puts the rules in the same results table as the Random Forest and
the GRU, on the same test sets (GMDCSA24 S3-4, staged test, OOPS).

No video is decoded: it only reads the pose caches made by cache_poses.py.

Output: results/rules_splits.md and results/rules_splits_videos.csv

Usage:
  python src/eval_rules_splits.py --poses /kaggle/input/.../poses
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict
from dataclasses import replace

import numpy as np

from baseline_fall import Params, run_rules
from train_rf import fmt, video_metrics

ORDER = ["train", "val", "test_staged", "test_gmdcsa24", "test_oops", "test_own"]


def main():
    ap = argparse.ArgumentParser(description="Score the rule baseline on every split")
    ap.add_argument("--poses", required=True, help="output folder of cache_poses.py (has index.csv)")
    ap.add_argument("--labels", default="labels/omnifall")
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--out", default="results/rules_splits.md")
    a = ap.parse_args()

    has_fall = defaultdict(bool)
    for fname in os.listdir(a.labels):
        if fname.endswith(".csv"):
            with open(os.path.join(a.labels, fname), newline="") as f:
                for r in csv.DictReader(f):
                    has_fall[(r["dataset"], r["path"])] |= int(r["label"]) == 1

    with open(os.path.join(a.poses, "index.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in ("cached", "already")]
    params = replace(Params(), hold=a.hold)
    per = defaultdict(lambda: ([], []))          # split -> (truth list, prediction list)
    out_rows = []
    for n, r in enumerate(rows, 1):
        d = pickle.load(open(os.path.join(a.poses, r["dataset"], r["path"] + ".pkl"), "rb"))
        events, _ = run_rules(d, params)
        truth = has_fall[(r["dataset"], r["path"])]
        per[r["split"]][0].append(truth)
        per[r["split"]][1].append(bool(events))
        out_rows.append(dict(split=r["split"], dataset=r["dataset"], path=r["path"], has_fall=int(truth),
                             alarm=int(bool(events)), alarm_time_s=events[0][1] if events else "",
                             fall_start_est_s=events[0][2] if events else ""))
        if n % 200 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}", flush=True)

    lines = ["# Rules v2 on every split", "",
             f"Rule baseline (v2, hold {a.hold} s) on the cached YOLO11m-pose skeletons; video-level "
             f"scoring identical to the learned models.", "",
             "| Split | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |",
             "|---|---|---|---|---|---|---|---|---|"]
    print()
    for split in sorted(per, key=lambda s: ORDER.index(s) if s in ORDER else 99):
        truth, pred = map(np.array, per[split])
        m = video_metrics(pred.astype(float), truth.astype(bool), 0.5)
        print(f"{split:<14} {fmt(m)}")
        lines.append(f"| {split} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_out = a.out.replace(".md", "_videos.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {a.out} and {csv_out}")


if __name__ == "__main__":
    main()
