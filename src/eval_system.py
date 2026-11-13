"""
Final evaluation of the COMPLETE system (Member 1 / Member 2, Steps 25 and 23).

Scores detect_falls.py end to end on whole videos, using the cached skeletons so no video has to be
decoded again. It calls the same alarms_for_video() that detect_falls.py uses, so the numbers in the
report describe exactly the system that runs on a new video: every tracked person is scored, not
only the most visible one, and a video raises an alarm if any person does.

Reported per split (and per group, e.g. day / night, with --groups):
  detection   precision, recall, F1, accuracy, false alarms, missed falls (one alarm per video)
  timing      median error of the estimated fall start, and median alarm delay after the true start
  who         for videos with more than one person: whether anybody was flagged at all

The alarm threshold is the one chosen on validation by fuse.py; it is never tuned here.

Output: results/system_<name>.md and results/system_<name>_videos.csv

Usage:
  python src/eval_system.py --poses POSES --gru models/gru.pt --threshold 0.849
  python src/eval_system.py --poses POSES --gru models/gru.pt --threshold 0.849 \
                            --splits test_own --groups splits/own_videos_labels.csv --name own
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
import torch

from detect_falls import alarms_for_video, load_gru
from train_rf import fmt, video_metrics

FALL = 1


def load_labels(label_dir):
    has_fall, start = defaultdict(bool), {}
    for fname in os.listdir(label_dir):
        if not fname.endswith(".csv"):
            continue
        for r in csv.DictReader(open(os.path.join(label_dir, fname), newline="")):
            key = (r["dataset"], r["path"])
            if int(r["label"]) == FALL:
                has_fall[key] = True
                start[key] = min(start.get(key, 1e9), float(r["start"]))
    return has_fall, start


def load_groups(path):
    """own_videos_labels.csv -> {filename without extension: 'day'/'night'} and true fall starts."""
    groups, starts, truth = {}, {}, {}
    if not path or not os.path.exists(path):
        return groups, starts, truth
    for r in csv.DictReader(open(path, newline="")):
        if not r.get("file") or "example" in (r.get("notes") or ""):
            continue
        key = os.path.splitext(r["file"])[0]
        groups[key] = (r.get("light") or "all").strip() or "all"
        truth[key] = (r.get("type") or "").strip().lower() == "fall"
        if (r.get("fall_start_s") or "").strip():
            starts[key] = float(r["fall_start_s"])
    return groups, starts, truth


def main():
    ap = argparse.ArgumentParser(description="Score the complete system on cached skeletons")
    ap.add_argument("--poses", required=True, help="output folder of cache_poses.py (has index.csv)")
    ap.add_argument("--labels", default="labels/omnifall")
    ap.add_argument("--gru", nargs="+", required=True)
    ap.add_argument("--rf", default=None)
    ap.add_argument("--score", choices=["fall+fallen", "falling"], default="fall+fallen")
    ap.add_argument("--threshold", type=float, required=True, help="chosen on validation by fuse.py")
    ap.add_argument("--splits", nargs="+", default=["test_gmdcsa24", "test_staged", "test_oops", "test_own"])
    ap.add_argument("--groups", default=None, help="own_videos_labels.csv: also report day vs night")
    ap.add_argument("--name", default="all")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grus = [load_gru(g, device) for g in a.gru]
    rf = None
    if a.rf:
        import joblib
        rf = joblib.load(a.rf)

    has_fall, true_start = load_labels(a.labels)
    groups, own_starts, own_truth = load_groups(a.groups)

    with open(os.path.join(a.poses, "index.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r["status"] in ("cached", "already") and r["split"] in a.splits]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} videos, threshold {a.threshold}, models {', '.join(os.path.basename(g) for g in a.gru)}"
          + (" + RF" if rf is not None else ""), flush=True)

    out_rows = []
    for n, r in enumerate(rows, 1):
        key = (r["dataset"], r["path"])
        data = pickle.load(open(os.path.join(a.poses, r["dataset"], r["path"] + ".pkl"), "rb"))
        alarms = alarms_for_video(data, grus, rf, device, a.threshold, a.score)
        best = max(alarms, key=lambda x: x[4]) if alarms else None
        base = os.path.basename(r["path"]) if r["dataset"] == "own" else None
        truth = own_truth.get(base, has_fall[key]) if base is not None else has_fall[key]
        t0 = own_starts.get(base) if base is not None else true_start.get(key)
        if t0 is None:
            t0 = true_start.get(key)
        out_rows.append(dict(
            split=r["split"], group=groups.get(base, "all") if base is not None else "all", dataset=r["dataset"], path=r["path"],
            has_fall=int(truth), alarm=int(bool(alarms)), people_flagged=len(alarms),
            score=best[4] if best else 0.0,
            alarm_time_s=best[1] if best else "", fall_start_est_s=best[2] if best else "",
            true_fall_start_s=t0 if t0 is not None else "",
            person_id=best[0] if best else "",
            box=" ".join(map(str, best[3])) if best and best[3] else ""))
        if n % 100 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}", flush=True)

    def report(name, rs):
        truth = np.array([bool(x["has_fall"]) for x in rs])
        pred = np.array([bool(x["alarm"]) for x in rs], float)
        m = video_metrics(pred, truth, 0.5)
        errs = [abs(x["fall_start_est_s"] - x["true_fall_start_s"]) for x in rs
                if x["alarm"] and x["has_fall"] and x["true_fall_start_s"] != ""]
        dels = [x["alarm_time_s"] - x["true_fall_start_s"] for x in rs
                if x["alarm"] and x["has_fall"] and x["true_fall_start_s"] != ""]
        med = lambda v: f"{np.median(v):.1f} s" if v else "n/a"
        print(f"{name:<22} {fmt(m)}  fall-start error={med(errs)}  alarm delay={med(dels)}")
        return m, med(errs), med(dels), len(rs)

    lines = ["# Complete system on whole videos (Steps 23 and 25)", "",
             f"detect_falls.py scored end to end on the cached skeletons: every tracked person is scored, "
             f"a video alarms if any person does. Threshold {a.threshold} (chosen on validation).", "",
             "| Set | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Fall-start error | Alarm delay |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    print()
    for split in a.splits:
        rs = [x for x in out_rows if x["split"] == split]
        if not rs:
            continue
        m, e, d, n_ = report(split, rs)
        lines.append(f"| {split} | {n_} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {e} | {d} |")
        gs = sorted({x["group"] for x in rs})
        if gs != ["all"]:
            for g in gs:
                sub = [x for x in rs if x["group"] == g]
                m, e, d, n_ = report(f"  {split} / {g}", sub)
                lines.append(f"| {split} - {g} | {n_} | {m['f1']:.3f} | {m['precision']:.3f} | "
                             f"{m['recall']:.3f} | {m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | "
                             f"{m['fn']} | {e} | {d} |")

    wrong = [x for x in out_rows if bool(x["has_fall"]) != bool(x["alarm"])]
    lines += ["", f"Mistakes: {len(wrong)}", "", "| Video | True | Predicted | Score |", "|---|---|---|---|"]
    for x in wrong[:40]:
        lines.append(f"| {x['dataset']}/{x['path']} | {'fall' if x['has_fall'] else 'no fall'} | "
                     f"{'alarm' if x['alarm'] else 'no alarm'} | {x['score']:.3f} |")

    out = f"results/system_{a.name}.md"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_out = out.replace(".md", "_videos.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nSaved {out} and {csv_out}")


if __name__ == "__main__":
    main()
