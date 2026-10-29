"""
Evaluate the rule-based baseline on GMDCSA24.

GMDCSA24 layout:  <root>/Subject N/Fall/*.mp4, <root>/Subject N/ADL/*.mp4, plus Fall.csv / ADL.csv
Each video is one clip, so we score at video level:
    a Fall video is correct if the baseline raises at least one alert,
    an ADL video is correct if it raises none.

Good practice (Lecture 11): thresholds are tuned on Subjects 1-2 only and reported on Subjects 3-4.

Usage:
  python src/eval_gmdcsa24.py --root GMDCSA24 --cache cache          # extract once, evaluate defaults
  python src/eval_gmdcsa24.py --root GMDCSA24 --cache cache --tune   # grid-search on S1-2, report S3-4
  python src/eval_gmdcsa24.py --root GMDCSA24 --cache cache --model yolo11m-pose.pt   # bigger pose model
  python src/eval_gmdcsa24.py --root GMDCSA24 --cache cache --rules v1               # original rules
"""

import argparse
import csv
import itertools
import os
import pickle
import re
import time
from dataclasses import replace
from pathlib import Path

from baseline_fall import Params, extract_tracks, params_v1, run_rules

TUNE_SUBJECTS = {1, 2}
TEST_SUBJECTS = {3, 4}


def load_labels(root):
    """Return {video_path: dict(subject, is_fall, fall_start, description)}."""
    items = {}
    for subj_dir in sorted(Path(root).glob("Subject *")):
        subj = int(subj_dir.name.split()[-1])
        for kind in ("Fall", "ADL"):
            meta = {}
            csv_path = subj_dir / f"{kind}.csv"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
                    rows = list(csv.reader(f))
                for row in rows[1:]:
                    if not row:
                        continue
                    classes = ",".join(row[5:])
                    m = re.search(r"Fall\w*\s*\([^)]*\)\s*\[\s*([\d.]+)\s*to", classes)
                    meta[row[0].strip()] = (float(m.group(1)) if m else None, row[4].strip()
                                            if len(row) > 4 else "")
            for v in sorted((subj_dir / kind).glob("*.mp4")):
                start, desc = meta.get(v.name, (None, ""))
                items[str(v)] = dict(subject=subj, is_fall=(kind == "Fall"),
                                     fall_start=start, description=desc,
                                     rel=f"{subj_dir.name}/{kind}/{v.name}")
    return items


def get_tracks(video, cache_dir, model, imgsz, night="off", conf=0.25, small_px=0):
    tag = "" if (night == "off" and conf == 0.25) else f"_{night}_c{conf}"
    tag += f"_small{small_px}" if small_px else ""
    key = Path(cache_dir) / (re.sub(r"[^\w]+", "_", video) + f"_{Path(model).stem}_{imgsz}{tag}.pkl")
    if key.exists():
        return pickle.loads(key.read_bytes())
    data, _ = extract_tracks(video, model, 15.0, imgsz, night=night, conf=conf, small_px=small_px)
    key.write_bytes(pickle.dumps(data))
    return data


def score(items, tracks, params, subjects):
    tp = fp = tn = fn = 0
    delays, start_errs, rows = [], [], []
    for v, info in items.items():
        if info["subject"] not in subjects:
            continue
        events, _ = run_rules(tracks[v], params)
        pred = len(events) > 0
        if info["is_fall"]:
            tp += pred
            fn += not pred
            if pred and info["fall_start"] is not None:
                delays.append(events[0][1] - info["fall_start"])
                start_errs.append(abs(events[0][2] - info["fall_start"]))   # temporal localization
        else:
            fp += pred
            tn += not pred
        rows.append(dict(video=info["rel"], subject=info["subject"], label="Fall" if info["is_fall"] else "ADL",
                         predicted="Fall" if pred else "ADL",
                         alert_time_s=events[0][1] if events else "",
                         est_fall_start_s=events[0][2] if events else "",
                         true_fall_start_s=info["fall_start"] if info["fall_start"] is not None else "",
                         box_x1_y1_x2_y2=" ".join(map(str, events[0][3])) if events else "",
                         description=info["description"]))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, f1=f1, accuracy=acc,
                median_delay=med(delays), median_start_err=med(start_errs)), rows


def show(name, m):
    d = f"{m['median_delay']:.1f} s" if m["median_delay"] is not None else "n/a"
    s = f"{m['median_start_err']:.1f} s" if m["median_start_err"] is not None else "n/a"
    print(f"{name:<28} F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
          f"Acc={m['accuracy']:.3f}  [TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}]  "
          f"alert delay={d}  fall-start error={s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="GMDCSA24 folder containing 'Subject 1' ...")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tune", action="store_true", help="grid-search thresholds on Subjects 1-2")
    ap.add_argument("--rules", choices=["v1", "v2"], default="v2",
                    help="v1 = original rules, v2 = sitting-fall + false-positive fixes")
    ap.add_argument("--hold", type=float, default=None,
                    help="seconds a person must stay down before an alert (v2 default 2.0)")
    ap.add_argument("--out", default="results/gmdcsa24_rules.csv")
    ap.add_argument("--night", choices=["off", "clahe", "denoise"], default="off")
    ap.add_argument("--conf", type=float, default=0.25, help="person detection threshold")
    ap.add_argument("--small-px", type=int, default=0, help="zoom re-check for people shorter than this (0 = off)")
    a = ap.parse_args()

    os.makedirs(a.cache, exist_ok=True)
    items = load_labels(a.root)
    print(f"{len(items)} videos "
          f"({sum(i['is_fall'] for i in items.values())} Fall, "
          f"{sum(not i['is_fall'] for i in items.values())} ADL)")

    tracks, t0 = {}, time.time()
    for n, v in enumerate(items, 1):
        tracks[v] = get_tracks(v, a.cache, a.model, a.imgsz, a.night, a.conf, a.small_px)
        if n % 10 == 0 or n == len(items):
            print(f"  pose extracted {n}/{len(items)}  ({time.time() - t0:.0f} s)", flush=True)

    base = params_v1() if a.rules == "v1" else Params()
    if a.hold is not None:
        base = replace(base, hold=a.hold)
    params = base
    print(f"Rules: {a.rules}, hold = {base.hold} s")
    if a.tune:
        grid = dict(drop_ratio=[0.15, 0.2, 0.25, 0.3, 0.4], down_angle=[45, 55, 65],
                    hold=[0.5, 1.0, 1.5], down_aspect=[0.8, 1.0, 1.2])
        best = None
        for combo in itertools.product(*grid.values()):
            p = replace(base, **dict(zip(grid, combo)))
            m, _ = score(items, tracks, p, TUNE_SUBJECTS)
            if best is None or m["f1"] > best[0]["f1"]:
                best = (m, p)
        params = best[1]
        print("\nBest thresholds (chosen on Subjects 1-2):",
              {k: getattr(params, k) for k in grid})

    print()
    m_tune, _ = score(items, tracks, params, TUNE_SUBJECTS)
    m_test, rows = score(items, tracks, params, TEST_SUBJECTS)
    m_all, rows_all = score(items, tracks, params, TUNE_SUBJECTS | TEST_SUBJECTS)
    show("Subjects 1-2 (tuning)", m_tune)
    show("Subjects 3-4 (held out)", m_test)
    show("All subjects", m_all)

    print("\nMistakes on held-out Subjects 3-4 (watch these videos):")
    for r in rows:
        if r["label"] != r["predicted"]:
            print(f"  {r['label']:>4} -> predicted {r['predicted']:<4} | {r['video']} | {r['description'][:70]}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys()))
        w.writeheader()
        w.writerows(rows_all)
    print(f"\nPer-video results saved to {a.out}")


if __name__ == "__main__":
    main()
