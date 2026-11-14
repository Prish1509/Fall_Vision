"""
Evaluation of the submitted (causal) system, Step 28.

infer.py works frame by frame and only uses the current and past frames, so its scores differ a
little from the earlier whole-video evaluation (eval_system.py), where body height and floor level
were measured over the whole clip. This script runs exactly the infer.py engine (fv_engine) on
every video of the chosen splits and stores, per video, the collapse score over time (highest over
all people, every 0.5 s).

  --run     score the videos (resumable: videos already in --scores are skipped)
  --report  choose the alarm threshold on the VALIDATION split only (highest F1; ties -> fewer false
            alarms), then report every test split once at that threshold, next to the earlier
            threshold from config.yaml; also writes the chosen threshold to results/causal_eval.md

Usage:
  python src/eval_causal.py --run --omnifall-root OMNI --gmd-root GMD --own-root OWN --device cuda:0
  python src/eval_causal.py --report
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_poses import video_file                                  # noqa: E402
from eval_system import load_labels                                  # noqa: E402
from fv_engine import CollapseEngine, PoseTracker, load_config, new_gru, torch_device   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ["split", "dataset", "path", "has_fall", "true_start_s", "max_score", "series", "frames", "seconds"]


def score_video(path, cfg, pose, gru, device):
    """Return [(t, highest smoothed score over people)] every stride, exactly as infer.py computes it."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps > 1000:
        src_fps = None
    pose.reset()
    eng = CollapseEngine(cfg, gru, device)
    series, last_slot, idx = [], -1, -1
    while True:
        ok, img = cap.read()
        if not ok:
            break
        idx += 1
        t = idx / src_fps if src_fps else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        slot = int(round(t * cfg["model_fps"]))
        if slot <= last_slot:
            continue
        last_slot = slot
        det = pose.track(img, eng.pose_input_size(slot))
        eng.add_frame(slot, det, (img.shape[1], img.shape[0]))
        due = slot >= eng.next_score
        eng.step(slot, t)
        if due:
            series.append((round(t, 2), round(max([p.score for p in eng.people.values()] or [0.0]), 4)))
    cap.release()
    return series, idx + 1


def videos_to_score(a):
    has_fall, start = load_labels(os.path.join(ROOT, "labels", "omnifall"))
    rows = []
    for r in csv.DictReader(open(os.path.join(ROOT, "splits", "videos.csv"), newline="")):
        if r["split"] in a.splits and r["dataset"] != "own":
            key = (r["dataset"], r["path"])
            rows.append(dict(split=r["split"], dataset=r["dataset"], path=r["path"], has_fall=int(has_fall[key]),
                             true_start_s=start.get(key, ""),
                             file=video_file(r["dataset"], r["path"], a.omnifall_root, a.gmd_root)))
    if "test_own" in a.splits and a.own_root:
        for r in csv.DictReader(open(os.path.join(ROOT, "splits", "own_videos_labels.csv"), newline="")):
            if r.get("file") and "example" not in (r.get("notes") or ""):
                stem = os.path.splitext(r["file"])[0]
                rows.append(dict(split="test_own", dataset="own", path=stem,
                                 has_fall=int(r["type"].strip().lower() == "fall"),
                                 true_start_s=r["fall_start_s"] or "",
                                 file=video_file("own", stem, None, None, a.own_root)))
    return rows


def run(a):
    cfg = load_config(a.config)
    dev, ydev, warn = torch_device(a.device)
    if warn:
        print("Warning:", warn)
    gru, pose = new_gru(cfg, dev), PoseTracker(cfg, ydev)
    done = set()
    if os.path.exists(a.scores):
        done = {(r["split"], r["dataset"], r["path"]) for r in csv.DictReader(open(a.scores, newline=""))}
    todo = [r for r in videos_to_score(a) if (r["split"], r["dataset"], r["path"]) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(done)} videos already scored, {len(todo)} to go", flush=True)
    new_file = not os.path.exists(a.scores)
    os.makedirs(os.path.dirname(os.path.abspath(a.scores)), exist_ok=True)
    t0 = time.time()
    with open(a.scores, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for n, r in enumerate(todo, 1):
            if not os.path.exists(r["file"]):
                print(f"  missing video: {r['file']}")
                continue
            series, frames = score_video(r["file"], cfg, pose, gru, dev)
            w.writerow(dict(split=r["split"], dataset=r["dataset"], path=r["path"], has_fall=r["has_fall"],
                            true_start_s=r["true_start_s"], max_score=max([s for _, s in series] or [0.0]),
                            series=json.dumps(series), frames=frames,
                            seconds=round(series[-1][0], 2) if series else 0))
            f.flush()
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / 60:.0f} min, ~{el / n * (len(todo) - n) / 60:.0f} min left", flush=True)


def metrics(rows, thr):
    tp = sum(1 for r in rows if r["has_fall"] and r["max_score"] >= thr)
    fp = sum(1 for r in rows if not r["has_fall"] and r["max_score"] >= thr)
    fn = sum(1 for r in rows if r["has_fall"] and r["max_score"] < thr)
    tn = len(rows) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    delays = []
    for r in rows:
        if r["has_fall"] and r["max_score"] >= thr and r["true_start_s"] != "":
            first = next(t for t, s in r["series"] if s >= thr)
            delays.append(first - float(r["true_start_s"]))
    return dict(n=len(rows), tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=rc, f1=f1,
                accuracy=(tp + tn) / max(1, len(rows)), delay=float(np.median(delays)) if delays else None)


def report(a):
    rows = list(csv.DictReader(open(a.scores, newline="")))
    for r in rows:
        r["has_fall"] = int(r["has_fall"])
        r["max_score"] = float(r["max_score"])
        r["series"] = json.loads(r["series"])
    val = [r for r in rows if r["split"] == "val"]
    if not val:
        sys.exit("no validation videos scored yet")
    best = None
    for thr in sorted({r["max_score"] for r in val} | {1.01}):
        m = metrics(val, thr)
        if best is None or m["f1"] > best[1]["f1"] + 1e-12 or (abs(m["f1"] - best[1]["f1"]) < 1e-12 and m["fp"] < best[1]["fp"]):
            best = (thr, m)
    chosen = round(best[0], 4)
    old = load_config(a.config)["threshold"]
    order = ["val", "test_gmdcsa24", "test_staged", "test_oops", "test_own"]
    lines = ["# Submitted (causal) system: evaluation (Step 28)", "",
             "infer.py engine on whole videos, frame by frame, current and past frames only. One alarm per video "
             "is counted if the highest collapse score reaches the threshold.", "",
             f"Threshold chosen on VALIDATION only (highest F1, ties -> fewer false alarms): **{chosen}**. "
             f"Earlier threshold (whole-video evaluation): {old}.", "",
             "| Split | Threshold | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Median alarm delay |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for split in order:
        rs = [r for r in rows if r["split"] == split]
        if not rs:
            continue
        for name, thr in (("chosen", chosen), ("earlier", old)):
            m = metrics(rs, thr)
            d = f"{m['delay']:.1f} s" if m["delay"] is not None else "n/a"
            lines.append(f"| {split} | {name} {thr} | {m['n']} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {d} |")
    out = os.path.join(ROOT, "results", "causal_eval.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}\nCHOSEN_THRESHOLD={chosen}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the submitted causal system")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--scores", default=os.path.join(ROOT, "results", "causal_scores.csv"))
    ap.add_argument("--splits", nargs="+", default=["val", "test_gmdcsa24", "test_staged", "test_own", "test_oops"])
    ap.add_argument("--omnifall-root", default=".")
    ap.add_argument("--gmd-root", default=".")
    ap.add_argument("--own-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.run:
        run(a)
    if a.report:
        report(a)


if __name__ == "__main__":
    main()
