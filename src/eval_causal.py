"""
Evaluation of the submitted (causal) system, Step 28.

infer.py works frame by frame and only uses the current and past frames, so its scores differ a
little from the earlier whole-video evaluation (eval_system.py), where body height and floor level
were measured over the whole clip. This script runs exactly the infer.py engine (fv_engine) on
every video of the chosen splits and stores, per video, the collapse score over time (highest over
all people, every 0.5 s).

  --run     score the videos (resumable: videos already in --scores are skipped)
  --report  on the VALIDATION split only, choose the decision lag (best average precision; ties ->
            shorter lag) and then the alarm threshold (best F1; ties -> fewer false alarms); report
            every test split once with that choice, next to the current config.yaml setting
  --lags    decision lags to score in one pass (default 0, 0.5, 1.0, 1.5 s)

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


def score_video(path, cfg, pose, gru, device, lags_s):
    """Per decision lag: [(t, highest smoothed score over people)] every stride, as infer.py computes it."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps > 1000:
        src_fps = None
    pose.reset()
    eng = CollapseEngine(cfg, gru, device)
    lags = {l: int(round(l * eng.fps)) for l in lags_s}
    series, last_slot, idx = {l: [] for l in lags_s}, -1, -1
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
        if slot >= eng.next_score:
            eng.next_score = slot + eng.stride
            sc = eng.scores_for_lags(slot, list(lags.values()))
            for l, n in lags.items():
                series[l].append((round(t, 2), round(sc[n], 4)))
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
            series, frames = score_video(r["file"], cfg, pose, gru, dev, a.lags)
            s0 = series[a.lags[0]]
            w.writerow(dict(split=r["split"], dataset=r["dataset"], path=r["path"], has_fall=r["has_fall"],
                            true_start_s=r["true_start_s"], max_score=max([s for _, s in s0] or [0.0]),
                            series=json.dumps({str(l): v for l, v in series.items()}), frames=frames,
                            seconds=round(s0[-1][0], 2) if s0 else 0))
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


def average_precision(rows):
    """Threshold-free ranking quality on one split: average precision of fall videos by score."""
    order = sorted(rows, key=lambda r: -r["max_score"])
    n_pos, tp, ap = sum(r["has_fall"] for r in rows), 0, 0.0
    for i, r in enumerate(order, 1):
        if r["has_fall"]:
            tp += 1
            ap += tp / i
    return ap / max(n_pos, 1)


def best_threshold(rows):
    best = None
    for thr in sorted({r["max_score"] for r in rows} | {1.01}):
        m = metrics(rows, thr)
        if best is None or m["f1"] > best[1]["f1"] + 1e-12 or (abs(m["f1"] - best[1]["f1"]) < 1e-12 and m["fp"] < best[1]["fp"]):
            best = (thr, m)
    return round(best[0], 4), best[1]


def report(a):
    raw = list(csv.DictReader(open(a.scores, newline="")))
    lag_mode = raw and raw[0]["series"].lstrip().startswith("{")
    lags = sorted({float(k) for k in json.loads(raw[0]["series"]).keys()}) if lag_mode else [0.0]

    def rows_for(lag):
        out = []
        for r in raw:
            ser = json.loads(r["series"])
            ser = ser[str(lag)] if lag_mode else ser
            out.append(dict(split=r["split"], has_fall=int(r["has_fall"]), true_start_s=r["true_start_s"],
                            series=ser, max_score=max([s for _, s in ser] or [0.0])))
        return out

    # 1) decision lag: best average precision on VALIDATION (ties -> shorter lag); 2) threshold: best F1 on
    #    VALIDATION (ties -> fewer false alarms). Rule fixed before any test result was seen.
    cand = []
    for lag in lags:
        val = [r for r in rows_for(lag) if r["split"] == "val"]
        if not val:
            sys.exit("no validation videos scored yet")
        cand.append((average_precision(val), -lag, lag))
    ap_best, _, lag = max(cand)
    rows = rows_for(lag)
    thr, vm = best_threshold([r for r in rows if r["split"] == "val"])
    old = load_config(a.config)
    order = ["val", "test_gmdcsa24", "test_staged", "test_oops", "test_own"]
    lines = ["# Submitted (causal) system: evaluation" + (" with decision lag (Step 28b)" if lag_mode else " (Step 28)"), "",
             "infer.py engine on whole videos, frame by frame, current and past frames only. One alarm per video "
             "is counted if the highest collapse score reaches the threshold.", "",
             "Choice on VALIDATION only, rule fixed in advance: decision lag with the best average precision "
             "(ties -> shorter lag), then threshold with the best F1 (ties -> fewer false alarms).", "",
             "| Decision lag | Validation average precision |", "|---|---|"]
    lines += [f"| {l} s | {ap:.4f} |" for ap, _, l in sorted(cand, key=lambda c: c[2])]
    lines += ["", f"Chosen: decision lag **{lag} s**, threshold **{thr}**.", "",
              "| Split | Setting | Videos | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN | Median alarm delay |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ref = rows_for(lags[0])
    for split in order:
        settings = [(f"chosen: lag {lag} s, thr {thr}", rows, thr),
                    (f"config: lag {old['decision_lag_s']} s, thr {old['threshold']}", rows_for(old["decision_lag_s"])
                     if old["decision_lag_s"] in lags else ref, old["threshold"])]
        for name, rs_all, t in settings:
            rs = [r for r in rs_all if r["split"] == split]
            if not rs:
                continue
            m = metrics(rs, t)
            d = f"{m['delay']:.1f} s" if m["delay"] is not None else "n/a"
            lines.append(f"| {split} | {name} | {m['n']} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {d} |")
    out = os.path.join(ROOT, "results", "causal_eval_lags.md" if lag_mode else "causal_eval.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}\nCHOSEN_LAG={lag}\nCHOSEN_THRESHOLD={thr}")


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
    ap.add_argument("--lags", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5],
                    help="decision lags (s) to score; the first is also written as max_score")
    a = ap.parse_args()
    if a.run:
        run(a)
    if a.report:
        report(a)


if __name__ == "__main__":
    main()
