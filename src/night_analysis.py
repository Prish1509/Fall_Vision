"""
Night analysis (Member 3, Step 16).

Part A - why do falls get missed at night?
    For every GMDCSA24 video, compares the daylight pose cache with the night-simulated pose
    cache and measures, for the main (largest) person:
      det %    frames in which a person is detected at all
      torso %  frames in which both shoulders and both hips are confident (>= 0.30), i.e. the
               torso angle can be computed
      conf     mean keypoint confidence
    Each is measured over the whole clip and around the fall (1 s before to 4 s after the
    labelled fall start), because the person lying on the floor is the hardest to see.

Part B - can the rules be adjusted for night without new false alarms?
    Tries a small grid of rule settings (hold time, keypoint-confidence threshold, drop ratio).
    Settings are chosen on Subjects 1-2 only, using day AND night together, and a setting is
    allowed only if it gives no more false alarms than the current rules, by day or by night.
    The chosen setting is then reported on held-out Subjects 3-4.

No video is decoded again if the caches exist (they were made by eval_gmdcsa24.py).

Usage:
  python src/night_analysis.py --day-root GMDCSA24 --day-cache cache_day \
                               --night-root GMDCSA24_night --night-cache cache_night
"""

import argparse
import itertools
import os
from dataclasses import replace

import numpy as np

from baseline_fall import Params
from eval_gmdcsa24 import TEST_SUBJECTS, TUNE_SUBJECTS, get_tracks, load_labels, score

TORSO = (5, 6, 11, 12)          # COCO: left/right shoulder, left/right hip


def visibility(track_data, t0=None, t1=None, kp_conf=0.30):
    """det %, torso %, mean keypoint confidence for the largest person, optionally in [t0, t1]."""
    n = det = torso = 0
    confs = []
    for f in track_data["frames"]:
        if t0 is not None and not (t0 <= f["t"] <= t1):
            continue
        n += 1
        if not f["boxes"]:
            continue
        det += 1
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in f["boxes"]]
        k = np.asarray(f["kpts"][int(np.argmax(areas))])
        confs.append(float(k[:, 2].mean()))
        torso += bool(min(k[j, 2] for j in TORSO) >= kp_conf)
    if n == 0:
        return None
    return 100.0 * det / n, 100.0 * torso / n, (float(np.mean(confs)) if confs else 0.0)


def mean_rows(rows):
    rows = [r for r in rows if r is not None]
    return tuple(float(np.mean([r[i] for r in rows])) for i in range(3)) if rows else (0, 0, 0)


def combine(a, b):
    """Add the counts of two score() results and recompute F1."""
    tp, fp, fn = a["tp"] + b["tp"], a["fp"] + b["fp"], a["fn"] + b["fn"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def line(m):
    return (f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"[TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}]")


def main():
    ap = argparse.ArgumentParser(description="Day vs night pose visibility + night rule tuning")
    ap.add_argument("--day-root", required=True)
    ap.add_argument("--day-cache", required=True)
    ap.add_argument("--night-root", required=True)
    ap.add_argument("--night-cache", required=True)
    ap.add_argument("--night-conf", type=float, default=0.25,
                    help="detection threshold used for the night cache (0.25 or 0.10)")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="results/night_analysis.md")
    a = ap.parse_args()

    os.makedirs(a.day_cache, exist_ok=True)
    os.makedirs(a.night_cache, exist_ok=True)
    day_items, night_items = load_labels(a.day_root), load_labels(a.night_root)
    by_rel = {i["rel"]: v for v, i in night_items.items()}
    pairs = [(v, by_rel[i["rel"]]) for v, i in day_items.items() if i["rel"] in by_rel]
    print(f"{len(pairs)} videos in both day and night folders")

    day_tr, night_tr = {}, {}
    for n, (dv, nv) in enumerate(pairs, 1):
        day_tr[dv] = get_tracks(dv, a.day_cache, a.model, a.imgsz)
        night_tr[nv] = get_tracks(nv, a.night_cache, a.model, a.imgsz, conf=a.night_conf)
        if n % 20 == 0 or n == len(pairs):
            print(f"  loaded {n}/{len(pairs)}", flush=True)

    out = ["# Night analysis (Step 16)", "",
           f"Night copy: gamma 2.2, gain 0.45, noise 4 (simulate_night.py); "
           f"night detection threshold {a.night_conf}.", ""]

    # ---------------- Part A: visibility ----------------
    base = Params()
    groups = {}
    missed = []
    for dv, nv in pairs:
        info = day_items[dv]
        kind = "Fall" if info["is_fall"] else "ADL"
        for name, tr, v in (("day", day_tr, dv), ("night", night_tr, nv)):
            groups.setdefault((kind, name, "clip"), []).append(visibility(tr[v], kp_conf=base.kp_conf))
            if info["is_fall"] and info["fall_start"] is not None:
                s = info["fall_start"]
                groups.setdefault((kind, name, "fall"), []).append(
                    visibility(tr[v], s - 1.0, s + 4.0, base.kp_conf))
        if info["is_fall"]:
            m_day, _ = score({dv: info}, day_tr, base, {info["subject"]})
            m_night, _ = score({nv: night_items[nv]}, night_tr, base, {info["subject"]})
            if m_day["tp"] and not m_night["tp"]:
                s = info["fall_start"]
                vd = visibility(day_tr[dv], s - 1, s + 4, base.kp_conf) if s is not None else None
                vn = visibility(night_tr[nv], s - 1, s + 4, base.kp_conf) if s is not None else None
                missed.append((info["rel"], vd, vn, info["description"]))

    out += ["## A. What the pose model sees (largest person)", "",
            "| Videos | Window | Light | Person detected % | Torso visible % | Mean keypoint conf |",
            "|---|---|---|---|---|---|"]
    print("\nA. Pose visibility (largest person)")
    for kind in ("Fall", "ADL"):
        for win in ("clip", "fall"):
            for name in ("day", "night"):
                rows = groups.get((kind, name, win))
                if not rows:
                    continue
                d, t, c = mean_rows(rows)
                wtxt = "whole clip" if win == "clip" else "fall -1 s to +4 s"
                out.append(f"| {kind} | {wtxt} | {name} | {d:.0f} | {t:.0f} | {c:.2f} |")
                print(f"  {kind:<4} {wtxt:<18} {name:<5}  detected {d:5.1f} %   torso {t:5.1f} %   conf {c:.2f}")

    out += ["", f"Falls caught by day but missed at night: {len(missed)}", "",
            "| Video | Day: detected / torso % | Night: detected / torso % | Description |",
            "|---|---|---|---|"]
    print(f"\n  Falls caught by day but missed at night: {len(missed)}")
    for rel, vd, vn, desc in missed:
        fd = f"{vd[0]:.0f} / {vd[1]:.0f}" if vd else "n/a"
        fn_ = f"{vn[0]:.0f} / {vn[1]:.0f}" if vn else "n/a"
        out.append(f"| {rel} | {fd} | {fn_} | {desc[:60]} |")
        print(f"    {rel:<22} day {fd:<9} night {fn_:<9} {desc[:50]}")

    # ---------------- Part B: rule settings for night ----------------
    d_items = {v: day_items[v] for v, _ in pairs}
    n_items = {nv: night_items[nv] for _, nv in pairs}
    ref_d, _ = score(d_items, day_tr, base, TUNE_SUBJECTS)
    ref_n, _ = score(n_items, night_tr, base, TUNE_SUBJECTS)
    # current values first, so a change is kept only if it is strictly better
    grid = dict(hold=[2.0, 1.5, 1.0], kp_conf=[0.30, 0.20, 0.15], drop_ratio=[0.25, 0.20])
    best = None
    for combo in itertools.product(*grid.values()):
        p = replace(base, **dict(zip(grid, combo)))
        md, _ = score(d_items, day_tr, p, TUNE_SUBJECTS)
        mn, _ = score(n_items, night_tr, p, TUNE_SUBJECTS)
        if md["fp"] > ref_d["fp"] or mn["fp"] > ref_n["fp"]:
            continue                                   # no new false alarms allowed
        f = combine(md, mn)
        if best is None or f > best[0] + 1e-9:
            best = (f, p)
    chosen = best[1] if best else base
    ch = {k: getattr(chosen, k) for k in grid}

    out += ["", "## B. Rule settings (chosen on Subjects 1-2, day + night, no extra false alarms)", "",
            f"Searched: {grid}", f"Chosen: {ch}", "",
            "| Rules | Light | Held-out Subjects 3-4 |", "|---|---|---|"]
    print(f"\nB. Rule settings chosen on Subjects 1-2 (day + night, no extra false alarms): {ch}")
    for label, p in (("current v2", base), ("chosen", chosen)):
        for name, items, tr in (("day", d_items, day_tr), ("night", n_items, night_tr)):
            m, _ = score(items, tr, p, TEST_SUBJECTS)
            out.append(f"| {label} | {name} | {line(m)} |")
            print(f"  {label:<10} {name:<5}  S3-4: {line(m)}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
