"""
Random Forest on window summaries (Member 3, Step 19): the first learned model.

Each 2-second window (30 frames x 92 features, make_windows.py) is summarised by five numbers
per feature: mean, standard deviation, minimum, maximum and change (last frame - first frame),
giving 460 numbers. A Random Forest (scikit-learn) classifies the window as normal / falling /
fallen. Classes are weighted so the rare falling and fallen windows count as much as normal ones.

Video-level decision (the same scoring as the rules: one alarm per video or none):
    fall score of a window  = P(falling) + P(fallen), averaged over 2 neighbouring windows
    video score             = highest fall score in the video
    alarm                   = video score >= threshold
The threshold is chosen on the VALIDATION videos only: the lowest-recall-cost threshold that
keeps precision >= 0.95 (the instructor's "no false positives"), and, for comparison, the
threshold with the best F1. Test splits are only scored, never used for any choice.

Output: results/rf.md (all numbers), results/rf_videos.csv (per-video scores and decisions),
the trained model in --model-out (not committed: large file).

Usage:
  python src/train_rf.py --windows /kaggle/working/windows
"""

import argparse
import csv
import os
import time

import numpy as np

SPLITS_TEST = ["test_gmdcsa24", "test_staged", "test_oops", "test_own"]
STATS = ["mean", "std", "min", "max", "delta"]


def summarise(X):
    X = X.astype(np.float32)
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1] - X[:, 0]], 1)


def load(win_dir, split):
    p = os.path.join(win_dir, f"{split}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    with open(os.path.join(win_dir, f"{split}_videos.csv"), newline="") as f:
        vids = list(csv.DictReader(f))
    return dict(X=summarise(d["X"]), y=d["y"].astype(int), vid=d["vid"], t=d["t"], videos=vids)


def video_scores(proba, vid, n_videos):
    fall = proba[:, 1] + proba[:, 2]
    score = np.zeros(n_videos)
    when = np.full(n_videos, np.nan)
    for v in range(n_videos):
        idx = np.where(vid == v)[0]
        if len(idx) == 0:
            continue
        s = fall[idx]
        if len(s) > 1:
            s = np.convolve(s, [0.5, 0.5], mode="same")
        score[v] = s.max()
        when[v] = idx[int(s.argmax())]
    return score, when


def video_metrics(score, truth, thr):
    pred = score >= thr
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=p, recall=r, f1=f1, accuracy=acc)


def choose_thresholds(score, truth, min_precision=0.95):
    cands = np.unique(np.r_[score, 1.01])
    best_f1, best_prec = None, None
    for thr in cands:
        m = video_metrics(score, truth, thr)
        if best_f1 is None or m["f1"] > best_f1[1]["f1"]:
            best_f1 = (thr, m)
        if m["precision"] >= min_precision and m["tp"] > 0:
            if best_prec is None or m["recall"] > best_prec[1]["recall"] or \
               (m["recall"] == best_prec[1]["recall"] and thr < best_prec[0]):
                best_prec = (thr, m)
    return (best_prec or best_f1)[0], best_f1[0]


def fmt(m):
    return (f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  Acc={m['accuracy']:.3f}  "
            f"[TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}]")


def main():
    ap = argparse.ArgumentParser(description="Random Forest fall classifier on 2-s windows")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--out", default="results/rf.md")
    ap.add_argument("--model-out", default="/kaggle/working/models/rf.joblib")
    ap.add_argument("--probs-out", default="", help="folder to save window probabilities per split")
    a = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, f1_score
    import joblib

    tr, va = load(a.windows, "train"), load(a.windows, "val")
    print(f"train windows {len(tr['y'])}, val windows {len(va['y'])}, features {tr['X'].shape[1]}")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=a.trees, min_samples_leaf=2, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=0)
    rf.fit(tr["X"], tr["y"])
    print(f"trained in {time.time() - t0:.0f} s")
    os.makedirs(os.path.dirname(a.model_out) or ".", exist_ok=True)
    joblib.dump(rf, a.model_out)

    pv = rf.predict_proba(va["X"])
    v_truth = np.array([int(v["has_fall"]) for v in va["videos"]], bool)
    v_score, _ = video_scores(pv, va["vid"], len(va["videos"]))
    thr_p, thr_f1 = choose_thresholds(v_score, v_truth, a.min_precision)

    from preprocess import FEATURE_NAMES
    names = [f"{s}:{n}" for s in STATS for n in FEATURE_NAMES]

    lines = ["# Random Forest (Step 19)", "",
             f"{a.trees} trees, 460 window-summary features, class-balanced. "
             f"Thresholds chosen on validation videos only.", "",
             f"- precision-first threshold (val precision >= {a.min_precision}): {thr_p:.3f}",
             f"- best-F1 threshold (val): {thr_f1:.3f}", "",
             "## Video level (one alarm per video or none)", "",
             "| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows_out = []
    print(f"\nThresholds chosen on val: precision-first {thr_p:.3f}, best-F1 {thr_f1:.3f}\n")
    for split in ["val"] + SPLITS_TEST:
        d = va if split == "val" else load(a.windows, split)
        if d is None:
            continue
        proba = pv if split == "val" else rf.predict_proba(d["X"])
        if a.probs_out:
            os.makedirs(a.probs_out, exist_ok=True)
            np.savez_compressed(os.path.join(a.probs_out, f"{split}.npz"), proba=proba, vid=d["vid"], t=d["t"])
        truth = np.array([int(v["has_fall"]) for v in d["videos"]], bool)
        score, when = video_scores(proba, d["vid"], len(d["videos"]))
        for name, thr in (("precision-first", thr_p), ("best-F1", thr_f1)):
            m = video_metrics(score, truth, thr)
            print(f"{split:<14} {name:<16} {fmt(m)}")
            lines.append(f"| {split} | {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
        wf1 = f1_score(d["y"], proba.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(d["y"], proba.argmax(1), labels=[0, 1, 2])
        lines += [f"| {split} | window macro-F1 | {wf1:.3f} | | | | | | | |"]
        print(f"{split:<14} window macro-F1 {wf1:.3f}   confusion (rows true normal/falling/fallen):"
              f" {cm.tolist()}")
        for i, v in enumerate(d["videos"]):
            rows_out.append(dict(split=split, dataset=v["dataset"], path=v["path"], has_fall=v["has_fall"],
                                 score=round(float(score[i]), 3), alarm_precision_first=int(score[i] >= thr_p),
                                 alarm_best_f1=int(score[i] >= thr_f1),
                                 alarm_time_s=round(float(d["t"][int(when[i])]), 2) if not np.isnan(when[i]) else ""))

    imp = rf.feature_importances_
    top = np.argsort(imp)[::-1][:15]
    lines += ["", "## Most useful features", ""]
    print("\nMost useful features:")
    for j in top:
        nm = names[j]
        lines.append(f"- {nm}: {imp[j]:.3f}")
        print(f"  {nm:<28} {imp[j]:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_out = a.out.replace(".md", "_videos.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nSaved {a.out}, {csv_out} and the model in {a.model_out}")


if __name__ == "__main__":
    main()
