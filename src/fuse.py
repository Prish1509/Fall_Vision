"""
Choose the final model combination on validation (Member 2, Step 22 preparation).

Reads the window probabilities saved by train_rf.py / train_gru.py (--probs-out) and tries:
  models     each model alone, the mean of all models, and the mean of every pair
  score      "fall+fallen"  P(falling) + P(fallen)   (counts lying on the ground)
             "falling"      P(falling) only          (only the fall movement)
  threshold  best-F1 on validation, and precision-first (validation precision >= 0.95)
Each video score is the highest window score after averaging 2 neighbouring windows.

Why "falling only": the false alarms of the first models were mostly people lying down on purpose
(GMDCSA24 "lie down" / "lying" clips): the lying pose looks like "fallen", but the movement into
it is slow, unlike a fall.

Everything is chosen on VALIDATION videos only (best validation F1, ties -> fewer false alarms).
The chosen setting is then reported once on every test split.

Output: results/fusion.md (all settings on validation + chosen setting on tests) and
results/fusion_videos.csv (per-video scores and alarms of the chosen setting).

Usage:
  python src/fuse.py --windows WIN --models rf=probs/rf gru=probs/gru gru_noaug=probs/gru_noaug
"""

import argparse
import csv
import itertools
import os

import numpy as np

from train_rf import choose_thresholds, fmt, video_metrics

SPLITS = ["val", "test_gmdcsa24", "test_staged", "test_oops", "test_own"]
SCORES = {"fall+fallen": lambda p: p[:, 1] + p[:, 2], "falling": lambda p: p[:, 1]}


def video_score(win_score, vid, n):
    score, when = np.zeros(n), np.zeros(n, int)
    for v in range(n):
        idx = np.where(vid == v)[0]
        if len(idx) == 0:
            continue
        s = win_score[idx]
        if len(s) > 1:
            s = np.convolve(s, [0.5, 0.5], mode="same")
        score[v], when[v] = s.max(), idx[int(s.argmax())]
    return score, when


def main():
    ap = argparse.ArgumentParser(description="Pick the model combination and threshold on validation")
    ap.add_argument("--windows", required=True, help="windows folder (for <split>_videos.csv)")
    ap.add_argument("--models", nargs="+", required=True, help="name=probs_folder ...")
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--out", default="results/fusion.md")
    a = ap.parse_args()

    models = dict(m.split("=", 1) for m in a.models)
    data = {}                                     # split -> dict(videos, truth, vid, t, probs{name})
    for split in SPLITS:
        vcsv = os.path.join(a.windows, f"{split}_videos.csv")
        if not os.path.exists(vcsv):
            continue
        videos = list(csv.DictReader(open(vcsv, newline="")))
        probs, vid, t = {}, None, None
        for name, folder in models.items():
            d = np.load(os.path.join(folder, f"{split}.npz"))
            probs[name] = d["proba"]
            if vid is None:
                vid, t = d["vid"], d["t"]
            assert np.array_equal(vid, d["vid"]), f"{name}/{split}: windows differ between models"
        data[split] = dict(videos=videos, vid=vid, t=t, probs=probs,
                           truth=np.array([int(v["has_fall"]) for v in videos], bool))

    names = list(models)
    combos = [(n,) for n in names] + [c for k in range(2, len(names) + 1) for c in itertools.combinations(names, k)]
    va = data["val"]
    rows = []
    for combo in combos:
        for sname, sfun in SCORES.items():
            ws = np.mean([sfun(va["probs"][m]) for m in combo], 0)
            vs, _ = video_score(ws, va["vid"], len(va["videos"]))
            thr_p, thr_f = choose_thresholds(vs, va["truth"], a.min_precision)
            mf, mp = video_metrics(vs, va["truth"], thr_f), video_metrics(vs, va["truth"], thr_p)
            rows.append(dict(combo=combo, score=sname, thr_f=thr_f, thr_p=thr_p, mf=mf, mp=mp))

    lines = ["# Model combination chosen on validation (Step 22 preparation)", "",
             "Every setting below is scored on VALIDATION videos only; the best one (validation F1, ties ->",
             "fewer false alarms) is then reported once on the test splits.", "",
             "| Models | Score | Val F1 (best-F1 thr) | Val P | Val R | Val FP | Val F1 (precision-first thr) | Val FP |",
             "|---|---|---|---|---|---|---|---|"]
    print("Validation (all settings):")
    for r in sorted(rows, key=lambda r: (-r["mf"]["f1"], r["mf"]["fp"])):
        tag = "+".join(r["combo"])
        lines.append(f"| {tag} | {r['score']} | {r['mf']['f1']:.3f} | {r['mf']['precision']:.3f} | "
                     f"{r['mf']['recall']:.3f} | {r['mf']['fp']} | {r['mp']['f1']:.3f} | {r['mp']['fp']} |")
        print(f"  {tag:<22} {r['score']:<12} best-F1 {fmt(r['mf'])}")
    best = min(rows, key=lambda r: (-r["mf"]["f1"], r["mf"]["fp"], len(r["combo"])))
    tag = "+".join(best["combo"])
    print(f"\nChosen on validation: models {tag}, score {best['score']}, "
          f"best-F1 threshold {best['thr_f']:.3f}, precision-first threshold {best['thr_p']:.3f}\n")
    lines += ["", f"**Chosen:** models `{tag}`, score `{best['score']}`, best-F1 threshold {best['thr_f']:.3f}, "
              f"precision-first threshold {best['thr_p']:.3f}", "",
              "## Chosen setting on every split", "",
              "| Split | Threshold | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    out_rows = []
    for split, d in data.items():
        ws = np.mean([SCORES[best["score"]](d["probs"][m]) for m in best["combo"]], 0)
        vs, when = video_score(ws, d["vid"], len(d["videos"]))
        for tname, thr in (("best-F1", best["thr_f"]), ("precision-first", best["thr_p"])):
            m = video_metrics(vs, d["truth"], thr)
            print(f"{split:<14} {tname:<16} {fmt(m)}")
            lines.append(f"| {split} | {tname} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
        for i, v in enumerate(d["videos"]):
            out_rows.append(dict(split=split, dataset=v["dataset"], path=v["path"], has_fall=v["has_fall"],
                                 score=round(float(vs[i]), 3), alarm_best_f1=int(vs[i] >= best["thr_f"]),
                                 alarm_precision_first=int(vs[i] >= best["thr_p"]),
                                 alarm_time_s=round(float(d["t"][when[i]]), 2)))

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
