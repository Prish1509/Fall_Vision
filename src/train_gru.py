"""
GRU fall classifier on 2-second skeleton windows (Member 3, Step 21): the main learned model.

Input: the windows from make_windows.py (30 frames x 92 features, labels normal / falling / fallen).
Model: 2-layer GRU (hidden 128, dropout 0.3) reading the 30 frames in order, then a small
classifier on [last hidden state, mean of hidden states] -> 3 classes. Features are standardised
with the training mean and standard deviation (saved with the model).

Augmentation (training only, each applied with probability 0.5) so the model does not learn the
datasets' cameras and lighting:
  mirror        swap left and right joints, flip x                  (camera on the other side)
  rotate        rotate joint positions by up to +-15 degrees         (tilted camera)
  scale         joint positions x 0.9-1.1                            (body-height estimate errors)
  joint noise   Gaussian noise on positions                          (small, far-away people)
  night         hide 10-40 % of joints and lower all confidences     (dark video: Step 16 showed the
                                                                      torso is unreadable 1 in 3 frames)
  frame drop    blank up to 20 % of frames                           (person lost for a moment)

Training: AdamW, class-weighted cross-entropy, 40 epochs; the epoch with the best validation
window macro-F1 is kept. Thresholds for the video-level alarm are then chosen on validation videos
exactly as for the Random Forest (precision-first: val precision >= 0.95; and best-F1).

Output: results/gru.md, results/gru_videos.csv, model in --model-out (not committed).

Usage:
  python src/train_gru.py --windows /kaggle/input/.../windows
  python src/train_gru.py --windows ... --no-aug --out results/gru_noaug.md     # ablation row
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn

from train_rf import SPLITS_TEST, choose_thresholds, fmt, video_metrics, video_scores

# feature layout (preprocess.py): 0-33 pos (x,y per joint), 34-67 vel, 68-84 conf, 85-91 whole-body
POS, VEL, CONF = slice(0, 34), slice(34, 68), slice(68, 85)
PRESENT = 91
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]      # COCO left <-> right


def load_raw(win_dir, split):
    p = os.path.join(win_dir, f"{split}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    with open(os.path.join(win_dir, f"{split}_videos.csv"), newline="") as f:
        vids = list(csv.DictReader(f))
    return dict(X=d["X"].astype(np.float32), y=d["y"].astype(np.int64), vid=d["vid"], t=d["t"], videos=vids)


def augment(x, rng):
    """x: (B, 30, 92) raw features (numpy, float32). Returns an augmented copy."""
    x = x.copy()
    B = len(x)
    pos = x[:, :, POS].reshape(B, 30, 17, 2).copy()
    vel = x[:, :, VEL].reshape(B, 30, 17, 2).copy()
    conf = x[:, :, CONF].copy()
    drops = []
    for b in range(B):
        if rng.random() < 0.5:                                        # mirror
            pos[b] = pos[b][:, FLIP]; vel[b] = vel[b][:, FLIP]; conf[b] = conf[b][:, FLIP]
            pos[b, :, :, 0] *= -1; vel[b, :, :, 0] *= -1
        if rng.random() < 0.5:                                        # rotate
            a = np.deg2rad(rng.uniform(-15, 15)); c, s = np.cos(a), np.sin(a)
            R = np.array([[c, -s], [s, c]], np.float32)
            pos[b] = pos[b] @ R.T; vel[b] = vel[b] @ R.T
        if rng.random() < 0.5:                                        # scale
            k = rng.uniform(0.9, 1.1); pos[b] *= k; vel[b] *= k
        if rng.random() < 0.5:                                        # joint noise (small people)
            n = rng.normal(0, rng.uniform(0.01, 0.05), pos[b].shape).astype(np.float32)
            pos[b] += n; vel[b, 1:] += (n[1:] - n[:-1]) * 15
        if rng.random() < 0.5:                                        # night: hide joints, lower conf
            hide = rng.random((30, 17)) < rng.uniform(0.1, 0.4)
            pos[b][hide] = 0; vel[b][hide] = 0
            conf[b] *= rng.uniform(0.5, 0.9); conf[b][hide] = 0
        if rng.random() < 0.5:                                        # frame drop (person lost)
            drops.append((b, rng.random(30) < rng.uniform(0.0, 0.2)))
    x[:, :, POS] = pos.reshape(B, 30, 34)
    x[:, :, VEL] = vel.reshape(B, 30, 34)
    x[:, :, CONF] = conf
    for b, drop in drops:
        x[b, drop] = 0
    return x


class GRUNet(nn.Module):
    def __init__(self, n_in=92, hidden=128, layers=2, dropout=0.3, n_cls=3):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_cls))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(torch.cat([h[:, -1], h.mean(1)], 1))


def predict(model, X, mean, std, device, bs=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy((X[i:i + bs] - mean) / std).to(device)
            out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser(description="GRU fall classifier on 2-s skeleton windows")
    ap.add_argument("--windows", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--no-aug", action="store_true", help="train without augmentation (ablation)")
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gru.md")
    ap.add_argument("--model-out", default="/kaggle/working/models/gru.pt")
    a = ap.parse_args()

    from sklearn.metrics import confusion_matrix, f1_score
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va = load_raw(a.windows, "train"), load_raw(a.windows, "val")
    mean = tr["X"].reshape(-1, 92).mean(0).astype(np.float32)
    std = tr["X"].reshape(-1, 92).std(0).astype(np.float32) + 1e-3
    mean[PRESENT], std[PRESENT] = 0.0, 1.0
    counts = np.bincount(tr["y"], minlength=3)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    w = (w / w.mean()).to(device)
    print(f"device {device}, train windows {len(tr['y'])} {counts.tolist()}, val windows {len(va['y'])}, "
          f"augmentation {'off' if a.no_aug else 'on'}", flush=True)

    model = GRUNet(hidden=a.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    best, t0 = (-1, None, 0), time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        order = rng.permutation(len(tr["y"]))
        total = 0.0
        for i in range(0, len(order), a.batch):
            idx = order[i:i + a.batch]
            xb = tr["X"][idx] if a.no_aug else augment(tr["X"][idx], rng)
            xb = torch.from_numpy((xb - mean) / std).to(device)
            yb = torch.from_numpy(tr["y"][idx]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        pv = predict(model, va["X"], mean, std, device)
        f1 = f1_score(va["y"], pv.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
        if f1 > best[0]:
            best = (f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:>3}  loss {total / len(order):.3f}  val macro-F1 {f1:.3f}  "
                  f"best {best[0]:.3f} (epoch {best[2]})  {time.time() - t0:.0f} s", flush=True)
    model.load_state_dict(best[1])
    os.makedirs(os.path.dirname(a.model_out) or ".", exist_ok=True)
    torch.save(dict(state=best[1], mean=mean, std=std, hidden=a.hidden, epoch=best[2],
                    augmentation=not a.no_aug), a.model_out)

    pv = predict(model, va["X"], mean, std, device)
    v_truth = np.array([int(v["has_fall"]) for v in va["videos"]], bool)
    v_score, _ = video_scores(pv, va["vid"], len(va["videos"]))
    thr_p, thr_f1 = choose_thresholds(v_score, v_truth, a.min_precision)

    title = "GRU without augmentation" if a.no_aug else "GRU with augmentation"
    lines = [f"# {title} (Step 21)", "",
             f"2-layer GRU, hidden {a.hidden}, best epoch {best[2]} of {a.epochs} (val window macro-F1 "
             f"{best[0]:.3f}). Thresholds chosen on validation videos only.", "",
             f"- precision-first threshold (val precision >= {a.min_precision}): {thr_p:.3f}",
             f"- best-F1 threshold (val): {thr_f1:.3f}", "",
             "## Video level (one alarm per video or none)", "",
             "| Split | Setting | F1 | Precision | Recall | Accuracy | TP | FP | TN | FN |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows_out = []
    print(f"\nBest epoch {best[2]}; thresholds chosen on val: precision-first {thr_p:.3f}, best-F1 {thr_f1:.3f}\n")
    for split in ["val"] + SPLITS_TEST:
        d = va if split == "val" else load_raw(a.windows, split)
        if d is None:
            continue
        proba = pv if split == "val" else predict(model, d["X"], mean, std, device)
        truth = np.array([int(v["has_fall"]) for v in d["videos"]], bool)
        score, when = video_scores(proba, d["vid"], len(d["videos"]))
        for name, thr in (("precision-first", thr_p), ("best-F1", thr_f1)):
            m = video_metrics(score, truth, thr)
            print(f"{split:<14} {name:<16} {fmt(m)}")
            lines.append(f"| {split} | {name} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                         f"{m['accuracy']:.3f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
        wf1 = f1_score(d["y"], proba.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)
        cm = confusion_matrix(d["y"], proba.argmax(1), labels=[0, 1, 2])
        lines.append(f"| {split} | window macro-F1 | {wf1:.3f} | | | | | | | |")
        print(f"{split:<14} window macro-F1 {wf1:.3f}   confusion (rows true normal/falling/fallen): {cm.tolist()}")
        for i, v in enumerate(d["videos"]):
            rows_out.append(dict(split=split, dataset=v["dataset"], path=v["path"], has_fall=v["has_fall"],
                                 score=round(float(score[i]), 3), alarm_precision_first=int(score[i] >= thr_p),
                                 alarm_best_f1=int(score[i] >= thr_f1),
                                 alarm_time_s=round(float(d["t"][int(when[i])]), 2) if not np.isnan(when[i]) else ""))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_out = a.out.replace(".md", "_videos.csv")
    with open(csv_out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wr.writeheader()
        wr.writerows(rows_out)
    print(f"\nSaved {a.out}, {csv_out} and the model in {a.model_out}")


if __name__ == "__main__":
    main()
