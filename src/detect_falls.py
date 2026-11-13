"""
FallVision: the complete fall detector for any video (Member 1, Step 22).

Pipeline:
  1. YOLO11m-pose + ByteTrack on frames resampled to about 15 fps (baseline_fall.extract_tracks).
     If the people in the video are small (standing height < --small-px pixels) the video is
     run again at input 960, which Step 15b showed helps far-away people.
  2. Broken tracks are joined (preprocess.join_tracks): a person keeps one id through the fall.
  3. EVERY person tracked for at least 2 s gets their own 92-feature sequence (preprocess.py), so
     the person who falls does not have to be the most visible one.
  4. The learned models score overlapping 2-s windows (every 0.5 s): one or more GRUs (train_gru.py)
     and optionally the Random Forest (train_rf.py); their probabilities are averaged. The fall score
     of a window is P(falling) + P(fallen) or P(falling) only (--score), averaged with its neighbour.
     Models, score and threshold are the ones chosen on validation by fuse.py.
  5. Alarm when the score reaches the threshold.
     Localization: WHO (person id), WHEN (estimated fall start and alarm time), WHERE (the person's
     box at the alarm).
  6. The rule-based detector (baseline_fall.run_rules) runs on the same skeletons and is reported
     next to the GRU as a second opinion.

Outputs: printed alarms, --csv (one row per alarm), --out (video with boxes, skeletons and labels).

Usage:
  python src/detect_falls.py --video clip.mp4 --gru models/gru.pt --out clip_falls.mp4 --csv clip_falls.csv
"""

import argparse
import csv
import os

import cv2
import numpy as np
import torch

from baseline_fall import Params, draw, extract_tracks, run_rules
from preprocess import features_from_tracks, join_tracks
from train_gru import GRUNet

WIN, STRIDE_S = 30, 0.5

REF_W = 1280   # the 150-px small-person limit was set on 1280x720 GMDCSA24 video (Steps 15b-15c)


def standing_height(video, data):
    """90th percentile of box heights, rescaled to a 1280-pixel-wide frame.

    Raw pixels depend on the camera: a far person 200 px tall in a 1920x1080 phone video is only
    about 67 px after YOLO scales the frame to 640, yet looked "big enough" in raw pixels. Rescaling
    to the 1280-wide frame the limit was chosen on makes the check mean the same for every video
    (unchanged for 1280x720 video)."""
    heights = [b[3] - b[1] for f in data["frames"] for b in f["boxes"]]
    if not heights:
        return None
    cap = cv2.VideoCapture(video)
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    side = max(w, h) or REF_W
    return float(np.percentile(heights, 90)) * REF_W / side


def load_gru(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = GRUNet(hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["mean"], ck["std"]


def person_tracks(data):
    """Split the tracked video into one track_data dict per joined person (>= 2 s visible)."""
    frames, fps = data["frames"], data["fps"]
    people = {}
    for jid, items in join_tracks(frames, fps).items():
        if len(items) < WIN:
            continue
        per = [dict(t=f["t"], ids=[], boxes=[], kpts=[]) for f in frames]
        for i, b, k in items:
            if not per[i]["ids"]:
                per[i].update(ids=[jid], boxes=[b], kpts=[k])
        people[jid] = dict(fps=fps, frames=per)
    return people


def window_scores(grus, rf, feats, present, fps, device, score="fall+fallen"):
    """Window end indices, smoothed fall scores and mean class probabilities for one person."""
    step = max(1, int(round(STRIDE_S * fps)))
    ends = [e for e in range(WIN, len(feats) + 1, step) if present[e - WIN:e].mean() >= 0.5]
    if not ends:
        return np.array([], int), np.array([]), np.zeros((0, 3))
    X = np.stack([feats[e - WIN:e] for e in ends]).astype(np.float32)
    probs = []
    with torch.no_grad():
        for model, mean, std in grus:
            probs.append(torch.softmax(model(torch.from_numpy((X - mean) / std).to(device)), 1).cpu().numpy())
    if rf is not None:
        from train_rf import summarise
        probs.append(rf.predict_proba(summarise(X)))
    p = np.mean(probs, 0)
    s = p[:, 1] + p[:, 2] if score == "fall+fallen" else p[:, 1].copy()
    if len(s) > 1:
        s = np.convolve(s, [0.5, 0.5], mode="same")
    return np.array(ends), s, p


def alarms_for_video(data, grus, rf, device, threshold, score="fall+fallen"):
    """One alarm per person who falls: (person id, alarm time, estimated fall start, box, score).

    Used by detect_falls.py on a video and by eval_system.py on cached skeletons, so the scored
    system and the demonstrated system are exactly the same code.
    """
    fps = data["fps"]
    alarms = []
    for pid, td in person_tracks(data).items():
        f = features_from_tracks(td)
        if f is None:
            continue
        ends, s, p = window_scores(grus, rf, f["feats"], f["present"], fps, device, score)
        hit = np.where(s >= threshold)[0]
        if len(hit) == 0:
            continue
        k = hit[0]
        alarm_i = ends[k] - 1
        # fall start: the fastest head drop inside the alarm window, then back to where the drop began
        ts = f["feats"][:, 86]                                       # head-top vertical speed
        lo = max(0, ends[k] - WIN)
        start_i = lo + int(np.argmin(ts[lo:ends[k]]))
        while start_i > lo and ts[start_i - 1] < -0.3:
            start_i -= 1
        frames = td["frames"]
        bi = next((j for j in range(alarm_i, -1, -1) if frames[j]["boxes"]), None)
        box = [round(float(v), 1) for v in frames[bi]["boxes"][0]] if bi is not None else []
        alarms.append((pid, round(float(frames[alarm_i]["t"]), 2), round(float(frames[start_i]["t"]), 2),
                       box, round(float(s[hit].max()), 3)))
    return alarms


def main():
    ap = argparse.ArgumentParser(description="FallVision: detect and localize falls in a video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--gru", nargs="+", required=True, help="one or more models saved by train_gru.py")
    ap.add_argument("--rf", default=None, help="optional Random Forest saved by train_rf.py (rf.joblib)")
    ap.add_argument("--score", choices=["fall+fallen", "falling"], default="fall+fallen")
    ap.add_argument("--threshold", type=float, default=0.835, help="chosen on validation (fuse.py)")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--small-px", type=int, default=150, help="re-run at input 960 if the standing height (in a 1280-wide frame) is below this")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="annotated output video (.mp4)")
    ap.add_argument("--csv", default=None, help="CSV with one row per alarm")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grus = [load_gru(g, device) for g in a.gru]
    rf = None
    if a.rf:
        import joblib
        rf = joblib.load(a.rf)

    imgsz = 640
    data, images = extract_tracks(a.video, a.model, 15.0, imgsz, keep_frames=bool(a.out), conf=a.conf)
    # standing height = 90th percentile of box heights (a person lying down has a short box,
    # so the median would wrongly call a lying person "small")
    h_std = standing_height(a.video, data)
    if h_std is not None and h_std < a.small_px:
        print(f"People are small (standing height {h_std:.0f} px in a 1280-wide frame): re-running at input 960")
        imgsz = 960
        data, images = extract_tracks(a.video, a.model, 15.0, imgsz, keep_frames=bool(a.out), conf=a.conf)
    fps = data["fps"]
    print(f"{len(data['frames'])} frames at {fps:.1f} fps, input {imgsz}")

    alarms = alarms_for_video(data, grus, rf, device, a.threshold, a.score)

    rule_events, rule_labels = run_rules(data, Params())
    print("\nGRU (main model):")
    if alarms:
        for pid, t, st, box, sc in alarms:
            print(f"  FALL: person #{pid}, fall started ~{st:.1f} s, alarm at {t:.1f} s, "
                  f"box (x1, y1, x2, y2) = {box}, score {sc:.2f}")
    else:
        print("  no fall")
    print("Rules (second opinion):", "; ".join(f"person #{e[0]} at {e[1]:.1f} s" for e in rule_events)
          if rule_events else "no fall")

    if a.csv:
        with open(a.csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["video", "person_id", "fall_start_s", "alarm_time_s", "x1", "y1", "x2", "y2",
                        "score", "rules_agree"])
            for pid, t, st, box, sc in alarms:
                agree = any(abs(e[1] - t) <= 4.0 for e in rule_events)
                w.writerow([a.video, pid, st, t, *(box or ["", "", "", ""]), sc, int(agree)])

    if a.out and images:
        alarm_times = {pid: t for pid, t, *_ in alarms}
        joined = join_tracks(data["frames"], fps)
        owner = {}                                 # (frame index, raw id) -> joined id
        for jid, items in joined.items():
            for i, b, k in items:
                owner[(i, tuple(np.round(b, 1)))] = jid
        h, w_ = images[0].shape[:2]
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_, h))
        for i, (img, fr) in enumerate(zip(images, data["frames"])):
            labels = {}
            for tid, b in zip(fr["ids"], fr["boxes"]):
                jid = owner.get((i, tuple(np.round(b, 1))), tid)
                labels[tid] = "FALL" if jid in alarm_times and fr["t"] >= alarm_times[jid] else \
                    rule_labels[i].get(tid, "")
            vw.write(draw(img, fr, labels, Params()))
        vw.release()
        print(f"Saved annotated video to {a.out}")


if __name__ == "__main__":
    main()
