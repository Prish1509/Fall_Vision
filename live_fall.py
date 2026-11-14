"""
FallVision LIVE: fall or no fall while the video is running.

    python live_fall.py                    # webcam 0
    python live_fall.py 1                  # another webcam
    python live_fall.py "D:/videos/a.mp4"  # a video file, played as it runs
    python live_fall.py rtsp://user:pass@192.168.1.10/stream   # CCTV / IP camera
Options: --out annotated.mp4   --csv falls.csv   --no-show (no window)   --fast (files: no waiting)
Quit the window with q or Esc.

Works for any camera, resolution, orientation and number of people:
  - every person the tracker finds gets their own skeleton history and their own fall score;
  - frames are sampled by their time stamps onto the 15-per-second grid the model was trained on,
    whatever the camera's frame rate or how fast this computer is (missed slots are filled the same
    way as in training, or marked unseen);
  - body size and floor level are measured per person from the last few seconds, not fixed;
  - when people are small (far camera) the pose model switches to input 960, as in detect_falls.py;
  - a person who gets a new tracker id at the moment of the fall is joined back to themselves.
Every setting comes from configs/fallvision.json (chosen on validation); nothing is set in this file.
"""

import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from baseline_fall import Params, draw                      # noqa: E402
from detect_falls import REF_W, load_gru                   # noqa: E402
from preprocess import features_from_tracks, join_tracks    # noqa: E402


def load_config(path):
    cfg = json.load(open(path))
    for k in ("gru", "pose_model"):
        p = os.path.join(ROOT, cfg[k])
        if os.path.exists(p):
            cfg[k] = p
    return cfg


def pick_pose_model(name):
    """GPU: the PyTorch model. CPU: the same weights converted once to OpenVINO (much faster)."""
    if torch.cuda.is_available():
        return name
    ov = os.path.join(ROOT, os.path.splitext(os.path.basename(name))[0] + "_openvino_model")
    if os.path.isdir(ov):
        return ov
    try:
        import openvino  # noqa: F401
        from ultralytics import YOLO
        print("No GPU: converting the pose model to OpenVINO once (~1 min)...")
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            YOLO(name).export(format="openvino", dynamic=True, half=False, verbose=False)
        finally:
            os.chdir(cwd)
        return ov if os.path.isdir(ov) else name
    except Exception:
        return name


class LatestFrame:
    """Reads a camera / stream in the background and keeps only the newest frame, so a slow
    computer never falls behind a live camera (old frames are skipped, not queued)."""

    def __init__(self, cap):
        self.cap, self.q, self.ok = cap, queue.Queue(maxsize=1), True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                self.ok = False
                break
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put((time.monotonic(), f))

    def read(self, timeout=5.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class Person:
    def __init__(self, pid):
        self.pid = pid
        self.obs = deque()          # (slot index, box, kpts)
        self.alarm = None           # dict while this person is flagged
        self.low_since = None       # slot where the score went below threshold after an alarm
        self.prev_raw = 0.0         # raw score of the previous window (for the same smoothing as offline)


class LiveFall:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.fps = float(cfg["model_fps"])
        self.win = int(round(cfg["window_s"] * self.fps))
        self.step = max(1, int(round(cfg["stride_s"] * self.fps)))
        self.hist = int(round(cfg["history_s"] * self.fps))
        self.max_gap = int(round(join_tracks.__defaults__[0] * self.fps))   # same gap as preprocess.join_tracks
        self.model, self.mean, self.std = load_gru(cfg["gru"], device)
        self.people, self.owner, self.next_pid = {}, {}, 1      # owner: tracker id -> person id
        self.events = []

    # ---- identity: tracker ids -> persons (same rule as preprocess.join_tracks) ----
    def person_for(self, tid, slot, box, taken):
        if tid in self.owner:
            return self.people[self.owner[tid]]
        lo = -2 if self.cfg["join_v2"] else 0
        cx0, cy0, h0 = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[3] - box[1]
        best = None
        for p in self.people.values():
            if not p.obs or p.pid in taken:
                continue
            end, b1, _ = p.obs[-1]
            if not (lo < slot - end <= self.max_gap):
                continue
            cx1, cy1, h1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2, b1[3] - b1[1]
            d = math.hypot(cx0 - cx1, cy0 - cy1)
            if self.cfg["join_v2"]:
                d = min(d, math.hypot(cx0 - cx1, box[3] - b1[3]))
            if d <= max(h0, h1) and (best is None or d < best[0]):
                best = (d, p)
        p = best[1] if best else Person(self.next_pid)
        if not best:
            self.people[p.pid] = p
            self.next_pid += 1
        self.owner[tid] = p.pid
        return p

    def add_frame(self, slot, det):
        taken = set()
        for tid, b, k in zip(det["ids"], det["boxes"], det["kpts"]):
            p = self.person_for(tid, slot, b, taken)
            taken.add(p.pid)
            if p.obs and p.obs[-1][0] == slot:
                continue
            p.obs.append((slot, b, k))
            while p.obs and p.obs[0][0] < slot - self.hist:
                p.obs.popleft()
        for pid in [pid for pid, p in self.people.items() if not p.obs or p.obs[-1][0] < slot - self.hist]:
            del self.people[pid]
            self.owner = {t: q for t, q in self.owner.items() if q != pid}

    # ---- scoring: same features, windows, smoothing and threshold as detect_falls.py ----
    def score(self, slot):
        """Called every stride. Returns new alarms [(pid, alarm_t, start_t, box, score)]."""
        new, batch = [], []
        for p in self.people.values():
            if not p.obs or p.obs[-1][0] < slot - self.win:
                continue
            first = max(p.obs[0][0], slot - self.hist)
            frames = [dict(t=s / self.fps, ids=[], boxes=[], kpts=[]) for s in range(first, slot + 1)]
            for s, b, k in p.obs:
                if s >= first:
                    frames[s - first].update(ids=[p.pid], boxes=[b], kpts=[k])
            if len(frames) < self.win:
                continue
            f = features_from_tracks(dict(fps=self.fps, frames=frames))
            if f is None or f["present"][-self.win:].mean() < 0.5:
                continue
            batch.append((p, f, frames))
        if not batch:
            return new
        X = np.stack([f["feats"][-self.win:] for _, f, _ in batch]).astype(np.float32)
        with torch.no_grad():
            prob = torch.softmax(self.model(torch.from_numpy((X - self.mean) / self.std).to(self.device)), 1)
        prob = prob.cpu().numpy()
        for (p, f, frames), pr in zip(batch, prob):
            raw = float(pr[1] + pr[2]) if self.cfg["score"] == "fall+fallen" else float(pr[1])
            s = 0.5 * (raw + p.prev_raw)                       # same smoothing as detect_falls.py
            p.prev_raw = raw
            if p.alarm is None and s >= self.cfg["threshold"]:
                ts = f["feats"][:, 86]                          # head-top vertical speed
                lo = len(ts) - self.win
                st = lo + int(np.argmin(ts[lo:]))
                while st > lo and ts[st - 1] < -0.3:
                    st -= 1
                box = p.obs[-1][1]
                p.alarm = dict(t=frames[-1]["t"], start=frames[st]["t"], box=[round(float(v), 1) for v in box],
                               score=round(s, 3))
                p.low_since = None
                new.append((p.pid, p.alarm["t"], p.alarm["start"], p.alarm["box"], p.alarm["score"]))
            elif p.alarm is not None:
                if s < self.cfg["threshold"]:
                    p.low_since = slot if p.low_since is None else p.low_since
                    if slot - p.low_since >= self.cfg["rearm_s"] * self.fps:
                        p.alarm, p.low_since = None, None      # person recovered: may alarm again
                else:
                    p.low_since = None
        return new


def standing_height(boxes, frame_w, frame_h):
    h = [b[3] - b[1] for b in boxes]
    return float(np.percentile(h, 90)) * REF_W / max(frame_w, frame_h) if h else None


def main():
    ap = argparse.ArgumentParser(description="FallVision live: fall / no fall while the video runs")
    ap.add_argument("source", nargs="?", default="0", help="webcam number, video file, or stream URL")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "fallvision.json"))
    ap.add_argument("--out", default=None, help="save the annotated video here (.mp4)")
    ap.add_argument("--csv", default=None, help="save one row per fall here")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--fast", action="store_true", help="video files: process as fast as possible")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if not os.path.isfile(cfg["gru"]):
        sys.exit(f"Model not found: {cfg['gru']} (see configs/fallvision.json)")
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose = YOLO(pick_pose_model(cfg["pose_model"]), task="pose")
    live = LiveFall(cfg, device)

    for sz in (640, 960):                                      # warm up before the clock starts
        pose.predict(np.zeros((sz, sz, 3), np.uint8), imgsz=sz, verbose=False)

    src = int(a.source) if a.source.isdigit() else a.source
    is_file = isinstance(src, str) and os.path.isfile(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Cannot open {a.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps > 240:
        src_fps = 30.0
    reader = None if is_file else LatestFrame(cap)
    show = not a.no_show
    print(f"Source: {a.source} ({'file' if is_file else 'live'}), model {os.path.basename(cfg['gru'])}, "
          f"threshold {cfg['threshold']}, device {device}. Press q to quit." if show else
          f"Source: {a.source}, model {os.path.basename(cfg['gru'])}, threshold {cfg['threshold']}, device {device}")

    imgsz, recent_boxes, frame_size = 640, deque(), None
    writer, slot, last_slot, t0, n_proc, tick = None, 0, -1, None, 0, time.monotonic()
    frame_idx, next_score = 0, live.win
    while True:
        if is_file:
            ok, img = cap.read()
            if not ok:
                break
            t = frame_idx / src_fps
            frame_idx += 1
            slot = int(round(t * live.fps))
            if slot == last_slot:
                continue                                       # more source frames than model slots
            if not a.fast:                                     # play at the video's own speed
                t0 = time.monotonic() - t if t0 is None else t0
                wait = t - (time.monotonic() - t0)
                if wait > 0:
                    time.sleep(wait)
        else:
            got = reader.read()
            if got is None:
                break
            ts, img = got
            t0 = ts if t0 is None else t0
            slot = int(round((ts - t0) * live.fps))
            if slot == last_slot:
                continue
        last_slot = slot
        if frame_size is None:
            frame_size = (img.shape[1], img.shape[0])

        r = pose.track(img, persist=True, tracker="bytetrack.yaml", classes=[0], imgsz=imgsz,
                       conf=cfg["det_conf"], verbose=False)[0]
        det = dict(t=slot / live.fps, ids=[], boxes=[], kpts=[])
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            det["ids"] = r.boxes.id.int().cpu().tolist()
            det["boxes"] = r.boxes.xyxy.cpu().numpy().tolist()
            kxy = r.keypoints.xy.cpu().numpy()
            kcf = r.keypoints.conf
            kcf = kcf.cpu().numpy() if kcf is not None else np.ones(kxy.shape[:2])
            det["kpts"] = np.concatenate([kxy, kcf[..., None]], axis=2).tolist()
        live.add_frame(slot, det)

        # small people (far camera): use input 960, like detect_falls.py; re-checked every few seconds
        recent_boxes.append((slot, det["boxes"]))
        while recent_boxes and recent_boxes[0][0] < slot - live.hist:
            recent_boxes.popleft()
        if slot % live.step == 0 and slot >= live.win:
            h = standing_height([b for _, bs in recent_boxes for b in bs], *frame_size)
            want = 960 if (h is not None and h < cfg["small_px"]) else 640
            if want != imgsz:
                print(f"  {slot / live.fps:6.1f} s: people {'small' if want == 960 else 'large'} "
                      f"(standing height {h:.0f} px in a 1280-wide frame) -> pose input {want}")
                imgsz = want

        if slot >= next_score:
            next_score = slot + live.step
            for pid, at, st, box, sc in live.score(slot):
                live.events.append(dict(person_id=pid, fall_start_s=st, alarm_time_s=at,
                                        x1=box[0], y1=box[1], x2=box[2], y2=box[3], score=sc))
                print(f"  FALL: person #{pid}, fall started ~{st:.1f} s, alarm at {at:.1f} s, "
                      f"box {box}, score {sc:.2f}", flush=True)

        n_proc += 1
        if n_proc == int(5 * live.fps) and not (is_file and a.fast):
            rate = n_proc / max(time.monotonic() - tick, 1e-6)
            if rate < 0.8 * live.fps:
                print(f"  Warning: this computer processes {rate:.1f} frames/s, below the {live.fps:.0f} the model "
                      f"expects; missing frames are filled in, but a GPU or OpenVINO gives more reliable results.")
        if show or a.out:
            labels = {tid: ("FALL" if live.people.get(live.owner.get(tid)) is not None
                            and live.people[live.owner[tid]].alarm is not None else "")
                      for tid in det["ids"]}
            shown = draw(img.copy(), dict(ids=[live.owner.get(t, t) for t in det["ids"]], boxes=det["boxes"],
                                          kpts=det["kpts"]),
                         {live.owner.get(t, t): v for t, v in labels.items()}, Params())
            rate = n_proc / max(time.monotonic() - tick, 1e-6)
            cv2.putText(shown, f"{slot / live.fps:5.1f} s  {rate:4.1f} fps  people {len(det['ids'])}",
                        (20, shown.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if a.out:
                if writer is None:
                    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), live.fps,
                                             (shown.shape[1], shown.shape[0]))
                writer.write(shown)
            if show:
                try:
                    cv2.imshow("FallVision live (q to quit)", shown)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except cv2.error:
                    print("No screen available: continuing without a window (use --no-show).")
                    show = False

    if reader is not None:
        reader.ok = False
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["person_id", "fall_start_s", "alarm_time_s", "x1", "y1", "x2",
                                              "y2", "score"])
            w.writeheader()
            w.writerows(live.events)
    rate = n_proc / max(time.monotonic() - tick, 1e-6)
    print("\n" + "=" * 60)
    if live.events:
        print(f"RESULT: FALL DETECTED ({len(live.events)} alarm{'s' if len(live.events) > 1 else ''})")
        for e in live.events:
            print(f"  person #{e['person_id']}: fall started ~{e['fall_start_s']:.1f} s, alarm at "
                  f"{e['alarm_time_s']:.1f} s, box ({e['x1']}, {e['y1']}, {e['x2']}, {e['y2']})")
    else:
        print("RESULT: NO FALL")
    print(f"Processed {n_proc} frames at {rate:.1f} frames/s (real time needs {live.fps:.0f})")
    if a.out:
        print(f"Annotated video saved: {a.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
