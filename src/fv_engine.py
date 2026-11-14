"""
FallVision causal engine: person collapse detection frame by frame, using only the current and past
frames (used by infer.py and app/live_fall.py).

Per processed frame:
  1. YOLO11m-pose + ByteTrack give every person a box, 17 keypoints and a track id.
  2. Track ids are mapped to persons; a new id that appears within track_join_gap_s near where a
     person was lost is the same person (same rule as preprocess.join_tracks).
  3. Every stride_s, each person seen recently gets the 92-feature sequence of their last
     history_s seconds (preprocess.features_from_tracks: body height and floor level come only
     from these past frames) and the GRU scores their latest 2-s window.
  4. score = P(falling) + P(fallen), averaged with that person's previous window; an alarm is
     raised when it reaches the threshold. A person can alarm again after rearm_s below it.
  5. When people are small (standing height below small_person_px in a 1280-wide frame, measured
     over the past history_s), the pose model switches to input_size_small, and back.

All settings come from config.yaml. State is created per video (new_video()), never shared.
"""

import math
import os
from collections import deque

import cv2
import numpy as np
import torch
import yaml

from detect_falls import REF_W, load_gru
from preprocess import features_from_tracks

HEAD_TOP_SPEED = 86          # feature index of the head-top vertical speed (preprocess.FEATURE_NAMES)


def load_config(path):
    """config.yaml -> flat settings dict; model paths are made absolute (relative to the yaml)."""
    raw = yaml.safe_load(open(path))
    base = os.path.dirname(os.path.abspath(path))
    p, m, t, a = raw["pose"], raw["models"], raw["temporal"], raw["alarm"]
    absp = lambda x: x if os.path.isabs(x) else os.path.join(base, x)
    return dict(
        project=raw.get("project", "person_collapse"), label=a["label"],
        pose_model=absp(m["pose"]), gru=absp(m["classifier"]),
        imgsz=int(p["input_size"]), imgsz_small=int(p["input_size_small"]),
        small_px=float(p["small_person_px"]), det_conf=float(p["detection_confidence"]),
        tracker=p["tracker"], model_fps=float(raw["frame_sampling"]["model_fps"]),
        window_s=float(t["window_s"]), stride_s=float(t["stride_s"]), history_s=float(t["history_s"]),
        join_gap_s=float(t["track_join_gap_s"]), join_v2=bool(t["fall_aware_join"]),
        score=a["score"], threshold=float(a["threshold"]), rearm_s=float(a["rearm_s"]),
        config_path=os.path.abspath(path))


def torch_device(device):
    """'cpu', 'cuda', 'cuda:N' or 'N' -> (torch device string, ultralytics device string, warning)."""
    d = str(device).strip().lower()
    if d.isdigit():
        d = f"cuda:{d}"
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu", "cpu", f"device {device} requested but CUDA is not available; running on cpu"
    if d.startswith("cuda"):
        idx = d.split(":")[1] if ":" in d else "0"
        return f"cuda:{idx}", idx, None
    if d == "cpu":
        return "cpu", "cpu", None
    raise ValueError(f"unsupported device '{device}' (use cpu, cuda, cuda:N or N)")


class PoseTracker:
    """YOLO11-pose + ByteTrack, loaded once; reset() gives a new video a fresh tracker."""

    def __init__(self, cfg, yolo_device):
        from ultralytics import YOLO
        if not os.path.isfile(cfg["pose_model"]) and not os.path.isdir(cfg["pose_model"]):
            raise FileNotFoundError(f"pose model not found: {cfg['pose_model']}")
        self.cfg, self.device = cfg, yolo_device
        self.model = YOLO(cfg["pose_model"], task="pose")
        self.track(np.zeros((cfg["imgsz"], cfg["imgsz"], 3), np.uint8), cfg["imgsz"])   # warm-up
        self.reset()

    def reset(self):
        if getattr(self.model, "predictor", None) is not None and hasattr(self.model.predictor, "trackers"):
            del self.model.predictor.trackers

    def track(self, img, imgsz):
        r = self.model.track(img, persist=True, tracker=self.cfg["tracker"], classes=[0], imgsz=imgsz,
                             conf=self.cfg["det_conf"], device=self.device, verbose=False)[0]
        det = dict(ids=[], boxes=[], kpts=[])
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            det["ids"] = r.boxes.id.int().cpu().tolist()
            det["boxes"] = r.boxes.xyxy.cpu().numpy().tolist()
            kxy = r.keypoints.xy.cpu().numpy()
            kcf = r.keypoints.conf
            kcf = kcf.cpu().numpy() if kcf is not None else np.ones(kxy.shape[:2])
            det["kpts"] = np.concatenate([kxy, kcf[..., None]], axis=2).tolist()
        return det


class Person:
    def __init__(self, pid):
        self.pid = pid
        self.obs = deque()           # (slot, box, kpts) of the past history_s
        self.prev_raw = 0.0          # raw score of the previous window (smoothing)
        self.score = 0.0             # latest smoothed collapse score
        self.alarm = None            # dict while this person is in an alarm
        self.low_since = None


class CollapseEngine:
    """Per-video state. Feed frames in order with add_frame(); call step() on every processed frame."""

    def __init__(self, cfg, gru, device):
        self.cfg, self.device = cfg, device
        self.model, self.mean, self.std = gru
        self.fps = cfg["model_fps"]
        self.win = int(round(cfg["window_s"] * self.fps))
        self.stride = max(1, int(round(cfg["stride_s"] * self.fps)))
        self.hist = int(round(cfg["history_s"] * self.fps))
        self.max_gap = int(round(cfg["join_gap_s"] * self.fps))
        self.people, self.owner, self.next_pid = {}, {}, 1
        self.next_score = self.win
        self.imgsz = cfg["imgsz"]
        self.recent_boxes = deque()
        self.frame_size = None
        self.n_alarms = 0

    # ---- identity ----
    def _person_for(self, tid, slot, box, taken):
        if tid in self.owner and self.owner[tid] in self.people:
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
        if best:
            p = best[1]
        else:
            p = self.people[self.next_pid] = Person(self.next_pid)
            self.next_pid += 1
        self.owner[tid] = p.pid
        return p

    def add_frame(self, slot, det, frame_size):
        """Store one processed frame's detections; returns the person id of every detection."""
        self.frame_size = self.frame_size or frame_size
        taken, pids = set(), []
        for tid, b, k in zip(det["ids"], det["boxes"], det["kpts"]):
            p = self._person_for(tid, slot, b, taken)
            taken.add(p.pid)
            pids.append(p.pid)
            if not (p.obs and p.obs[-1][0] == slot):
                p.obs.append((slot, b, k))
            while p.obs and p.obs[0][0] < slot - self.hist:
                p.obs.popleft()
        for pid in [q for q, p in self.people.items() if not p.obs or p.obs[-1][0] < slot - self.hist]:
            del self.people[pid]
        self.owner = {t: q for t, q in self.owner.items() if q in self.people}
        self.recent_boxes.append((slot, det["boxes"]))
        while self.recent_boxes and self.recent_boxes[0][0] < slot - self.hist:
            self.recent_boxes.popleft()
        return pids

    def pose_input_size(self, slot):
        """Input size for the next frame: switch to the larger one while people are small (past frames)."""
        if slot % self.stride == 0 and slot >= self.win and self.frame_size:
            h = [b[3] - b[1] for _, bs in self.recent_boxes for b in bs]
            if h:
                std_h = float(np.percentile(h, 90)) * REF_W / max(self.frame_size)
                self.imgsz = self.cfg["imgsz_small"] if std_h < self.cfg["small_px"] else self.cfg["imgsz"]
        return self.imgsz

    # ---- scoring ----
    def step(self, slot, t_now):
        """Score the latest window of every person when a stride is due. Returns the new alarms."""
        if slot < self.next_score:
            return []
        self.next_score = slot + self.stride
        batch = []
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
            return []
        X = np.stack([f["feats"][-self.win:] for _, f, _ in batch]).astype(np.float32)
        with torch.no_grad():
            prob = torch.softmax(self.model(torch.from_numpy((X - self.mean) / self.std).to(self.device)), 1)
        prob = prob.cpu().numpy()
        new = []
        for (p, f, frames), pr in zip(batch, prob):
            raw = float(pr[1] + pr[2]) if self.cfg["score"] == "fall+fallen" else float(pr[1])
            p.score = 0.5 * (raw + p.prev_raw)
            p.prev_raw = raw
            if p.alarm is None and p.score >= self.cfg["threshold"]:
                ts = f["feats"][:, HEAD_TOP_SPEED]
                lo = len(ts) - self.win
                st = lo + int(np.argmin(ts[lo:]))
                while st > lo and ts[st - 1] < -0.3:
                    st -= 1
                self.n_alarms += 1
                p.alarm = dict(event_id=self.n_alarms, person_id=p.pid, decision_timestamp_s=round(t_now, 3),
                               onset_estimate_s=round(frames[st]["t"], 3), confidence=round(p.score, 4),
                               bbox=[round(float(v), 1) for v in p.obs[-1][1]])
                p.low_since = None
                new.append(p.alarm)
            elif p.alarm is not None:
                if p.score < self.cfg["threshold"]:
                    p.low_since = slot if p.low_since is None else p.low_since
                    if slot - p.low_since >= self.cfg["rearm_s"] * self.fps:
                        p.alarm, p.low_since = None, None
                else:
                    p.low_since = None
        return new

    def predictions(self, slot, det, pids):
        """Per-frame predictions: every person visible in this frame whose latest collapse score is at
        or above the threshold (label person_collapse); empty list otherwise."""
        out = []
        for b, pid in zip(det["boxes"], pids):
            p = self.people.get(pid)
            if p is not None and p.score >= self.cfg["threshold"]:
                out.append(dict(label=self.cfg["label"], confidence=round(p.score, 4), person_id=pid,
                                bbox_xyxy=[round(float(v), 1) for v in b]))
        return out


def draw(img, det, pids, engine):
    """Boxes and skeletons; red with COLLAPSE for a person in an alarm."""
    from baseline_fall import SKELETON
    for b, k, pid in zip(det["boxes"], det["kpts"], pids):
        p = engine.people.get(pid)
        alarm = p is not None and p.alarm is not None
        c = (0, 0, 255) if alarm else (0, 200, 0)
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3 if alarm else 2)
        k = np.asarray(k)
        for a, bb in SKELETON:
            if k[a, 2] >= 0.3 and k[bb, 2] >= 0.3:
                cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[bb, :2])), c, 2)
        cv2.putText(img, f"#{pid}{' COLLAPSE' if alarm else ''}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    if any(p.alarm is not None for p in engine.people.values()):
        cv2.putText(img, "PERSON COLLAPSE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    return img


def new_gru(cfg, device):
    if not os.path.isfile(cfg["gru"]):
        raise FileNotFoundError(f"classifier model not found: {cfg['gru']}")
    return load_gru(cfg["gru"], device)
