"""
Rule-based fall detection baseline.

Pipeline (per video):
  1. Resample to ~15 fps so timing rules work for any input frame rate.
  2. YOLO pose model + ByteTrack -> box + 17 COCO keypoints per person, per frame.
  3. Per-person features, all divided by the person's body height (camera/distance independent):
       - torso angle from vertical (shoulder-mid to hip-mid)
       - box aspect ratio (width / height)
       - drop of the top of the box (the head) within the last second
       - horizontal speed (for standing / walking / running)
  4. Rule-based state machine:
       FALL ALERT  = a fast drop happened recently  AND  the person stays "down" for HOLD seconds
                     AND  lies still (v2).
  5. Localization: every alert records WHO (track id), WHEN (estimated fall start, alert time)
     and WHERE (bounding box of the person who fell).

Rules v2 (instructor's requirements: flag falls from sitting, avoid false positives):
  fix 1  standing height remembered from upright frames only; drops must start from a not-down pose
  fix 2  "down" smoothed over 0.5 s (noisy torso angles reset the timer)
  fix 3  a collapsed box counts as down unless the torso is clearly upright (= sitting)
  fix 4  the person must lie still while down (push-ups / exercise keep moving)
  hold   2.0 s instead of 1.5 s (bending and picking things up recover within ~1.5 s)

Ideas used (cite in the report):
  - Two-stage definition: a fast "fall" movement, then a held "fallen" state (OmniFall, arXiv 2505.19889)
  - Pose front end: Ultralytics YOLO11-pose; tracking: ByteTrack (arXiv 2110.06864)
  - Dividing by body height: perspective projection, DS5602 Lecture 2

Usage:
  python src/baseline_fall.py --video input.mp4 --out output.mp4 --csv falls.csv --loc-csv boxes.csv
  python src/baseline_fall.py --video input.mp4 --rules v1      # original rules, for comparison
  python src/baseline_fall.py --video night.mp4 --conf 0.10      # night: keep more detections

Night: brightening dark frames (--night clahe / denoise) was tested on simulated-night clips and
made results WORSE (fewer people detected), so it is off by default. A lower detection threshold
kept more detections. Tune both on your own night clips.
"""

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# COCO keypoint indices
NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


# --------------------------------------------------------------------------------------
# Thresholds (tune these on training subjects only, never on your test set)
# --------------------------------------------------------------------------------------
@dataclass
class Params:
    drop_ratio: float = 0.25      # top of box (head) must drop >= this * body height ...
    drop_window: float = 1.0      # ... within this many seconds  -> "fast drop"
    down_angle: float = 45.0      # torso this many degrees from vertical -> "down"
    down_aspect: float = 0.8      # or box wider than tall (used when torso not visible)
    hold: float = 2.0             # must stay down this long after the drop -> ALERT
    recent: float = 4.0           # the drop must have happened within this many seconds
    kp_conf: float = 0.30         # minimum keypoint confidence to trust a joint
    walk_speed: float = 0.25      # body-heights / second
    run_speed: float = 1.20
    # ---- v2 fixes (set to False / 0 to switch one off, for the ablation table) ----
    ref_upright_only: bool = True  # fix 1: remember standing height from upright frames only
    drop_from_upright: bool = True # fix 1: a drop only counts if it starts from a not-down pose
    down_smooth: float = 0.5       # fix 2: "down" = majority vote over this many seconds
    collapse_ratio: float = 0.65   # fix 3: torso hidden + box <= 65% of standing height = down
    still_max: float = 0.15        # fix 4: while down, head may move <= this * body height in 1 s
    rot_deg: float = 0.0           # optional extra trigger: torso rotates this much within 1 s


def params_v1():
    """The original rules (before the sitting-fall and false-positive fixes)."""
    return Params(hold=1.5, recent=2.5, ref_upright_only=False, drop_from_upright=False,
                  down_smooth=0.0, collapse_ratio=0.0, still_max=0.0, rot_deg=0.0)


# --------------------------------------------------------------------------------------
# Night / low light (DS5602 Lecture 3: colour and intensity)
# --------------------------------------------------------------------------------------
_CLAHE = None


def enhance_low_light(frame, dark_below=80.0, target=110.0):
    """If the frame is dark, boost local contrast (CLAHE on the L channel of LAB) and then
    brighten with a gamma curve so the mean brightness moves towards `target`.
    Bright frames are returned unchanged."""
    global _CLAHE
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if gray_mean >= dark_below:
        return frame
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    m = max(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()), 1.0)
    gamma = min(max(math.log(target / 255.0) / math.log(m / 255.0), 0.3), 1.0)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


# --------------------------------------------------------------------------------------
# Step 1-2: detection, tracking, pose
# --------------------------------------------------------------------------------------
def denoise_brighten(frame, dark_below=80.0, target=100.0):
    """Gentler night option: smooth sensor noise first, then a gamma curve only (no CLAHE)."""
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if gray_mean >= dark_below:
        return frame
    out = cv2.bilateralFilter(frame, 7, 40, 7)
    m = max(gray_mean, 1.0)
    gamma = min(max(math.log(target / 255.0) / math.log(m / 255.0), 0.3), 1.0)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


_POSE_MODELS = {}


def _pose_model(model_name):
    """Load the pose model once per program run and reuse it; each video gets a fresh tracker, so
    results are the same as loading it again, only faster when several videos are processed."""
    from ultralytics import YOLO
    model = _POSE_MODELS.get(model_name)
    if model is None:
        model = _POSE_MODELS[model_name] = YOLO(model_name, task="pose")
    elif getattr(model, "predictor", None) is not None and hasattr(model.predictor, "trackers"):
        del model.predictor.trackers                        # new video: new ByteTrack state
    return model


def extract_tracks(video_path, model_name="yolo11m-pose.pt", target_fps=15.0, imgsz=640,
                   keep_frames=False, night="off", conf=0.25, small_px=0):
    """Run pose + tracking. Returns dict with effective fps and per-frame detections.
    night: "off" = no change (default), "clahe" = CLAHE + gamma on dark frames,
           "denoise" = denoise + gamma on dark frames.
    conf: detection confidence threshold (a lower value keeps more people in dark video).
    small_px: if > 0, people whose box is shorter than this many pixels are re-checked on an
              enlarged crop (small_person.py, Step 15). 0 = off."""
    model = _pose_model(model_name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps > 240:   # missing / NaN / bogus
        src_fps = 30.0
    step = max(1, int(round(src_fps / target_fps)))
    fps = src_fps / step

    frames, images = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if night == "clahe":
                frame = enhance_low_light(frame)
            elif night == "denoise":
                frame = denoise_brighten(frame)
            r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                            imgsz=imgsz, conf=conf, verbose=False)[0]
            det = {"t": len(frames) / fps, "ids": [], "boxes": [], "kpts": []}
            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                det["ids"] = r.boxes.id.int().cpu().tolist()
                det["boxes"] = r.boxes.xyxy.cpu().numpy().tolist()
                kxy = r.keypoints.xy.cpu().numpy()
                kcf = r.keypoints.conf
                kcf = kcf.cpu().numpy() if kcf is not None else np.ones(kxy.shape[:2])
                det["kpts"] = np.concatenate([kxy, kcf[..., None]], axis=2).tolist()
                if small_px:
                    from small_person import refine_small_people
                    det["kpts"], det["refined"] = refine_small_people(
                        model, frame, det["boxes"], det["kpts"], small_px=small_px, imgsz=imgsz,
                        conf=conf)
            frames.append(det)
            if keep_frames:
                images.append(frame)
        idx += 1
    cap.release()
    return {"video": str(video_path), "fps": fps, "frames": frames}, images


# --------------------------------------------------------------------------------------
# Step 3-4: features + rules, one object per tracked person
# --------------------------------------------------------------------------------------
@dataclass
class PersonState:
    # (t, top_y, cx, h, raw_down, upright, torso_angle), up to 20 s at 15 fps
    hist: deque = field(default_factory=lambda: deque(maxlen=300))
    down_since: float = None
    alerted: bool = False
    label: str = "standing"


def is_sitting(k, p, ang):
    """Upright torso + thighs that are horizontal (side view) or look very short because they
    point at the camera (front view). Used only for the display label."""
    k = np.asarray(k)
    if ang is not None and ang >= 35:
        return False
    torso = None
    if min(k[L_SH, 2], k[R_SH, 2], k[L_HIP, 2], k[R_HIP, 2]) >= p.kp_conf:
        torso = abs((k[L_HIP, 1] + k[R_HIP, 1]) / 2 - (k[L_SH, 1] + k[R_SH, 1]) / 2)
    for hip, knee in ((L_HIP, 13), (R_HIP, 14)):
        if k[hip, 2] >= p.kp_conf and k[knee, 2] >= p.kp_conf:
            dx, dy = k[knee, :2] - k[hip, :2]
            if math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)) >= 50:      # side view
                return True
            if torso and torso > 1e-6 and dy < 0.35 * torso:                  # front view
                return True
    return False


def torso_angle(k, p):
    """Degrees from vertical: 0 = upright, 90 = horizontal. None if joints not visible."""
    k = np.asarray(k)
    if min(k[L_SH, 2], k[R_SH, 2], k[L_HIP, 2], k[R_HIP, 2]) < p.kp_conf:
        return None
    sh = (k[L_SH, :2] + k[R_SH, :2]) / 2
    hp = (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    v = sh - hp                                  # hip -> shoulder; image y grows downward
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    cos_up = float(-v[1]) / n                    # cosine with the "up" direction (0, -1)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_up))))   # 0 upright ... 180 head-down


class FallDetector:
    """
    Note on tracking: when someone falls, their box changes shape so fast that the tracker
    often gives them a NEW id (e.g. #1 -> #4 -> #5). So a "fast drop" is stored with its
    location, not only with the track id, and any track that ends up lying near that spot
    soon afterwards inherits it.
    """

    def __init__(self, fps, params=None):
        self.fps, self.p = fps, params or Params()
        self.people = {}
        self.drops = []    # (time, cx, cy, body_h): fast downward movements seen anywhere
        # fall alerts with localization:
        # (track_id, alert_time_s, fall_start_s, [x1, y1, x2, y2] box of the person at the alert)
        self.events = []
        self.alert_spots = []  # (time, cx, cy, body_h) to avoid duplicate alerts

    def _near(self, items, t, cx, cy, max_age):
        for (te, ex, ey, eh) in reversed(items):
            if 0 <= t - te <= max_age and math.hypot(cx - ex, cy - ey) <= 1.5 * eh:
                return te
        return None

    def _first_near(self, items, t, cx, cy, max_age):
        """Earliest matching drop = estimated start time of this fall (temporal localization)."""
        times = [te for (te, ex, ey, eh) in items
                 if 0 <= t - te <= max_age and math.hypot(cx - ex, cy - ey) <= 1.5 * eh]
        return min(times) if times else None

    def update(self, t, tid, box, kpts):
        p = self.p
        st = self.people.setdefault(tid, PersonState())
        x1, y1, x2, y2 = box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        k = np.asarray(kpts)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        ang = torso_angle(k, p)
        aspect = w / h

        # body-height reference (Lecture 2: every distance is divided by it)
        if p.ref_upright_only:
            # fix 1: standing height from upright frames of the last 10 s, so it does not
            # shrink while someone is lying (that made tiny movements look like big drops)
            ups = [e[3] for e in st.hist if t - e[0] <= 10.0 and e[5]]
            if ups:
                body_h = max(ups + [h])
            else:   # no upright view of this id yet (new id mid-fall): borrow from a nearby drop
                borrowed = next((eh for (te, ex, ey, eh) in reversed(self.drops)
                                 if 0 <= t - te <= p.recent and math.hypot(cx - ex, cy - ey) <= 1.5 * eh), 0.0)
                body_h = max([e[3] for e in st.hist] + [h, borrowed])
        else:
            body_h = max([e[3] for e in st.hist if t - e[0] <= 3.0] + [h])   # v1

        raw_down = (ang is not None and ang >= p.down_angle) or (ang is None and aspect >= p.down_aspect) \
            or aspect >= 1.6
        if p.collapse_ratio and h <= p.collapse_ratio * body_h and (
                ang is None or (aspect >= p.down_aspect and ang >= 30)):
            # fix 3: the box collapsed to a fraction of standing height and the torso is hidden
            # (fell back onto a bed) or tilted and the box is no longer tall (lying towards the
            # camera). A clearly upright torso (< 30 deg) means sitting, so it is NOT down.
            raw_down = True
        upright = (not raw_down) and (ang is None or ang < 30) and aspect < 0.8
        st.hist.append((t, y1, cx, h, raw_down, upright, ang))

        # fast drop: top of the box (head) now vs its highest point within drop_window
        win = [e for e in st.hist if t - e[0] <= p.drop_window]
        starts = [e for e in win if not e[4]] if p.drop_from_upright else win
        trigger = bool(starts) and (y1 - min(e[1] for e in starts)) / body_h >= p.drop_ratio
        if p.rot_deg and ang is not None:   # optional: fast torso rotation also counts
            angs = [e[6] for e in win if e[6] is not None]
            trigger = trigger or (bool(angs) and ang - min(angs) >= p.rot_deg)
        if trigger:
            self.drops.append((t, cx, cy, body_h))

        # fix 2: smooth "down" (noisy torso angles made it flicker and reset the hold timer)
        if p.down_smooth:
            votes = [e[4] for e in st.hist if t - e[0] <= p.down_smooth]
            down = sum(votes) >= 0.5 * len(votes)
        else:
            down = raw_down

        # fix 4: a fallen person lies still; push-ups / exercise keep moving up and down
        still = True
        if p.still_max:
            ys = [e[1] for e in st.hist if t - e[0] <= 1.0]
            still = (max(ys) - min(ys)) / body_h <= p.still_max

        # horizontal speed in body-heights per second (over ~1 s)
        sp_win = [e for e in st.hist if t - e[0] <= 1.0]
        speed = 0.0
        if len(sp_win) >= 2 and sp_win[-1][0] > sp_win[0][0]:
            speed = abs(sp_win[-1][2] - sp_win[0][2]) / body_h / (sp_win[-1][0] - sp_win[0][0])

        # state machine
        if down:
            st.down_since = t if st.down_since is None else st.down_since
        else:
            st.down_since = None
            st.alerted = False                           # person is up again -> re-arm

        recent_drop = self._near(self.drops, t, cx, cy, p.recent) is not None
        held = st.down_since is not None and t - st.down_since >= p.hold
        if down and held and recent_drop and still and not st.alerted:
            st.alerted = True
            if self._near(self.alert_spots, t, cx, cy, 5.0) is None:   # same fall, other id
                start = self._first_near(self.drops, t, cx, cy, p.recent + p.hold)
                self.events.append((tid, round(t, 2), round(start, 2),
                                    [round(float(v), 1) for v in box]))
            self.alert_spots.append((t, cx, cy, body_h))

        if st.alerted:
            st.label = "FALL"
        elif down:
            st.label = "possible fall" if recent_drop else "lying"
        elif speed >= p.run_speed:
            st.label = "running"
        elif speed >= p.walk_speed:
            st.label = "walking"
        else:
            st.label = "sitting" if is_sitting(k, p, ang) else "standing"
        return st.label


def run_rules(track_data, params=None):
    det = FallDetector(track_data["fps"], params)
    labels = []
    for fr in track_data["frames"]:
        labels.append({tid: det.update(fr["t"], tid, b, k)
                       for tid, b, k in zip(fr["ids"], fr["boxes"], fr["kpts"])})
    return det.events, labels


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------
COLORS = {"FALL": (0, 0, 255), "possible fall": (0, 140, 255), "lying": (0, 200, 255),
          "running": (255, 120, 0), "walking": (0, 200, 0), "standing": (200, 200, 200),
          "sitting": (255, 200, 0)}


def draw(frame, fr, labels, p):
    for tid, b, k in zip(fr["ids"], fr["boxes"], fr["kpts"]):
        lab = labels.get(tid, "")
        c = COLORS.get(lab, (255, 255, 255))
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3 if lab == "FALL" else 2)
        k = np.asarray(k)
        for a, bb in SKELETON:
            if k[a, 2] >= p.kp_conf and k[bb, 2] >= p.kp_conf:
                cv2.line(frame, tuple(map(int, k[a, :2])), tuple(map(int, k[bb, :2])), c, 2)
        cv2.putText(frame, f"#{tid} {lab}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, c, 2)
    if any(v == "FALL" for v in labels.values()):
        cv2.putText(frame, "FALL DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 255), 3)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Rule-based fall detection baseline")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None, help="annotated output video (.mp4)")
    ap.add_argument("--csv", default=None, help="CSV log of fall events (who, when, where)")
    ap.add_argument("--loc-csv", default=None,
                    help="CSV with the box of every possible-fall / FALL person in every frame")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640, help="raise to 960/1280 for small people")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--rules", choices=["v1", "v2"], default="v2")
    ap.add_argument("--night", choices=["off", "clahe", "denoise"], default="off",
                    help="low-light handling for dark frames (test on your own night clips)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="person detection threshold; try 0.10-0.15 for night video")
    ap.add_argument("--small-px", type=int, default=0,
                    help="re-check people shorter than this many pixels on a zoomed crop (0 = off)")
    a = ap.parse_args()

    p = params_v1() if a.rules == "v1" else Params()
    data, images = extract_tracks(a.video, a.model, a.fps, a.imgsz, keep_frames=bool(a.out),
                                  night=a.night, conf=a.conf, small_px=a.small_px)
    events, labels = run_rules(data, p)

    print(f"{len(data['frames'])} frames at {data['fps']:.1f} fps")
    if events:
        for tid, t, start, box in events:
            print(f"FALL detected: person #{tid}, fall started ~{start:.1f} s, alert at {t:.1f} s, "
                  f"box (x1, y1, x2, y2) = {box}")
    else:
        print("No fall detected")

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "track_id", "fall_start_s", "alert_time_s", "x1", "y1", "x2", "y2"])
            for tid, t, start, box in events:
                w.writerow([a.video, tid, start, t, *box])

    if a.loc_csv:   # spatial + temporal localization, frame by frame
        with open(a.loc_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "track_id", "state", "x1", "y1", "x2", "y2"])
            for fr, lab in zip(data["frames"], labels):
                for tid, b in zip(fr["ids"], fr["boxes"]):
                    if lab.get(tid) in ("possible fall", "FALL"):
                        w.writerow([round(fr["t"], 2), tid, lab[tid], *[round(v, 1) for v in b]])

    if a.out and images:
        hgt, wid = images[0].shape[:2]
        vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), data["fps"], (wid, hgt))
        for img, fr, lab in zip(images, data["frames"], labels):
            vw.write(draw(img, fr, lab, p))
        vw.release()
        print(f"Saved annotated video to {a.out}")


if __name__ == "__main__":
    main()
