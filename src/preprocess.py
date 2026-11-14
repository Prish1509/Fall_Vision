"""
Skeleton preprocessing (Member 2, Step 17).

Turns each cached pose file (cache_poses.py) into one clean, fixed-rate feature sequence for the
main person in the video, ready for 2-second windows (Step 18) and the learned models.

Per video:
  1. Join broken tracks. When a person falls, the tracker often gives them a new id. A track that
     starts within 1.5 s of another ending, near the same place (centre distance < 1 body height),
     is treated as the same person.
  2. Main person = the joined track that is visible longest (ties: larger box).
  3. Fill short gaps (up to 0.5 s) by linear interpolation; longer gaps stay empty and are marked
     in `present`, so the model can tell "not seen" from "seen".
  4. Normalise (DS5602 Lecture 2): positions relative to the hip centre, divided by the standing
     body height (90th percentile of box height in the clip), so camera distance and image size do
     not matter.
  5. Features per frame (F = 92):
       34  joint x, y relative to the hip centre / body height
       34  joint velocities (body heights per second)
       17  joint confidences (low at night or when hidden)
        7  whole-body cues: head-top height above the clip's lowest box bottom, its vertical speed,
           box height / standing height, box width / height, torso angle / 90 deg, hip-centre
           vertical speed, present flag

Output: <out>/<dataset>/<path>.npz with arrays t, feats (T x 92), present, fps, plus a manifest.csv.

Usage:
  python src/preprocess.py --poses /kaggle/input/.../poses --out /kaggle/working/features
"""

import argparse
import csv
import math
import os
import pickle

import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
N_FEATS = 92
FEATURE_NAMES = ([f"pos_{a}{j}" for j in range(17) for a in "xy"] +
                 [f"vel_{a}{j}" for j in range(17) for a in "xy"] +
                 [f"conf_{j}" for j in range(17)] +
                 ["top_height", "top_speed", "height_ratio", "aspect", "torso_angle", "hip_speed", "present"])
assert len(FEATURE_NAMES) == N_FEATS


JOIN_V2 = False   # set True (eval_own.py / detect_falls.py --join-v2) to use the version below


def join_tracks(frames, fps, max_gap_s=1.5):
    """Return {joined_id: [(frame_idx, box, kpts), ...]} after linking broken tracks.

    v2 (JOIN_V2 = True), for the moment of a fall: the tracker often starts the new id in the very
    frame the old one ends (or 1-2 frames before), and the box changes from tall (standing) to wide
    (lying), so its centre jumps. v2 also accepts an overlap of up to 2 frames and measures the
    distance between the boxes' bottom centres (where the person touches the ground) as well as
    their centres, both within one body height."""
    tracks = {}
    for i, f in enumerate(frames):
        for tid, b, k in zip(f["ids"], f["boxes"], f["kpts"]):
            tracks.setdefault(tid, []).append((i, b, k))
    order = sorted(tracks, key=lambda t: tracks[t][0][0])          # by start frame
    joined = {}
    for tid in order:
        start, b0, _ = tracks[tid][0]
        cx0, cy0, h0 = (b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2, b0[3] - b0[1]
        best = None
        for jid, items in joined.items():
            end, b1, _ = items[-1]
            gap = start - end
            if not ((-2 if JOIN_V2 else 0) < gap <= max_gap_s * fps):
                continue
            cx1, cy1, h1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2, b1[3] - b1[1]
            dist = math.hypot(cx0 - cx1, cy0 - cy1)
            if JOIN_V2:                                                  # ground point stays put
                dist = min(dist, math.hypot(cx0 - cx1, b0[3] - b1[3]))
            if dist <= max(h0, h1) and (best is None or dist < best[0]):
                best = (dist, jid)
        if best is None:
            joined[tid] = list(tracks[tid])
        else:
            joined[best[1]].extend(tracks[tid])
    return joined


def main_person(joined):
    if not joined:
        return None
    def score(items):
        area = np.mean([(b[2] - b[0]) * (b[3] - b[1]) for _, b, _ in items])
        return (len(items), area)
    return joined[max(joined, key=lambda j: score(joined[j]))]


def fill_gaps(arr, present, max_fill):
    """Linear interpolation over gaps of at most max_fill frames; arr is (T, ...) with NaN where absent."""
    T = len(arr)
    idx = np.where(present)[0]
    filled = present.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_fill:
            for g in range(1, gap + 1):
                w = g / (gap + 1)
                arr[a + g] = (1 - w) * arr[a] + w * arr[b]
                filled[a + g] = True
    return arr, filled


def torso_angle(k, kp_conf=0.3):
    if min(k[L_SH, 2], k[R_SH, 2], k[L_HIP, 2], k[R_HIP, 2]) < kp_conf:
        return np.nan
    v = (k[L_SH, :2] + k[R_SH, :2]) / 2 - (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    n = float(np.linalg.norm(v))
    return np.nan if n < 1e-6 else math.degrees(math.acos(max(-1.0, min(1.0, -v[1] / n))))


def features_from_tracks(track_data, max_fill_s=0.5):
    """Main-person feature sequence. Returns dict(t, feats, present, fps) or None if nobody is seen."""
    frames, fps = track_data["frames"], float(track_data["fps"])
    T = len(frames)
    person = main_person(join_tracks(frames, fps))
    if T == 0 or person is None:
        return None

    kp = np.full((T, 17, 3), np.nan, np.float32)
    box = np.full((T, 4), np.nan, np.float32)
    seen = np.zeros(T, bool)
    for i, b, k in person:
        if not seen[i]:                       # one detection per frame for the main person
            kp[i], box[i], seen[i] = np.asarray(k, np.float32), b, True
    kp, filled = fill_gaps(kp, seen, int(round(max_fill_s * fps)))
    box, _ = fill_gaps(box, seen, int(round(max_fill_s * fps)))

    heights = box[filled, 3] - box[filled, 1]
    body_h = float(np.percentile(heights, 90)) if len(heights) else 1.0
    body_h = max(body_h, 1.0)
    floor_y = float(np.nanmax(box[filled, 3])) if filled.any() else 0.0

    hip = (kp[:, L_HIP, :2] + kp[:, R_HIP, :2]) / 2
    hip_ok = (kp[:, L_HIP, 2] >= 0.3) & (kp[:, R_HIP, 2] >= 0.3)
    centre = np.where(hip_ok[:, None], hip, np.stack([(box[:, 0] + box[:, 2]) / 2,
                                                      (box[:, 1] + box[:, 3]) / 2], 1))
    pos = (kp[:, :, :2] - centre[:, None, :]) / body_h                 # T x 17 x 2
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * fps
    conf = kp[:, :, 2]

    top_h = (floor_y - box[:, 1]) / body_h                             # head-top height above floor
    top_speed = np.zeros(T, np.float32)
    top_speed[1:] = (top_h[1:] - top_h[:-1]) * fps
    hip_y = centre[:, 1] / body_h
    hip_speed = np.zeros(T, np.float32)
    hip_speed[1:] = -(hip_y[1:] - hip_y[:-1]) * fps                    # up is positive
    h_ratio = (box[:, 3] - box[:, 1]) / body_h
    aspect = (box[:, 2] - box[:, 0]) / np.maximum(box[:, 3] - box[:, 1], 1.0)
    ang = np.array([torso_angle(k) if f else np.nan for k, f in zip(kp, filled)], np.float32) / 90.0

    feats = np.concatenate([pos.reshape(T, 34), vel.reshape(T, 34), conf,
                            np.stack([top_h, top_speed, h_ratio, aspect, ang, hip_speed,
                                      filled.astype(np.float32)], 1)], 1).astype(np.float32)
    # velocities right after a long gap are meaningless
    gap_start = np.r_[False, filled[1:] & ~filled[:-1]]
    feats[gap_start, 34:68] = 0.0
    feats[gap_start, 86] = 0.0                                          # top_speed
    feats[gap_start, 90] = 0.0                                          # hip_speed
    feats[~filled] = 0.0                                               # absent frames: zeros + present=0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(feats, -20, 20, out=feats)
    t = np.array([f["t"] for f in frames], np.float32)
    return dict(t=t, feats=feats, present=filled, fps=np.float32(fps), body_h=np.float32(body_h))


def main():
    ap = argparse.ArgumentParser(description="Clean skeleton features for every cached video")
    ap.add_argument("--poses", required=True, help="output folder of cache_poses.py (has index.csv)")
    ap.add_argument("--out", default="features")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    with open(os.path.join(a.poses, "index.csv"), newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] in ("cached", "already")]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} cached videos")

    manifest, empty = [], 0
    for n, r in enumerate(rows, 1):
        src = os.path.join(a.poses, r["dataset"], r["path"] + ".pkl")
        d = pickle.load(open(src, "rb"))
        res = features_from_tracks(d)
        status, T, seen = "ok", 0, 0.0
        if res is None:
            status, empty = "nobody detected", empty + 1
        else:
            dst = os.path.join(a.out, r["dataset"], r["path"] + ".npz")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **res)
            T, seen = len(res["t"]), float(res["present"].mean())
        manifest.append(dict(dataset=r["dataset"], path=r["path"], split=r["split"], status=status,
                             frames=T, seen_frac=round(seen, 3)))
        if n % 100 == 0 or n == len(rows):
            print(f"  {n}/{len(rows)}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(a.out, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_NAMES) + "\n")

    print("\nDataset     videos   person seen (mean % of frames)")
    for ds in sorted({m["dataset"] for m in manifest}):
        ms = [m for m in manifest if m["dataset"] == ds and m["status"] == "ok"]
        print(f"  {ds:<10} {len(ms):>5}   {100 * np.mean([m['seen_frac'] for m in ms]) if ms else 0:5.1f}")
    print(f"Videos with nobody detected: {empty}")
    print(f"Saved {a.out}/manifest.csv and one .npz per video ({N_FEATS} features per frame)")


if __name__ == "__main__":
    main()
