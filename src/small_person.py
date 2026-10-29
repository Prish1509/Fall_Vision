"""
Small-person handling (Member 1, Step 15).

Far-away people are only a few dozen pixels tall, so the pose model sees too few pixels to
place 17 joints well. For every tracked person whose box is shorter than `small_px`, we crop
the area around them (with a margin), enlarge the crop, run the pose model again on the
enlarged crop and map the joints back to full-frame coordinates. The new joints are kept only
if they are more confident than the originals, so this can never make a skeleton worse.

Ideas used (cite in the report):
  - Crop-and-zoom / sliced inference for small objects: SAHI (Akyon et al., ICIP 2022)
  - Small, distant people in fall detection: Liu et al., Scientific Reports 2025
  - Why far people are small: perspective projection, DS5602 Lecture 2
"""

import cv2
import numpy as np


def refine_small_people(model, frame, boxes, kpts, small_px=160, target_h=320, margin=0.3,
                        imgsz=640, conf=0.25):
    """Return (new_kpts_list, number_of_people_refined)."""
    H, W = frame.shape[:2]
    out, n_refined = [], 0
    for b, k in zip(boxes, kpts):
        x1, y1, x2, y2 = b
        bh, bw = y2 - y1, x2 - x1
        if bh >= small_px or bh < 8:
            out.append(k)
            continue
        # square crop around the person (a falling person's box gets wide, so use the larger side)
        side = max(bh, bw) * (1 + 2 * margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cx1, cy1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        cx2, cy2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            out.append(k)
            continue
        s = min(target_h / max(crop.shape[0], 1), 4.0)          # enlarge, at most 4x
        up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        r = model.predict(up, imgsz=imgsz, conf=conf, classes=[0], verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            out.append(k)
            continue
        # the detection closest to the crop centre is our person
        bx = r.boxes.xyxy.cpu().numpy()
        ccx, ccy = up.shape[1] / 2, up.shape[0] / 2
        j = int(np.argmin(((bx[:, 0] + bx[:, 2]) / 2 - ccx) ** 2 + ((bx[:, 1] + bx[:, 3]) / 2 - ccy) ** 2))
        kxy = r.keypoints.xy.cpu().numpy()[j]
        kc = r.keypoints.conf
        kc = kc.cpu().numpy()[j] if kc is not None else np.ones(len(kxy))
        new = np.concatenate([kxy / s + np.array([cx1, cy1]), kc[:, None]], axis=1)
        old = np.asarray(k)
        if new[:, 2].mean() > old[:, 2].mean():
            out.append(new.tolist())
            n_refined += 1
        else:
            out.append(k)
    return out, n_refined
