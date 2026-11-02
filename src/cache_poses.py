"""
Pose caches for every video in the split file (Member 2, Step 14).

Runs YOLO11-pose + ByteTrack once per video (the same extract_tracks() as the rule baseline:
about 15 fps, boxes, track ids, 17 keypoints with confidences) and saves one pickle per video,
so later steps (preprocessing, windows, Random Forest, GRU) never decode video again.

Video locations:
  OmniFall datasets   <omnifall-root>/<dataset>/video/<path>.mp4
  GMDCSA24            <gmd-root>/Subject N/<Fall|ADL>/<nn>.mp4   (labels say "Subject_N")

Output:
  <out>/<dataset>/<path>.pkl    dict(video, fps, frames, dataset, path, split, model, imgsz)
  <out>/index.csv               one row per video: dataset, path, split, status, frames, fps, seconds

Already-cached videos are skipped, so an interrupted run can be repeated. The run stops by itself
after --max-hours so Kaggle's time limit never cuts it off mid-file.

Usage:
  python src/cache_poses.py --omnifall-root /kaggle/input/.../omnifall --gmd-root /kaggle/input/.../GMDCSA24 \
                            --out /kaggle/working/poses --imgsz 640
"""

import argparse
import csv
import os
import pickle
import time

from baseline_fall import extract_tracks


def video_file(ds, path, omnifall_root, gmd_root):
    if ds.lower() == "gmdcsa24":
        subj, rest = path.split("/", 1)                       # "Subject_1", "ADL/01"
        return os.path.join(gmd_root, subj.replace("_", " "), rest + ".mp4")
    return os.path.join(omnifall_root, ds, "video", path + ".mp4")


def main():
    ap = argparse.ArgumentParser(description="Cache YOLO11-pose tracks for every split video")
    ap.add_argument("--splits", default="splits/videos.csv")
    ap.add_argument("--omnifall-root", required=True, help="folder with mcfd/ caucafall/ le2i/ OOPS/")
    ap.add_argument("--gmd-root", required=True, help="GMDCSA24 folder with 'Subject 1' ...")
    ap.add_argument("--out", default="poses")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--datasets", default="", help="comma list to limit, e.g. le2i,caucafall (default all)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (testing)")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop cleanly after this long")
    a = ap.parse_args()

    with open(a.splits, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] != "own"]
    if a.datasets:
        keep = {d.strip().lower() for d in a.datasets.split(",")}
        rows = [r for r in rows if r["dataset"].lower() in keep]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(rows)} videos to cache at imgsz {a.imgsz} with {a.model}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    index, t0, done_new = [], time.time(), 0
    for n, r in enumerate(rows, 1):
        ds, path, split = r["dataset"], r["path"], r["split"]
        pkl = os.path.join(a.out, ds, path + ".pkl")
        src = video_file(ds, path, a.omnifall_root, a.gmd_root)
        status, frames, fps = "cached", 0, 0.0
        if os.path.exists(pkl):
            status = "already"
            d = pickle.load(open(pkl, "rb"))
            frames, fps = len(d["frames"]), d["fps"]
        elif not os.path.exists(src):
            status = "missing video"
        elif (time.time() - t0) / 3600 > a.max_hours:
            status = "skipped (time limit)"
        else:
            try:
                d, _ = extract_tracks(src, a.model, a.fps, a.imgsz)
                d.update(dataset=ds, path=path, split=split, model=a.model, imgsz=a.imgsz)
                os.makedirs(os.path.dirname(pkl), exist_ok=True)
                with open(pkl + ".tmp", "wb") as f:
                    pickle.dump(d, f)
                os.replace(pkl + ".tmp", pkl)                 # never leave a half-written file
                frames, fps = len(d["frames"]), d["fps"]
                done_new += 1
            except Exception as e:                            # a broken video must not stop the run
                status = f"error: {type(e).__name__}: {str(e)[:80]}"
        index.append(dict(dataset=ds, path=path, split=split, status=status, frames=frames,
                          fps=round(fps, 2), seconds=round(frames / fps, 1) if fps else 0))
        if n % 25 == 0 or n == len(rows):
            el = time.time() - t0
            eta = el / max(done_new, 1) * sum(1 for x in rows[n:]) if done_new else 0
            print(f"  {n}/{len(rows)}  new {done_new}  elapsed {el / 60:.0f} min  "
                  f"remaining ~{eta / 60:.0f} min", flush=True)

    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)

    by = {}
    for x in index:
        ok = x["status"] in ("cached", "already")
        s = by.setdefault(x["dataset"], [0, 0, 0.0])
        s[0] += ok
        s[1] += 1
        s[2] += x["seconds"]
    print("\nDataset     cached / total   video minutes")
    for ds, (ok, tot, sec) in sorted(by.items()):
        print(f"  {ds:<10} {ok:>5} / {tot:<5}   {sec / 60:8.1f}")
    bad = [x for x in index if x["status"] not in ("cached", "already")]
    print(f"\nProblems: {len(bad)}")
    for x in bad[:15]:
        print(f"  {x['dataset']}/{x['path']}: {x['status']}")
    print(f"Saved {os.path.join(a.out, 'index.csv')}")


if __name__ == "__main__":
    main()
