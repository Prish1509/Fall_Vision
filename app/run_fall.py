"""
FallVision on video files: prints FALL DETECTED or NO FALL (who, when, where) for each video and
saves it with boxes, skeletons and a red FALL label on the person who fell.

    python app/run_fall.py                         # asks for videos one after another (models load once)
    python app/run_fall.py my_clip.mp4             # a video in test_videos/ (or any full path)
    python app/run_fall.py a.mp4 b.mp4             # several videos
    python app/run_fall.py test_videos             # every video in a folder

The models are loaded once and kept in memory, so every video after the first starts at once.
Results go to outputs/<name>_falls.mp4 and outputs/<name>_falls.csv.
All settings come from configs/fallvision.json; nothing is set in this file.
"""

import csv
import glob
import os
import sys
import time

from common import OUT_DIR, ROOT, check_model, load_config, pose_model

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm")


def find_videos(arg):
    arg = arg.strip().strip('"').strip("'")
    for p in (arg, os.path.join(ROOT, "test_videos", arg)):     # a bare name means test_videos/<name>
        if os.path.isdir(p):
            return sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(VIDEO_EXT))
        if os.path.isfile(p):
            return [p]
    print(f"  Not found: {arg}")
    return []


def run_one(video, cfg, model):
    import detect_falls
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(video))[0])
    out_video, out_csv = stem + "_falls.mp4", stem + "_falls.csv"
    sys.argv = ["detect_falls.py", "--video", video, "--gru", cfg["gru"], "--threshold", str(cfg["threshold"]),
                "--score", cfg["score"], "--conf", str(cfg["det_conf"]), "--small-px", str(cfg["small_px"]),
                "--model", model, "--out", out_video, "--csv", out_csv] + (["--join-v2"] if cfg["join_v2"] else [])
    print(f"\n>>> {os.path.basename(video)}")
    t0 = time.time()
    detect_falls.main()
    took = time.time() - t0

    alarms = list(csv.DictReader(open(out_csv, newline="")))
    print("=" * 60)
    if alarms:
        print(f"RESULT: FALL DETECTED ({len(alarms)} person{'s' if len(alarms) > 1 else ''})")
        for r in alarms:
            print(f"  WHO : person #{r['person_id']}")
            print(f"  WHEN: fall started ~{float(r['fall_start_s']):.1f} s, alarm at {float(r['alarm_time_s']):.1f} s")
            print(f"  WHERE: box x1={r['x1']} y1={r['y1']} x2={r['x2']} y2={r['y2']}  (score {r['score']})")
    else:
        print("RESULT: NO FALL")
    print(f"Annotated video saved: {out_video}")
    print(f"Alarm list saved     : {out_csv}")
    print(f"Processing time      : {took:.0f} s")
    print("=" * 60)


def main():
    cfg = load_config()
    check_model(cfg)
    print("Loading the models (once)...")
    model = pose_model(cfg)
    import numpy as np
    import baseline_fall
    import detect_falls  # noqa: F401  (loads torch and the pipeline now, not at the first video)
    baseline_fall._pose_model(model).track(np.zeros((640, 640, 3), np.uint8), persist=True, tracker="bytetrack.yaml",
                                           classes=[0], imgsz=640, verbose=False)    # same mode as the real runs
    print(f"Ready. Pose model: {os.path.basename(model.rstrip(os.sep))}, threshold {cfg['threshold']}")

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            for v in find_videos(arg):
                run_one(v, cfg, model)
        return
    while True:                                                 # ask for videos until Enter on an empty line
        try:
            arg = input("\nVideo (name in test_videos/, full path or folder; Enter to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if not arg.strip():
            break
        for v in find_videos(arg):
            run_one(v, cfg, model)


if __name__ == "__main__":
    main()
