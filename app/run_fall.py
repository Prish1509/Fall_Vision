"""
FallVision on video files, the easy way: the same engine as infer.py (frame by frame, current and
past frames only), with the models loaded once and the result printed in plain words.

    python app/run_fall.py                         # asks for videos one after another (models load once)
    python app/run_fall.py my_clip.mp4             # a video in test_videos/ (or any full path)
    python app/run_fall.py a.mp4 b.mp4             # several videos
    python app/run_fall.py test_videos             # every video in a folder

For each video, outputs/<name>/ gets predictions.jsonl, alarms.jsonl, run_summary.json and
<name>_annotated.mp4 (red COLLAPSE box on the person). All settings come from config.yaml.
"""

import glob
import json
import os
import sys

from common import OUT_DIR, ROOT, check_model, load_config, pose_model

sys.path.insert(0, ROOT)
import infer                                                    # noqa: E402
from fv_engine import PoseTracker, new_gru, torch_device        # noqa: E402

VIDEO_EXT = infer.VIDEO_EXT


def find_videos(arg):
    arg = arg.strip().strip('"').strip("'")
    for p in (arg, os.path.join(ROOT, "test_videos", arg)):     # a bare name means test_videos/<name>
        if os.path.isdir(p):
            return sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(VIDEO_EXT))
        if os.path.isfile(p):
            return [p]
    print(f"  Not found: {arg}")
    return []


def run_one(video, cfg, pose, gru, dev):
    name = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(OUT_DIR, name)
    s = infer.run([video], cfg, pose, gru, dev, out_dir, display=False, save_video=True)
    alarms = [json.loads(l) for l in open(os.path.join(out_dir, "alarms.jsonl")) if l.strip()]
    print("=" * 60)
    if alarms:
        print(f"RESULT: PERSON COLLAPSE ({len(alarms)} alarm{'s' if len(alarms) > 1 else ''})")
        for a in alarms:
            print(f"  WHO : person #{a['person_id']}")
            print(f"  WHEN: collapse started ~{a['onset_estimate_s']:.1f} s, alarm at {a['decision_timestamp_s']:.1f} s")
            print(f"  WHERE: box {a['bbox_xyxy']}  (confidence {a['confidence']:.2f})")
    else:
        print("RESULT: NO COLLAPSE")
    print(f"Results in      : {out_dir}")
    print(f"Processing time : {s['total_runtime_s']:.0f} s")
    print("=" * 60)


def main():
    cfg = load_config()
    check_model(cfg)
    dev, ydev, warn = torch_device("cuda:0")
    if warn:
        dev, ydev = "cpu", "cpu"
    print(f"Loading the models once (device {dev})...")
    cfg_run = dict(cfg, pose_model=pose_model(cfg))       # same weights; OpenVINO copy on a CPU (faster)
    gru, pose = new_gru(cfg, dev), PoseTracker(cfg_run, ydev)
    print(f"Ready. Threshold {cfg['threshold']}")
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            for v in find_videos(arg):
                run_one(v, cfg, pose, gru, dev)
        return
    while True:
        try:
            arg = input("\nVideo (name in test_videos/, full path or folder; Enter to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if not arg.strip():
            break
        for v in find_videos(arg):
            run_one(v, cfg, pose, gru, dev)


if __name__ == "__main__":
    main()
