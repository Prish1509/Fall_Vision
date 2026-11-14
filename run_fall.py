"""
FallVision on one video: put the path below (or give it in the terminal) and run.

    python run_fall.py                      # uses VIDEO below
    python run_fall.py "D:/videos/test.mp4" # or give the path here

Prints FALL DETECTED or NO FALL (who, when, where) and saves the video with boxes, skeletons and
a red FALL label on the person who fell, next to the input as <name>_falls.mp4.
Needs models/gru.pt (from the fallvision-models Kaggle output) and, the first time, internet for
the YOLO11m-pose weights (downloaded automatically). Without an NVIDIA GPU the pose model is
converted once to OpenVINO (pip install openvino), which runs about 4x faster on a CPU.
"""

import csv
import os
import sys
import time

VIDEO = r"path/to/your/video.mp4"          # <-- put the video path here
GRU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gru.pt")
THRESHOLD = 0.849                          # chosen on validation; do not change
ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(ROOT, "src"))


def pose_model():
    """Same YOLO11m-pose weights either way. With an NVIDIA GPU: the normal PyTorch model.
    Without one: the model converted once to OpenVINO (Intel's CPU runtime), about 4x faster on a
    CPU with practically the same keypoints (checked: same alarm, same fall time)."""
    import torch
    pt = os.path.join(ROOT, "yolo11m-pose.pt")
    if not os.path.isfile(pt):
        pt = "yolo11m-pose.pt"                           # downloaded automatically on first use
    if torch.cuda.is_available():
        return pt
    ov = os.path.join(ROOT, "yolo11m-pose_openvino_model")
    if os.path.isdir(ov):
        return ov
    try:
        import openvino  # noqa: F401
    except ImportError:
        print("Tip: 'pip install openvino' makes this about 4x faster without a GPU. Using the normal model.")
        return pt
    from ultralytics import YOLO
    print("No GPU: converting YOLO11m-pose to OpenVINO for fast CPU use (only the first time, ~1 min)...")
    cwd = os.getcwd()
    os.chdir(ROOT)                                       # weights and converted model stay in the repo folder
    try:
        YOLO("yolo11m-pose.pt").export(format="openvino", dynamic=True, half=False, verbose=False)
    except Exception as e:                               # e.g. a CPU OpenVINO does not support
        print(f"OpenVINO conversion failed ({type(e).__name__}); using the normal model.")
    finally:
        os.chdir(cwd)
    return ov if os.path.isdir(ov) else pt


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    if not os.path.isfile(video):
        sys.exit(f"Video not found: {video}\nPut the path in VIDEO at the top of run_fall.py, "
                 f"or run:  python run_fall.py \"path/to/video.mp4\"")
    if not os.path.isfile(GRU):
        sys.exit(f"Model not found: {GRU}\nDownload gru.pt from the fallvision-models Kaggle output "
                 f"(Output -> models -> gru.pt) and put it in the models folder.")

    import detect_falls
    stem = os.path.splitext(video)[0]
    out_video, out_csv = stem + "_falls.mp4", stem + "_falls.csv"
    model = pose_model()
    print(f"Pose model: {os.path.basename(model.rstrip(os.sep))}")
    sys.argv = ["detect_falls.py", "--video", video, "--gru", GRU, "--threshold", str(THRESHOLD),
                "--model", model, "--out", out_video, "--csv", out_csv]
    t0 = time.time()
    detect_falls.main()
    took = time.time() - t0

    alarms = list(csv.DictReader(open(out_csv, newline="")))
    print("\n" + "=" * 60)
    if alarms:
        print(f"RESULT: FALL DETECTED ({len(alarms)} person{'s' if len(alarms) > 1 else ''})")
        for r in alarms:
            print(f"  WHO : person #{r['person_id']}")
            print(f"  WHEN: fall started ~{float(r['fall_start_s']):.1f} s, alarm at {float(r['alarm_time_s']):.1f} s")
            print(f"  WHERE: box x1={r['x1']} y1={r['y1']} x2={r['x2']} y2={r['y2']}  (score {r['score']})")
    else:
        print("RESULT: NO FALL")
    print(f"\nAnnotated video saved: {out_video}")
    print(f"Alarm list saved     : {out_csv}")
    print(f"Processing time      : {took:.0f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()
