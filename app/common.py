"""
Shared by run_fall.py and live_fall.py: where things are, the settings file, and the pose model.
(The official entry point for the course is infer.py in the main folder.)

Folder layout (only code, settings and results are in git; the rest stays on your computer):
  app/            run_fall.py, live_fall.py, this file
  config.yaml     every setting (model paths, threshold, input size, sampling, windows, alarm)
  models/         gru_hard.pt (from Kaggle), yolo11m-pose.pt (downloaded automatically),
                  yolo11m-pose_openvino_model/ (made automatically on a computer without a GPU)
  test_videos/    your own videos to test
  outputs/        annotated videos and CSV files written by run_fall.py / live_fall.py
  src/            the pipeline (pose, tracking, features, GRU, evaluation)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")
sys.path.insert(0, os.path.join(ROOT, "src"))


def load_config(path=None):
    """The one settings file, config.yaml (also used by infer.py)."""
    from fv_engine import load_config as _load
    return _load(path or os.path.join(ROOT, "config.yaml"))


def check_model(cfg):
    if not os.path.isfile(cfg["gru"]):
        sys.exit(f"Model not found: {cfg['gru']}\nDownload {os.path.basename(cfg['gru'])} from the fallvision-models "
                 f"Kaggle output (Output -> models) and put it in the models folder.")


def pose_model(cfg):
    """Path of the pose model to use. The weights are downloaded into models/ the first time.
    With an NVIDIA GPU: the PyTorch weights. Without one: the same weights converted once to
    OpenVINO (Intel's CPU runtime, about 4x faster on a CPU, practically the same keypoints)."""
    import torch
    from ultralytics import YOLO
    pt = cfg["pose_model"]
    folder, name = os.path.dirname(pt), os.path.basename(pt)
    os.makedirs(folder, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(folder)                                   # downloads and conversions land in models/
        if not os.path.isfile(name):
            print(f"Downloading {name} into {folder} (first time only)...")
            YOLO(name)
        if torch.cuda.is_available():
            return pt
        ov = os.path.join(folder, os.path.splitext(name)[0] + "_openvino_model")
        if os.path.isdir(ov):
            return ov
        try:
            import openvino  # noqa: F401
        except ImportError:
            print("Tip: 'pip install openvino' makes this about 4x faster without a GPU. Using the normal model.")
            return pt
        print("No GPU: converting the pose model to OpenVINO for fast CPU use (first time only, ~1 min)...")
        try:
            YOLO(name).export(format="openvino", dynamic=True, half=False, verbose=False)
        except Exception as e:
            print(f"OpenVINO conversion failed ({type(e).__name__}); using the normal model.")
        return ov if os.path.isdir(ov) else pt
    finally:
        os.chdir(cwd)
