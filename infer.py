"""
FallVision inference: person collapse detection on video files (course submission entry point).

    python infer.py --source INPUT.mp4 --config config.yaml --output-dir results --device cuda:0 --no-display

--source      a video file, or a folder of videos (each video is processed separately)
--config      config.yaml (all settings; nothing is set in the code)
--output-dir  where predictions.jsonl, alarms.jsonl and run_summary.json are written
--device      cpu | cuda | cuda:N | N   (N = GPU number; cuda without a GPU falls back to cpu with a warning)
--no-display  no window (the default when no screen is available); nothing is ever asked interactively
--save-video  also write <source_id>_annotated.mp4 to the output directory

Videos are read frame by frame in order. A frame is processed when it falls on the next slot of the
model_fps time grid; other frames are counted as skipped. Each processed frame uses only itself and
earlier frames, alarms are written the moment they are raised, and all temporal state is reset for
every new video. decision_timestamp_s is the time of the latest frame used for the alarm; the
estimated start of the collapse is written separately as onset_estimate_s.

Works offline: both models are loaded from the paths in config.yaml; nothing is downloaded.
"""

import argparse
import glob
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("YOLO_OFFLINE", "1")             # never try to reach the internet

import cv2                                               # noqa: E402
import torch                                             # noqa: E402

from fv_engine import CollapseEngine, PoseTracker, draw, load_config, new_gru, torch_device  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".webm", ".3gp")


def list_sources(src):
    if os.path.isdir(src):
        return sorted(p for p in glob.glob(os.path.join(src, "**", "*"), recursive=True)
                      if p.lower().endswith(VIDEO_EXT))
    return [src]


def process_video(path, cfg, pose, gru, device, outs, display, save_dir):
    source_id = os.path.splitext(os.path.basename(path))[0]
    stats = dict(source_id=source_id, path=os.path.abspath(path), status="completed", frames_read=0,
                 frames_processed=0, frames_skipped=0, alarms=0, runtime_s=0.0, errors=[])
    t_start = time.time()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        stats.update(status="failed", errors=[f"cannot open video: {path}"])
        return stats
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps != src_fps or src_fps > 1000:
        src_fps = None                                       # fall back to the frame time stamps
    pose.reset()                                             # new video: new tracker ...
    engine = CollapseEngine(cfg, gru, device)                # ... and new temporal state
    writer, last_slot, frame_index = None, -1, -1
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frame_index += 1
            stats["frames_read"] += 1
            t = frame_index / src_fps if src_fps else cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            slot = int(round(t * cfg["model_fps"]))
            if slot <= last_slot:                            # not on the next 15-per-second slot
                stats["frames_skipped"] += 1
                continue
            last_slot = slot
            stats["frames_processed"] += 1

            det = pose.track(img, engine.pose_input_size(slot))
            pids = engine.add_frame(slot, det, (img.shape[1], img.shape[0]))
            for a in engine.step(slot, t):
                rec = dict(source_id=source_id, event_id=a["event_id"], label=cfg["label"],
                           decision_timestamp_s=a["decision_timestamp_s"], confidence=a["confidence"],
                           frame_index=frame_index, onset_estimate_s=a["onset_estimate_s"],
                           person_id=a["person_id"], bbox_xyxy=a["bbox"])
                outs["alarms"].write(json.dumps(rec) + "\n")
                outs["alarms"].flush()                       # alarms are emitted as they occur
                stats["alarms"] += 1
                print(f"  ALARM {source_id}: person #{a['person_id']} collapse at {a['decision_timestamp_s']:.2f} s "
                      f"(onset ~{a['onset_estimate_s']:.2f} s), confidence {a['confidence']:.3f}", flush=True)
            outs["predictions"].write(json.dumps(dict(source_id=source_id, frame_index=frame_index,
                                                      timestamp_s=round(t, 3),
                                                      predictions=engine.predictions(slot, det, pids))) + "\n")

            if display or save_dir:
                shown = draw(img.copy(), det, pids, engine)
                if save_dir:
                    if writer is None:
                        writer = cv2.VideoWriter(os.path.join(save_dir, f"{source_id}_annotated.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), cfg["model_fps"],
                                                 (shown.shape[1], shown.shape[0]))
                    writer.write(shown)
                if display:
                    try:
                        cv2.imshow("FallVision", shown)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            stats["errors"].append("stopped by the user (q)")
                            break
                    except cv2.error:
                        display = False                      # no screen: carry on without a window
    except Exception as e:                                   # one bad video must not stop the others
        stats["status"] = "failed"
        stats["errors"].append(f"{type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        outs["predictions"].flush()
    stats["runtime_s"] = round(time.time() - t_start, 2)
    return stats


def run(sources, cfg, pose, gru, dev, output_dir, display=False, save_video=False, summary=None):
    """Process the videos one after another and write the three output files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    summary = summary or new_summary(cfg.get("config_path"), dev)
    t0 = time.time()
    outs = dict(predictions=open(os.path.join(output_dir, "predictions.jsonl"), "w"),
                alarms=open(os.path.join(output_dir, "alarms.jsonl"), "w"))
    try:
        for path in sources:
            print(f"Processing {path}", flush=True)
            st = process_video(path, cfg, pose, gru, dev, outs, display, output_dir if save_video else None)
            summary["sources"].append(st)
            for k in ("frames_read", "frames_processed", "frames_skipped", "alarms"):
                summary[k] += st[k]
            summary["errors"] += [f"{st['source_id']}: {e}" for e in st["errors"]]
            print(f"  {st['source_id']}: {st['status']}, {st['frames_processed']} frames processed, "
                  f"{st['alarms']} alarm(s), {st['runtime_s']} s", flush=True)
        failed = [s_ for s_ in summary["sources"] if s_["status"] != "completed"]
        if failed:
            summary["status"] = "failed" if len(failed) == len(summary["sources"]) else "completed_with_errors"
    finally:
        for f in outs.values():
            f.close()
        summary["total_runtime_s"] = round(summary["total_runtime_s"] + time.time() - t0, 2)
        with open(os.path.join(output_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def new_summary(config_path, device_requested):
    return dict(status="completed", project="person_collapse", config=config_path, device_requested=device_requested,
                device_used=None, sources=[], frames_read=0, frames_processed=0, frames_skipped=0, alarms=0,
                total_runtime_s=0.0, errors=[], warnings=[])


def main():
    ap = argparse.ArgumentParser(description="FallVision person collapse detection")
    ap.add_argument("--source", required=True, help="video file or folder of videos")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                    help="cpu, cuda, cuda:N or N (default: cuda:0 if available, else cpu)")
    ap.add_argument("--no-display", action="store_true", help="do not open a window")
    ap.add_argument("--save-video", action="store_true", help="also save annotated videos in the output dir")
    a = ap.parse_args()

    t0 = time.time()
    summary = new_summary(os.path.abspath(a.config), a.device)
    try:
        cfg = load_config(a.config)
        dev, yolo_dev, warn = torch_device(a.device)
        if warn:
            summary["warnings"].append(warn)
            print("Warning:", warn)
        summary["device_used"] = dev
        sources = list_sources(a.source)
        if not sources or not all(os.path.isfile(s) for s in sources):
            raise FileNotFoundError(f"no video found at {a.source}")
        print(f"Loading models on {dev} ...", flush=True)
        gru = new_gru(cfg, dev)
        pose = PoseTracker(cfg, yolo_dev)
        summary["total_runtime_s"] = round(time.time() - t0, 2)          # model loading counts too
        summary = run(sources, cfg, pose, gru, dev, a.output_dir, not a.no_display, a.save_video, summary)
    except Exception as e:
        summary["status"] = "failed"
        summary["errors"].append(f"{type(e).__name__}: {e}")
        summary["total_runtime_s"] = round(time.time() - t0, 2)
        traceback.print_exc()
        os.makedirs(a.output_dir, exist_ok=True)
        for name in ("predictions.jsonl", "alarms.jsonl"):
            p = os.path.join(a.output_dir, name)
            if not os.path.exists(p):
                open(p, "w").close()
        with open(os.path.join(a.output_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    finally:
        if not a.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    print(f"Done: {summary['status']}. {summary['alarms']} alarm(s). Outputs in {os.path.abspath(a.output_dir)}")
    sys.exit(0 if summary["status"] != "failed" else 1)


if __name__ == "__main__":
    main()
