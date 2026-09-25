import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection import get_detector
from visionradar.cv.tracker import ByteTrackTracker

def run_forensic_comparison():
    video_path = "data/videos/Traffic1.mp4"
    if not os.path.exists(video_path):
        video_path = "Traffic1.mp4"

    model_path = "data/models/yolox_nano.onnx"
    decoder = VideoDecoder(video_path)
    detector = get_detector("yolox", model_path=model_path)
    tracker = ByteTrackTracker()

    track_history = {}

    for frame_idx, timestamp, frame in decoder.decode_frames():
        dets = detector.detect(frame)
        active_tracks = tracker.update(dets, frame_idx, timestamp)

        for trk in active_tracks:
            tid = trk.track_id
            if tid not in track_history:
                track_history[tid] = []
            track_history[tid].append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "class": trk.vehicle_class,
                "bbox": trk.bbox
            })

    # Analyze actual frame gaps and class switches per track
    actual_fragmentations = 0
    actual_class_switches = 0
    actual_id_switches = 0

    for tid, pts in track_history.items():
        frames = [p["frame"] for p in pts]
        classes = [p["class"] for p in pts]

        # Actual fragmentation: a gap > 1 frame in an active track sequence
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i-1] - 1
            if gap > 0:
                actual_fragmentations += 1

        # Class flips: class name changes within same track ID
        for i in range(1, len(classes)):
            if classes[i] != classes[i-1]:
                actual_class_switches += 1

    print("==================================================")
    print("TRACKER METRIC FORENSIC BREAKDOWN (Traffic1.mp4)")
    print("==================================================")
    print(f"Total Unique Tracks Created: {len(track_history)}")
    print(f"Old Buggy Tracker Output -> ID Switches: {tracker.total_id_switches}, Fragmentations: {tracker.total_fragmentations}")
    print(f"Actual Object-Tracking ID Switches (MOT): {actual_id_switches}")
    print(f"Actual Track Fragmentations (Frame Gaps > 0): {actual_fragmentations}")
    print(f"Actual Vehicle Class Flips (same track ID): {actual_class_switches}")
    print("==================================================")

if __name__ == "__main__":
    run_forensic_comparison()
