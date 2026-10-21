import sys
sys.path.insert(0, "packages")
import cv2
import numpy as np
from visionradar.examination.detector import ExaminationDetector
from visionradar.examination.tracker import PersonTracker

video_path = "data/videos/adcfc2a1e1e84047a8639805000a49d0.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

print(f"Tracking Exam 2 across {total_frames} frames...")

detector_default = ExaminationDetector(confidence_threshold=0.30)
tracker_default = PersonTracker(max_lost_frames=30, iou_threshold=0.30)

detector_calib = ExaminationDetector(class_thresholds={0: 0.20, 67: 0.15, 73: 0.18})
tracker_calib = PersonTracker(max_lost_frames=30, iou_threshold=0.25)

# Run on first 300 frames to compare
track_counts_default = []
track_counts_calib = []

for f in range(300):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Default pipeline
    p_def, _ = detector_default.detect(frame, conf_thresh=0.30)
    act_def = tracker_default.update(p_def, f)
    track_counts_default.append(len(act_def))
    
    # Calibrated pipeline
    p_cal, _ = detector_calib.detect(frame, conf_thresh=0.20)
    act_cal = tracker_calib.update(p_cal, f)
    track_counts_calib.append(len(act_cal))

cap.release()

print("\n--- RESULTS OVER FIRST 300 FRAMES (10s) ---")
print(f"Default (conf=0.30, iou=0.30):")
print(f"   Avg active persons/frame: {np.mean(track_counts_default):.2f}")
print(f"   Total unique track IDs created: {len(tracker_default.tracks)}")
print(f"   Max active in a single frame: {max(track_counts_default)}")

print(f"\nCalibrated (conf=0.20, iou=0.25):")
print(f"   Avg active persons/frame: {np.mean(track_counts_calib):.2f}")
print(f"   Total unique track IDs created: {len(tracker_calib.tracks)}")
print(f"   Max active in a single frame: {max(track_counts_calib)}")
