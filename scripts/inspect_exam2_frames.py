import os
import sys
sys.path.insert(0, "packages")
import cv2
import numpy as np

video_path = "data/videos/adcfc2a1e1e84047a8639805000a49d0.mp4"
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Exam 2 video: {w}x{h}, {total_frames} frames ({total_frames/fps:.1f}s)")

# Let's inspect frame 0, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600
os.makedirs("data/debug/exam2_audit", exist_ok=True)
from visionradar.examination.detector import ExaminationDetector
detector = ExaminationDetector()

audit_frames = [0, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
results = {}

for f_idx in audit_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Run detector at different thresholds: 0.30 (default), 0.25, 0.20, 0.15
    dets_30, objs_30 = detector.detect(frame, conf_thresh=0.30)
    dets_25, objs_25 = detector.detect(frame, conf_thresh=0.25)
    dets_20, objs_20 = detector.detect(frame, conf_thresh=0.20)
    dets_15, objs_15 = detector.detect(frame, conf_thresh=0.15)
    
    results[f_idx] = {
        "p_30": len(dets_30),
        "p_25": len(dets_25),
        "p_20": len(dets_20),
        "p_15": len(dets_15),
        "o_30": [(o.object_class, round(o.confidence, 3)) for o in objs_30],
        "o_15": [(o.object_class, round(o.confidence, 3)) for o in objs_15]
    }
    
    print(f"Frame {f_idx:4d} ({f_idx/fps:4.1f}s): "
          f"Persons [c>=0.30]: {len(dets_30):2d} | "
          f"[c>=0.25]: {len(dets_25):2d} | "
          f"[c>=0.20]: {len(dets_20):2d} | "
          f"[c>=0.15]: {len(dets_15):2d} | "
          f"Objects: {len(objs_15)}")

cap.release()
