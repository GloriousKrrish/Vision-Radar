import sys
sys.path.insert(0, "packages")
import cv2
import numpy as np
from visionradar.cv.detection.yolox import YOLOXDetector
from visionradar.cv.tracker import ByteTrackTracker, compute_iou, _bipartite_match

vpath = "data/videos/4da052095e20419fad598ef93c00548b.mp4"
cap = cv2.VideoCapture(vpath)

detector = YOLOXDetector(input_size=(640, 640), confidence_threshold=0.25)
tracker = ByteTrackTracker(max_age=30, iou_threshold=0.3, high_threshold=0.40, low_threshold=0.10)

print("--- Frame-by-frame Tracking Diagnostics for Traffic 2.mp4 (First 15 frames) ---")

for f_idx in range(15):
    ret, frame = cap.read()
    if not ret:
        break

    dets = detector.detect(frame)
    # Let's inspect before update:
    active_before = list(tracker.active_tracks.values())
    pred_boxes = [t.predict() for t in active_before]

    # Partition detections
    dets_h = [d for d in dets if d.confidence >= tracker.high_threshold]
    dets_l = [d for d in dets if tracker.low_threshold <= d.confidence < tracker.high_threshold]

    # Stage 1 match
    if pred_boxes and dets_h:
        det_boxes = [d.bbox for d in dets_h]
        matches, unassigned_t, unassigned_d = _bipartite_match(pred_boxes, det_boxes, tracker.iou_threshold)
        
        # Calculate maximum possible IoU for unassigned detections
        unassigned_max_ious = []
        for d_i in unassigned_d:
            db = det_boxes[d_i]
            max_iou = max([compute_iou(pb, db) for pb in pred_boxes]) if pred_boxes else 0.0
            unassigned_max_ious.append(max_iou)
    else:
        matches = []
        unassigned_t = list(range(len(pred_boxes)))
        unassigned_d = list(range(len(dets_h)))
        unassigned_max_ious = []

    # Update tracker
    tracks = tracker.update(dets, frame_index=f_idx, timestamp=f_idx / 30.0)

    print(f"Frame {f_idx:2d}: {len(dets):2d} dets ({len(dets_h):2d} high, {len(dets_l):2d} low) | "
          f"Active before: {len(active_before):2d} | Matches: {len(matches):2d} | "
          f"Unassigned Dets: {len(unassigned_d):2d} (Max IoUs: {[round(x, 2) for x in unassigned_max_ious[:5]]}) | "
          f"Total active now: {len(tracker.active_tracks):2d} | Next track ID: {tracker.next_id}")

cap.release()
