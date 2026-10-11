"""
Forensic Discrepancy Debugger — Traces 58-61 km/h vs 252-292 km/h in Traffic1.mp4
"""
import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "packages"))

from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.perception.pipeline import PerceptionPipeline
from visionradar.perception.preprocess import letterbox_preprocess
from visionradar.perception.postprocess import postprocess_yolox_output

video_path = str(BASE_DIR / "data" / "videos" / "Traffic1.mp4")
cap = cv2.VideoCapture(video_path)

pipeline = PerceptionPipeline(
    video_path=video_path,
    model_path=str(BASE_DIR / "data" / "models" / "yolox_nano.onnx"),
    confidence_threshold=0.25,
    nms_threshold=0.45
)

img_pts = [(200.0, 150.0), (600.0, 150.0), (750.0, 400.0), (50.0, 400.0)]
w_pts = [(0.0, 60.0), (12.0, 60.0), (12.0, 0.0), (0.0, 0.0)]
calibrator = HomographyCalibrator(image_points=img_pts, world_points=w_pts)
estimator = MonocularSpeedEstimator(calibrator)

frame_idx = 0
all_trajectories = {}

while cap.isOpened() and frame_idx < 120:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h_src, w_src = frame.shape[:2]

    blob, scale, pad = letterbox_preprocess(frame, target_size=(640, 640))
    raw_output, infer_ms = pipeline.inference_engine.infer(blob)
    dets = postprocess_yolox_output(
        raw_outputs=raw_output,
        frame_id=frame_idx,
        source_shape=(w_src, h_src),
        input_size=(640, 640),
        confidence_threshold=0.25,
        nms_threshold=0.45,
        scale=scale,
        pad=pad
    )
    raw_trks = pipeline.tracker.update(dets, frame_idx, time.perf_counter())
    conf_trks, _ = pipeline.lifecycle_manager.update(raw_trks, frame_idx)

    for trk in conf_trks:
        tid = trk.track_id
        if tid not in all_trajectories:
            all_trajectories[tid] = VehicleTrajectory(track_id=tid, vehicle_class="car")
        cx = (trk.bbox.x1 + trk.bbox.x2) / 2.0
        cy = trk.bbox.y2
        all_trajectories[tid].add_point(TrajectoryPoint(
            frame_index=frame_idx,
            timestamp=frame_idx * 0.033,
            anchor_pixel=(cx, cy)
        ))

cap.release()

print(f"Total tracks discovered: {len(all_trajectories)}")
for tid, traj in all_trajectories.items():
    print(f"\n--- Track #{tid}: {len(traj.points)} points ---")
    estimator.project_trajectory(traj)
    valid_pts = [p for p in traj.points if getattr(p, "roi_status", "VALID") == "VALID"]
    print(f"  Valid ROI points: {len(valid_pts)}")
    for i, p in enumerate(valid_pts[:10]):
        w_str = f"({p.world_pos[0]:.2f}, {p.world_pos[1]:.2f})" if p.world_pos else "None"
        print(f"    Pt #{i}: frame={p.frame_index}, t={p.timestamp:.3f}s, pixel=({p.anchor_pixel[0]:.1f}, {p.anchor_pixel[1]:.1f}), world={w_str}, roi={p.roi_status}")
    
    if len(valid_pts) >= 2:
        t0 = valid_pts[0].timestamp
        t1 = valid_pts[-1].timestamp
        dt = t1 - t0
        dist = sum(
            float(np.hypot(valid_pts[k].world_pos[0] - valid_pts[k-1].world_pos[0],
                           valid_pts[k].world_pos[1] - valid_pts[k-1].world_pos[1]))
            for k in range(1, len(valid_pts))
        )
        spd = (dist / dt) * 3.6 if dt > 0 else 0
        print(f"  Overall Path Avg Speed: dist={dist:.2f}m, dt={dt:.3f}s -> {spd:.2f} km/h")
