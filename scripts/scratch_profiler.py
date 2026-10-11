import time
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("packages"))

import cv2
import numpy as np
from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection.yolox import YOLOXDetector
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.intelligence.traffic import (
    VehicleCountingEngine, TrafficFlowEngine, LaneIntelligenceEngine,
    VehicleDensityEngine, QueueDetector, CongestionEngine
)
from visionradar.intelligence.events import TrafficEventEngine


def profile_video_pipeline(video_path: str, max_frames: int = 150):
    print(f"\n=======================================================")
    print(f"PROFILING PIPELINE ON: {video_path}")
    print(f"=======================================================")
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # 1. Video file properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps
    cap.release()
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}, Duration: {duration_s:.2f}s")

    # 2. Initialize modules
    t0_init = time.perf_counter()
    decoder = VideoDecoder(video_path)
    detector = YOLOXDetector(model_path="data/models/yolox_nano.onnx")
    tracker = ByteTrackTracker()
    
    img_pts = [[330.0, 160.0], [470.0, 160.0], [748.0, 435.0], [51.0, 435.0]]
    world_pts = [[-1.0, 150.0], [13.0, 150.0], [13.0, 0.5], [-1.0, 0.5]]
    calibrator = HomographyCalibrator(img_pts, world_pts, camera_height=9.0, pitch_deg=14.0)
    speed_estimator = MonocularSpeedEstimator(calibrator=calibrator, fps=fps)
    
    counting_engine = VehicleCountingEngine()
    flow_engine = TrafficFlowEngine(interval_sec=duration_s)
    lane_engine = LaneIntelligenceEngine()
    density_engine = VehicleDensityEngine(road_length_m=150.0, image_points=calibrator.image_points)
    queue_detector = QueueDetector()
    congestion_engine = CongestionEngine(policy_version="v1.0.0")
    event_engine = TrafficEventEngine()
    t_init = (time.perf_counter() - t0_init) * 1000.0
    print(f"Initialization time: {t_init:.2f}ms")

    # Metrics accumulators
    decode_times = []
    detector_times = []
    tracker_times = []
    speed_times = []
    intel_times = []
    total_frame_times = []
    
    active_tracks = {}
    frames_processed = 0

    t_start_pipeline = time.perf_counter()
    for frame_idx, timestamp, frame in decoder.decode_frames():
        if frames_processed >= max_frames:
            break

        t_f_start = time.perf_counter()

        # Decoding time is measured by the generator step, but we can measure subsequent stages
        # 1. Detector
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000.0
        detector_times.append(t_det)

        # 2. Tracker
        t0 = time.perf_counter()
        tracks = tracker.update(dets, frame_idx, timestamp)
        t_trk = (time.perf_counter() - t0) * 1000.0
        tracker_times.append(t_trk)

        # 3. Speed estimator (per track in frame)
        t0 = time.perf_counter()
        for trk in tracks:
            active_tracks[trk.track_id] = trk
            speed_estimator.project_trajectory(trk.trajectory, source_width=width, source_height=height)
            est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=frame_idx)
        t_spd = (time.perf_counter() - t0) * 1000.0
        speed_times.append(t_spd)

        # 4. Intelligence engines
        t0 = time.perf_counter()
        active_track_dicts = [t.to_dict() for t in tracks]
        density_engine.update_frame_occupancy(frame_idx, timestamp, active_track_dicts, calibrator)
        queue_detector.update(active_track_dicts, frame_idx, timestamp)
        newly_events = event_engine.process_frame(active_track_dicts, frame_idx, timestamp)
        t_intel = (time.perf_counter() - t0) * 1000.0
        intel_times.append(t_intel)

        t_frame_total = (time.perf_counter() - t_f_start) * 1000.0
        total_frame_times.append(t_frame_total)
        frames_processed += 1

    total_pipeline_time_s = time.perf_counter() - t_start_pipeline

    # Summary statistics
    def stats(arr):
        if not arr: return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.percentile(arr, 95)), float(np.max(arr))

    mean_det, p95_det, max_det = stats(detector_times)
    mean_trk, p95_trk, max_trk = stats(tracker_times)
    mean_spd, p95_spd, max_spd = stats(speed_times)
    mean_intel, p95_intel, max_intel = stats(intel_times)
    mean_frame, p95_frame, max_frame = stats(total_frame_times)

    effective_fps = frames_processed / max(0.001, total_pipeline_time_s)
    real_time_factor = effective_fps / fps

    print("\n--- MEASURED LATENCY BREAKDOWN (per frame) ---")
    print(f"Frames Profiled: {frames_processed}")
    print(f"Detector (YOLOX-Nano ONNX CPU): Mean={mean_det:.2f}ms | p95={p95_det:.2f}ms | Max={max_det:.2f}ms")
    print(f"Tracker (ByteTrack):           Mean={mean_trk:.2f}ms | p95={p95_trk:.2f}ms | Max={max_trk:.2f}ms")
    print(f"Speed Estimation:              Mean={mean_spd:.2f}ms | p95={p95_spd:.2f}ms | Max={max_spd:.2f}ms")
    print(f"Traffic Intelligence Engines:  Mean={mean_intel:.2f}ms | p95={p95_intel:.2f}ms | Max={max_intel:.2f}ms")
    print(f"Total Frame Processing:        Mean={mean_frame:.2f}ms | p95={p95_frame:.2f}ms | Max={max_frame:.2f}ms")
    print(f"Effective Processing FPS:      {effective_fps:.2f} FPS")
    print(f"Real-Time Factor (RTF):        {real_time_factor:.3f}x (1.0 = real-time, <1.0 = slower than real-time)")
    print(f"Estimated time for 10-min video (18,000 frames): {(18000 * mean_frame / 1000.0) / 60.0:.2f} minutes")

    # Measure End-of-job finalization time on accumulated tracks
    t0_fin = time.perf_counter()
    fin_eval_count = 0
    for trk_id, trk in active_tracks.items():
        for p in trk.trajectory.points:
            est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=p.frame_index)
            fin_eval_count += 1
    t_fin = (time.perf_counter() - t0_fin) * 1000.0
    print(f"\nFinalization trajectory re-evaluations: {fin_eval_count} calls took {t_fin:.2f}ms ({t_fin/max(1, fin_eval_count):.2f}ms/point)")

if __name__ == "__main__":
    profile_video_pipeline("data/videos/Traffic1.mp4", max_frames=90)
    profile_video_pipeline("data/videos/synthetic_highway.mp4", max_frames=90)
