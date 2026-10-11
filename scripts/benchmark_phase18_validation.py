"""
Phase 18 Comprehensive Pipeline Benchmark & Performance Validation Script.
Measures:
1. Video decoding FPS.
2. Detector FPS (ONNX Runtime CPU vs baseline).
3. Tracker FPS (Two-Stage ByteTrack).
4. Speed estimation & Intelligence FPS.
5. Complete pipeline FPS, Mean and p95 latency.
6. Total job duration, RTF, CPU/RAM utilization.
7. Queue depth, dropped frames, and finalization duration.
Runs on:
- data/videos/Traffic1.mp4 (1080p, 29.97 FPS)
- data/videos/synthetic_highway.mp4 (450p, 30.00 FPS)
"""
import os
import sys
import time
import json
import psutil
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("packages"))

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


def benchmark_pipeline(video_path: str, max_frames: int = 90):
    print(f"\n=======================================================")
    print(f"PHASE 18 VALIDATION BENCHMARK: {video_path}")
    print(f"=======================================================")
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps
    cap.release()
    print(f"Video Info: {width}x{height} @ {fps:.2f} FPS | Total Frames: {total_frames} | Duration: {duration_s:.2f}s")

    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Measure pure decoding FPS
    t0_dec = time.perf_counter()
    dec_test = VideoDecoder(video_path)
    dec_count = 0
    for f_idx, ts, frm in dec_test.decode_frames():
        dec_count += 1
        if dec_count >= max_frames:
            break
    t_dec_total = time.perf_counter() - t0_dec
    pure_decoding_fps = dec_count / max(0.001, t_dec_total)
    mean_dec_ms = (t_dec_total / max(1, dec_count)) * 1000.0
    print(f"Pure Video Decoding: {pure_decoding_fps:.2f} FPS ({mean_dec_ms:.2f} ms/frame)")

    # 2. Pipeline Initialization
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
    init_ms = (time.perf_counter() - t0_init) * 1000.0
    print(f"Pipeline Initialization: {init_ms:.2f} ms")

    # Accumulators
    detector_times = []
    tracker_times = []
    speed_times = []
    intel_times = []
    frame_latencies = []
    active_tracks = {}
    cached_speed_estimates = {}

    frames_processed = 0
    cpu_measurements = []

    t_start = time.perf_counter()
    for frame_idx, timestamp, frame in decoder.decode_frames():
        if frames_processed >= max_frames:
            break

        tf0 = time.perf_counter()

        # Detector
        t0 = time.perf_counter()
        dets = detector.detect(frame)
        t_det = (time.perf_counter() - t0) * 1000.0
        detector_times.append(t_det)

        # Tracker
        t0 = time.perf_counter()
        tracks = tracker.update(dets, frame_idx, timestamp)
        t_trk = (time.perf_counter() - t0) * 1000.0
        tracker_times.append(t_trk)

        # Speed Estimator
        t0 = time.perf_counter()
        for trk in tracks:
            active_tracks[trk.track_id] = trk
            speed_estimator.project_trajectory(trk.trajectory, source_width=width, source_height=height)
            est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=frame_idx)
            if trk.track_id not in cached_speed_estimates:
                cached_speed_estimates[trk.track_id] = {}
            cached_speed_estimates[trk.track_id][frame_idx] = est
        t_spd = (time.perf_counter() - t0) * 1000.0
        speed_times.append(t_spd)

        # Intelligence
        t0 = time.perf_counter()
        active_track_dicts = [t.to_dict() for t in tracks]
        density_engine.update_frame_occupancy(frame_idx, timestamp, active_track_dicts, calibrator)
        queue_detector.update(active_track_dicts, frame_idx, timestamp)
        newly_events = event_engine.process_frame(active_track_dicts, frame_idx, timestamp)
        t_intel = (time.perf_counter() - t0) * 1000.0
        intel_times.append(t_intel)

        frame_latency = (time.perf_counter() - tf0) * 1000.0
        frame_latencies.append(frame_latency)
        frames_processed += 1

        if frames_processed % 30 == 0:
            cpu_measurements.append(process.cpu_percent(interval=None))

    total_pipeline_time_s = time.perf_counter() - t_start
    ram_peak_mb = process.memory_info().rss / (1024 * 1024)

    # 3. Finalization Benchmark (Optimized Cached vs Uncached)
    t0_fin_cached = time.perf_counter()
    fin_cached_count = 0
    for trk_id, trk in active_tracks.items():
        trk_cached = cached_speed_estimates.get(trk_id, {})
        for p in trk.trajectory.points:
            est = trk_cached.get(p.frame_index)
            if est is None:
                est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=p.frame_index)
            fin_cached_count += 1
    t_fin_cached_ms = (time.perf_counter() - t0_fin_cached) * 1000.0

    # Uncached baseline simulation
    t0_fin_uncached = time.perf_counter()
    fin_uncached_count = 0
    for trk_id, trk in active_tracks.items():
        for p in trk.trajectory.points:
            est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=p.frame_index)
            fin_uncached_count += 1
    t_fin_uncached_ms = (time.perf_counter() - t0_fin_uncached) * 1000.0

    def stats(arr):
        if not arr: return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.percentile(arr, 95)), float(np.max(arr))

    mean_det, p95_det, max_det = stats(detector_times)
    mean_trk, p95_trk, max_trk = stats(tracker_times)
    mean_spd, p95_spd, max_spd = stats(speed_times)
    mean_intel, p95_intel, max_intel = stats(intel_times)
    mean_lat, p95_lat, max_lat = stats(frame_latencies)

    pipeline_fps = frames_processed / max(0.001, total_pipeline_time_s)
    rtf = pipeline_fps / fps

    results = {
        "video": video_path,
        "width": width,
        "height": height,
        "native_fps": fps,
        "frames_profiled": frames_processed,
        "pure_decoding_fps": round(pure_decoding_fps, 2),
        "pure_decoding_ms": round(mean_dec_ms, 2),
        "detector": {
            "mean_ms": round(mean_det, 2),
            "p95_ms": round(p95_det, 2),
            "max_ms": round(max_det, 2),
            "fps": round(1000.0 / max(0.1, mean_det), 2)
        },
        "tracker": {
            "mean_ms": round(mean_trk, 2),
            "p95_ms": round(p95_trk, 2),
            "max_ms": round(max_trk, 2)
        },
        "speed_estimation": {
            "mean_ms": round(mean_spd, 2),
            "p95_ms": round(p95_spd, 2)
        },
        "intelligence": {
            "mean_ms": round(mean_intel, 2),
            "p95_ms": round(p95_intel, 2)
        },
        "frame_latency": {
            "mean_ms": round(mean_lat, 2),
            "p95_ms": round(p95_lat, 2),
            "max_ms": round(max_lat, 2)
        },
        "pipeline_fps": round(pipeline_fps, 2),
        "rtf": round(rtf, 3),
        "total_duration_s": round(total_pipeline_time_s, 2),
        "ram_start_mb": round(ram_initial_mb, 1),
        "ram_peak_mb": round(ram_peak_mb, 1),
        "ram_delta_mb": round(ram_peak_mb - ram_initial_mb, 1),
        "finalization": {
            "points_count": fin_cached_count,
            "cached_ms": round(t_fin_cached_ms, 2),
            "uncached_ms": round(t_fin_uncached_ms, 2),
            "speedup_factor": round(t_fin_uncached_ms / max(0.01, t_fin_cached_ms), 1)
        },
        "tracker_metrics": tracker.get_tracking_metrics()
    }

    print("\n--- RESULTS SUMMARY ---")
    print(f"Decoding:               {results['pure_decoding_fps']} FPS ({results['pure_decoding_ms']} ms/frame)")
    print(f"Detector (ORT CPU):     Mean={results['detector']['mean_ms']}ms | p95={results['detector']['p95_ms']}ms | Throughput={results['detector']['fps']} FPS")
    print(f"Tracker (ByteTrack 2S): Mean={results['tracker']['mean_ms']}ms | p95={results['tracker']['p95_ms']}ms")
    print(f"Speed Estimation:       Mean={results['speed_estimation']['mean_ms']}ms")
    print(f"Complete Frame Latency: Mean={results['frame_latency']['mean_ms']}ms | p95={results['frame_latency']['p95_ms']}ms | Max={results['frame_latency']['max_ms']}ms")
    print(f"Complete Pipeline FPS:  {results['pipeline_fps']} FPS (RTF: {results['rtf']}x)")
    print(f"Total Run Time:         {results['total_duration_s']}s for {frames_processed} frames")
    print(f"RAM Utilization:        Start={results['ram_start_mb']}MB -> Peak={results['ram_peak_mb']}MB (Delta: +{results['ram_delta_mb']}MB)")
    print(f"EOF Finalization:       Cached={results['finalization']['cached_ms']}ms vs Uncached={results['finalization']['uncached_ms']}ms ({results['finalization']['speedup_factor']}x speedup on {fin_cached_count} points)")
    print(f"Tracking Metrics:       {results['tracker_metrics']}")

    return results


if __name__ == "__main__":
    r1 = benchmark_pipeline("data/videos/Traffic1.mp4", max_frames=90)
    r2 = benchmark_pipeline("data/videos/synthetic_highway.mp4", max_frames=90)
    
    os.makedirs("reports/phase18", exist_ok=True)
    with open("reports/phase18/benchmark_results.json", "w") as f:
        json.dump({"Traffic1": r1, "synthetic_highway": r2}, f, indent=2)
    print("\nBenchmark results saved to reports/phase18/benchmark_results.json")
