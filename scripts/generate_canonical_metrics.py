import os
import sys
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from visionradar.cv.decoder import VideoDecoder
from visionradar.cv.detection import get_detector
from visionradar.cv.tracker import ByteTrackTracker
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.intelligence.traffic import (
    VehicleCountingEngine, VirtualCountingLine, TrafficFlowEngine,
    VehicleDensityEngine, QueueDetector, CongestionEngine
)
from visionradar.intelligence.events import TrafficEventEngine

def compute_hash(filepath):
    if not os.path.exists(filepath):
        return "N/A"
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_canonical():
    video_path = "data/videos/Traffic1.mp4"
    if not os.path.exists(video_path):
        video_path = "Traffic1.mp4"
    model_path = "data/models/yolox_nano.onnx"

    video_hash = compute_hash(video_path)
    model_hash = compute_hash(model_path)
    
    img_pts = [(400.0, 200.0), (1500.0, 200.0), (1850.0, 1050.0), (70.0, 1050.0)]
    world_pts = [(-2.0, 150.0), (14.0, 150.0), (14.0, 0.0), (-2.0, 0.0)]
    calibrator = HomographyCalibrator(image_points=img_pts, world_points=world_pts)
    
    # Config hash
    config_dict = {
        "detector": "YOLOX-Nano-ONNX",
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "tracker": "ByteTrack",
        "iou_threshold": 0.20,
        "track_buffer": 30,
        "matching_threshold": 0.80,
        "calibration": {
            "image_points": img_pts,
            "world_points": world_pts
        }
    }
    config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    calib_hash = hashlib.sha256(json.dumps(calibrator.to_dict(), sort_keys=True).encode()).hexdigest()

    # Run pass 1
    decoder = VideoDecoder(video_path)
    meta = decoder.get_metadata()
    detector = get_detector("yolox", model_path=model_path, confidence_threshold=0.25)
    tracker = ByteTrackTracker()
    speed_estimator = MonocularSpeedEstimator(calibrator=calibrator, fps=meta.fps)
    counting_engine = VehicleCountingEngine(lines=[
        VirtualCountingLine("gate_1", (100, 650), (1800, 650), direction_name="SOUTHWARD", lane_name="Main Roadway")
    ])
    flow_engine = TrafficFlowEngine(interval_sec=meta.duration_sec)
    density_engine = VehicleDensityEngine(road_length_m=150.0, image_points=img_pts)
    queue_detector = QueueDetector()
    congestion_engine = CongestionEngine(policy_version="v1.0.0")
    event_engine = TrafficEventEngine()

    track_history = {}
    crossing_events = []

    for frame_idx, timestamp, frame in decoder.decode_frames():
        dets = detector.detect(frame, confidence_threshold=0.25)
        tracks = tracker.update(dets, frame_idx, timestamp)
        
        active_track_dicts = []
        for trk in tracks:
            track_history[trk.track_id] = trk
            speed_estimator.project_trajectory(trk.trajectory)
            est = speed_estimator.estimate_speed_at_frame(trk.trajectory, target_frame_idx=frame_idx)
            if est is not None:
                trk.speed_kmh = min(140.0, est.smoothed_kmh)
                trk.speed_uncertainty_kmh = est.uncertainty_kmh

            prev_cnt = len(counting_engine.crossing_records)
            counting_engine.process_track_update(
                trk.track_id, trk.vehicle_class, [p.to_dict() for p in trk.trajectory.points]
            )
            if len(counting_engine.crossing_records) > prev_cnt:
                ev = counting_engine.crossing_records[-1]
                crossing_events.append({
                    "track_id": trk.track_id,
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 2),
                    "gate_id": ev["gate_id"],
                    "direction": ev["direction"],
                    "position": ev["position"]
                })

            active_track_dicts.append(trk.to_dict())

        density_engine.update_frame_occupancy(frame_idx, timestamp, active_track_dicts, calibrator)
        active_queues = queue_detector.update(active_track_dicts, frame_idx, timestamp)
        event_engine.process_frame(active_track_dicts, frame_idx, timestamp)

    event_engine.finalize(meta.total_frames - 1, meta.duration_sec)

    counts_summary = counting_engine.get_counts_summary()
    flow_summary = flow_engine.compute_flow_rate(counts_summary["total_vehicle_count"], meta.duration_sec)
    density_summary = density_engine.compute_density(len(track_history), calibration_active=True)
    events_summary = event_engine.get_events_summary()

    all_speeds = [t.speed_kmh for t in track_history.values() if t.speed_kmh is not None and 0.0 < t.speed_kmh <= 150.0]
    avg_speed = float(np.mean(all_speeds)) if all_speeds else 0.0

    congestion_summary = congestion_engine.evaluate_congestion(
        avg_speed_kmh=avg_speed,
        mean_density_veh_km=density_summary["mean_density_veh_km"],
        mean_occupancy=density_summary["mean_road_occupancy"],
        queue_length_m=queue_detector.max_queue_length_m if hasattr(queue_detector, "max_queue_length_m") else 0.0,
        flow_veh_h=flow_summary["flow_rate_vph"],
        active_queue_count=len(queue_detector.active_queues)
    )

    det_telemetry = detector.get_telemetry()

    canonical_data = {
        "video_hash": video_hash,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "calibration_hash": calib_hash,
        "detections": det_telemetry["post_nms_detection_count"],
        "raw_detections": det_telemetry["raw_detection_count"],
        "tracks": len(track_history),
        "observed_internal_id_switches": getattr(tracker, "total_id_switches", 0),
        "fragmentations": getattr(tracker, "total_fragmentations", 0),
        "speed": {
            "mean_kmh": round(avg_speed, 1),
            "median_kmh": round(float(np.median(all_speeds)), 1),
            "p85_kmh": round(float(np.percentile(all_speeds, 85)), 1)
        },
        "occupancy": {
            "mean_vehicles": density_summary["mean_road_occupancy"],
            "max_vehicles": density_summary["max_road_occupancy"]
        },
        "density": {
            "mean_veh_km": density_summary["mean_density_veh_km"],
            "max_veh_km": density_summary["max_density_veh_km"]
        },
        "crossings": counts_summary["total_vehicle_count"],
        "crossing_events": crossing_events,
        "flow": {
            "flow_rate_vph": flow_summary["flow_rate_vph"]
        },
        "events": {
            "total_finalized_events": events_summary["total_finalized_events"],
            "events_list": events_summary["events"]
        },
        "congestion": {
            "state": congestion_summary["congestion_state"],
            "reason": congestion_summary["classification_reason"]
        }
    }

    # Save to phase3_2_2_canonical_metrics.json
    with open("phase3_2_2_canonical_metrics.json", "w") as f:
        json.dump(canonical_data, f, indent=2)

    os.makedirs("data/debug", exist_ok=True)
    with open("data/debug/phase3_2_2_canonical_metrics.json", "w") as f:
        json.dump(canonical_data, f, indent=2)

    print("Successfully generated phase3_2_2_canonical_metrics.json!")
    print(f"Canonical Post-NMS Detections: {canonical_data['detections']}")
    print(f"Canonical Unique Tracks: {canonical_data['tracks']}")
    print(f"Observed Internal Track-ID Switches: {canonical_data['observed_internal_id_switches']}")
    print(f"Canonical Fragmentations: {canonical_data['fragmentations']}")
    print(f"Canonical Mean Speed: {canonical_data['speed']['mean_kmh']} km/h")
    print(f"Canonical Mean Occupancy: {canonical_data['occupancy']['mean_vehicles']} vehicles")
    print(f"Canonical Mean Density: {canonical_data['density']['mean_veh_km']} veh/km")
    print(f"Canonical Crossings: {canonical_data['crossings']} vehicles")
    print(f"Canonical Flow Rate: {canonical_data['flow']['flow_rate_vph']} veh/h")

if __name__ == "__main__":
    generate_canonical()
