import os
import pytest
import numpy as np
import cv2

from visionradar.intelligence.traffic.counting import VehicleCountingEngine, VirtualCountingLine
from visionradar.intelligence.traffic.flow import TrafficFlowEngine
from visionradar.intelligence.traffic.density import VehicleDensityEngine
from visionradar.intelligence.traffic.lanes import LaneIntelligenceEngine, LanePolygon
from visionradar.intelligence.traffic.queue import QueueDetector
from visionradar.intelligence.traffic.congestion import CongestionEngine
from visionradar.intelligence.events.engine import TrafficEventEngine, TrackEventState
from visionradar.cv.evidence import EvidenceWriter

def test_vehicle_counting_and_deduplication():
    line = VirtualCountingLine("gate_1", (100, 200), (500, 200), direction_name="SOUTHWARD")
    engine = VehicleCountingEngine(lines=[line])
    
    # Track 1 crosses line from y=150 to y=250
    pts1 = [{"anchor_pixel": (200, 150), "timestamp": 1.0}, {"anchor_pixel": (200, 250), "timestamp": 1.1}]
    evs1 = engine.process_track_update(track_id=1, vehicle_class="Car", trajectory_points=pts1)
    assert len(evs1) == 1
    assert evs1[0]["track_id"] == 1
    assert evs1[0]["gate_id"] == "gate_1"

    # Deduplication test: Track 1 crosses again -> ignored
    pts2 = [{"anchor_pixel": (200, 250), "timestamp": 1.2}, {"anchor_pixel": (200, 300), "timestamp": 1.3}]
    evs2 = engine.process_track_update(track_id=1, vehicle_class="Car", trajectory_points=pts2)
    assert len(evs2) == 0

def test_traffic_flow_normalization():
    engine = TrafficFlowEngine(interval_sec=60.0)
    # 10 vehicles observed over 30 seconds -> (10 * 3600 / 30) = 1200.0 vph
    res = engine.compute_flow_rate(vehicle_count=10, duration_sec=30.0)
    assert res["flow_rate_vph"] == 1200.0
    assert res["vehicle_count"] == 10
    assert res["duration_sec"] == 30.0

def test_occupancy_and_density_calibration_guard():
    engine = VehicleDensityEngine(road_length_m=150.0)
    active_tracks = [
        {"track_id": 1, "bbox": [100, 100, 50, 50], "anchor_pixel": (125, 150)},
        {"track_id": 2, "bbox": [200, 200, 50, 50], "anchor_pixel": (225, 250)}
    ]
    
    # Uncalibrated evaluation
    uncalib_res = engine.compute_density(total_unique_tracks=2, calibration_active=False)
    assert uncalib_res["density_status"] == "NOT_AVAILABLE"

    # Calibrated evaluation
    engine.update_frame_occupancy(frame_idx=1, timestamp=0.0, active_tracks=active_tracks)
    calib_res = engine.compute_density(total_unique_tracks=2, calibration_active=True)
    assert calib_res["density_status"] in ("VALID", "AVAILABLE")
    assert calib_res["mean_density_veh_km"] > 0.0

def test_lane_intelligence():
    engine = LaneIntelligenceEngine()
    tracks = [
        {"track_id": 1, "lane": "Lane 1 (Left)", "speed_kmh": 60.0},
        {"track_id": 2, "lane": "Lane 2 (Center)", "speed_kmh": 70.0}
    ]
    metrics = engine.compute_lane_metrics(tracks)
    assert len(metrics) >= 3

def test_queue_detection_persistence():
    detector = QueueDetector(low_speed_thresh_kmh=15.0, min_persistence_frames=3)
    tracks_slow = [
        {"track_id": 1, "speed_kmh": 5.0, "bbox": [100, 100, 50, 50]},
        {"track_id": 2, "speed_kmh": 4.0, "bbox": [120, 100, 50, 50]}
    ]
    
    # Frame 1: count=1
    detector.update(tracks_slow, current_frame=1, timestamp=0.1)
    # Frame 2: count=2
    detector.update(tracks_slow, current_frame=2, timestamp=0.2)
    # Frame 3: count=3 -> persistence threshold reached
    q3 = detector.update(tracks_slow, current_frame=3, timestamp=0.3)
    assert detector.slow_vehicle_persistence[1] >= 3
    assert detector.slow_vehicle_persistence[2] >= 3

def test_track_event_state_machine_lifecycle():
    sm = TrackEventState(track_id=1, event_type="stopped_vehicle", min_duration_frames=2, min_cooldown_frames=5)
    assert sm.state == "IDLE"

    trig = {"metric_value": 0.0, "severity": "HIGH", "confidence": 0.9, "explanation": "Vehicle stationary"}
    
    # Trigger 1: IDLE -> ACTIVE
    ev1 = sm.update(has_trigger=True, trigger_info=trig, frame_idx=1, timestamp=0.1)
    assert sm.state == "ACTIVE"
    assert ev1 is None

    # Trigger 2: ACTIVE -> still ACTIVE
    ev2 = sm.update(has_trigger=True, trigger_info=trig, frame_idx=2, timestamp=0.2)
    assert sm.state == "ACTIVE"

    # Release 5 frames: ACTIVE -> COOLDOWN -> yields finalized event
    final_ev = None
    for f in range(3, 10):
        e = sm.update(has_trigger=False, trigger_info=None, frame_idx=f, timestamp=f*0.1)
        if e:
            final_ev = e
            break

    assert final_ev is not None
    assert final_ev.event_type == "stopped_vehicle"
    assert final_ev.track_id == 1
    assert sm.state == "COOLDOWN"

def test_congestion_engine_classifier():
    engine = CongestionEngine()
    res_free = engine.evaluate_congestion(avg_speed_kmh=65.0, mean_density_veh_km=20.0, active_queue_count=0)
    assert res_free["congestion_state"] == "FREE_FLOW"

    res_severe = engine.evaluate_congestion(avg_speed_kmh=5.0, mean_density_veh_km=150.0, active_queue_count=3)
    assert res_severe["congestion_state"] == "SEVERE"

def test_evidence_writer(tmp_path):
    writer = EvidenceWriter(output_dir=str(tmp_path))
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = writer.write_evidence_package("evt_test_101", dummy_img, (50, 50, 100, 100), {"type": "wrong_way"})
    assert os.path.exists(res.get("full_frame_path") or res.get("event_frame_path") or os.path.join(str(tmp_path), "evt_test_101", "event_frame.jpg"))
