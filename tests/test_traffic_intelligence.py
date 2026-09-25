import pytest
from visionradar.intelligence.traffic.counting import VehicleCountingEngine, VirtualCountingLine
from visionradar.intelligence.traffic.flow import TrafficFlowEngine
from visionradar.intelligence.traffic.lanes import LaneIntelligenceEngine
from visionradar.intelligence.traffic.density import VehicleDensityEngine
from visionradar.intelligence.traffic.queue import QueueDetector
from visionradar.intelligence.traffic.congestion import CongestionEngine
from visionradar.intelligence.events.engine import TrafficEventEngine

def test_virtual_counting_line():
    line = VirtualCountingLine("gate_1", (0, 100), (200, 100))
    # Crossing from (100, 50) to (100, 150)
    crossed = line.check_crossing((100, 50), (100, 150))
    assert bool(crossed) is True

    # Non-crossing trajectory
    not_crossed = line.check_crossing((100, 50), (100, 80))
    assert bool(not_crossed) is False

def test_vehicle_counting_engine():
    engine = VehicleCountingEngine()
    pts = [
        {"frame_index": 1, "timestamp": 0.1, "anchor_pixel": [100.0, 200.0]},
        {"frame_index": 2, "timestamp": 0.2, "anchor_pixel": [100.0, 300.0]}
    ]
    evs = engine.process_track_update(track_id=1, vehicle_class="Car", trajectory_points=pts)
    assert len(evs) == 1
    assert evs[0]["track_id"] == 1
    assert evs[0]["vehicle_class"] == "Car"

    # Double counting check
    evs_again = engine.process_track_update(track_id=1, vehicle_class="Car", trajectory_points=pts)
    assert len(evs_again) == 0

def test_traffic_flow_engine():
    flow = TrafficFlowEngine(interval_sec=60.0)
    res = flow.compute_flow_rate(vehicle_count=10, duration_sec=30.0)
    assert res["vehicle_count"] == 10
    assert res["flow_rate_vph"] == 1200.0

def test_lane_intelligence_engine():
    engine = LaneIntelligenceEngine()
    lane_name = engine.assign_lane((100.0, 200.0))
    assert lane_name == "Lane 1 (Left)"

    tracks = [
        {"track_id": 1, "lane": "Lane 1 (Left)", "speed_kmh": 70.0},
        {"track_id": 2, "lane": "Lane 1 (Left)", "speed_kmh": 80.0},
        {"track_id": 3, "lane": "Lane 2 (Center)", "speed_kmh": 65.0}
    ]
    stats = engine.compute_lane_metrics(tracks)
    assert len(stats) >= 3

def test_vehicle_density_engine():
    engine = VehicleDensityEngine(road_length_m=200.0)
    res = engine.compute_density(vehicle_count=10, calibration_active=True)
    assert res["density_status"] == "VALID"
    assert res["density_veh_km"] == 50.0

    res_uncalib = engine.compute_density(vehicle_count=10, calibration_active=False)
    assert res_uncalib["density_status"] == "NOT_AVAILABLE"

def test_queue_and_congestion_engine():
    queue_det = QueueDetector(low_speed_thresh_kmh=15.0, min_persistence_frames=2)
    tracks = [
        {"track_id": 1, "speed_kmh": 10.0},
        {"track_id": 2, "speed_kmh": 12.0}
    ]
    queues1 = queue_det.update(tracks, current_frame=1, timestamp=0.1)
    queues2 = queue_det.update(tracks, current_frame=2, timestamp=0.2)
    assert len(queues2) == 1
    assert queues2[0]["status"] == "QUEUE_ACTIVE"

    cong_engine = CongestionEngine()
    cong_res = cong_engine.evaluate_congestion(avg_speed_kmh=8.0, active_queue_count=1)
    assert cong_res["congestion_state"] in ["CONGESTED", "SEVERE"]

def test_traffic_event_engine():
    engine = TrafficEventEngine()
    for f in range(20):
        tracks = [{"track_id": 1, "vehicle_class": "Car", "direction": "NORTHWARD", "speed_kmh": 70.0, "anchor_pixel": (400.0, 500.0 - f * 5.0)}]
        engine.process_frame(tracks, frame_index=f, timestamp=f * 0.033)

    engine.finalize(20, 0.66)
    summary = engine.get_events_summary()
    assert summary["total_event_candidates"] > 0
    assert len(summary["events"]) > 0
    assert summary["events"][0]["event_type"] in ["wrong_way", "WRONG_WAY_CANDIDATE"]
