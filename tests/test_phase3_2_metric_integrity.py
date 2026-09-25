import pytest
import numpy as np
from visionradar.intelligence.traffic.density import VehicleDensityEngine
from visionradar.intelligence.traffic.counting import VehicleCountingEngine, VirtualCountingLine
from visionradar.intelligence.traffic.flow import TrafficFlowEngine
from visionradar.intelligence.traffic.congestion import CongestionEngine
from visionradar.intelligence.events.engine import TrafficEventEngine
from visionradar.intelligence.events.deceleration import SuddenDecelerationDetector
from visionradar.intelligence.events.wrong_way import WrongWayDetector
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator, SpeedEstimate
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint

def test_density_occupancy_distinction():
    """
    Verifies that total unique tracks observed is strictly separated from instantaneous road occupancy.
    """
    engine = VehicleDensityEngine(road_length_m=150.0)

    # Frame 1: 5 active tracks
    f1_tracks = [{"track_id": i, "anchor_pixel": (400, 300)} for i in range(1, 6)]
    res1 = engine.update_frame_occupancy(frame_idx=1, timestamp=0.033, active_tracks=f1_tracks)
    assert res1["current_road_occupancy"] == 5

    # Frame 2: 5 different active tracks
    f2_tracks = [{"track_id": i, "anchor_pixel": (400, 300)} for i in range(6, 11)]
    res2 = engine.update_frame_occupancy(frame_idx=2, timestamp=0.066, active_tracks=f2_tracks)
    assert res2["current_road_occupancy"] == 5

    summary = engine.compute_density()
    assert summary["total_unique_vehicles_observed"] == 10
    assert summary["mean_road_occupancy"] == 5.0
    assert summary["mean_density_veh_km"] == 33.3  # 5 veh / 0.15 km = 33.33 veh/km
    # Total unique (10) != mean occupancy (5.0)


def test_crossing_count_deduplication():
    """
    Verifies line crossing deduplication (already_counted check).
    """
    line = VirtualCountingLine("gate_1", (0.0, 100.0), (800.0, 100.0))
    engine = VehicleCountingEngine(lines=[line])

    # Track 1 crosses gate
    traj = [
        {"anchor_pixel": (400.0, 80.0), "timestamp": 1.0},
        {"anchor_pixel": (400.0, 120.0), "timestamp": 1.1}
    ]
    events1 = engine.process_track_update(track_id=1, vehicle_class="Sedan", trajectory_points=traj)
    assert len(events1) == 1
    assert events1[0]["already_counted"] is True
    assert events1[0]["track_id"] == 1

    # Track 1 updates again across same gate
    traj2 = [
        {"anchor_pixel": (400.0, 120.0), "timestamp": 1.1},
        {"anchor_pixel": (400.0, 150.0), "timestamp": 1.2}
    ]
    events2 = engine.process_track_update(track_id=1, vehicle_class="Sedan", trajectory_points=traj2)
    assert len(events2) == 0

    summary = engine.get_counts_summary()
    assert summary["total_vehicle_count"] == 1


def test_event_state_machine_hysteresis_and_deduplication():
    """
    Verifies temporal state machine (start, continuation, finalization, cooldown) preventing candidate spam.
    """
    det = SuddenDecelerationDetector(drop_threshold_kmh=25.0)
    engine = TrafficEventEngine(detectors=[det])

    # Frames 1-10: Track 1 decelerates (speed drop 30 km/h)
    for f in range(1, 11):
        trk = [{"track_id": 1, "vehicle_class": "Car", "speed_kmh": 80.0 if f % 2 == 1 else 50.0}]
        newly_finalized = engine.process_frame(trk, frame_index=f, timestamp=f * 0.033)
        # Should NOT emit finalized events yet while active
        assert len(newly_finalized) == 0

    # Frames 11-18: Track 1 maintains constant speed (condition releases)
    finalized = []
    for f in range(11, 19):
        trk = [{"track_id": 1, "vehicle_class": "Car", "speed_kmh": 50.0}]
        res = engine.process_frame(trk, frame_index=f, timestamp=f * 0.033)
        finalized.extend(res)

    assert len(finalized) == 1
    ev = finalized[0]
    assert ev.track_id == 1
    assert ev.event_type == "sudden_deceleration"
    assert ev.status == "FINALIZED"
    assert ev.start_frame == 2

    # Verify Cooldown: immediate new drop during cooldown does NOT create a duplicate event
    for f in range(19, 25):
        trk = [{"track_id": 1, "vehicle_class": "Car", "speed_kmh": 50.0 if f % 2 == 1 else 20.0}]
        res = engine.process_frame(trk, frame_index=f, timestamp=f * 0.033)
        assert len(res) == 0


def test_wrong_way_temporal_persistence():
    """
    Verifies WrongWayDetector requiring temporal persistence & displacement.
    """
    ww_det = WrongWayDetector(expected_direction="SOUTHWARD", expected_vector=(0.0, 1.0), min_persistence_frames=5, min_displacement_px=20.0)
    
    # Track 2 moving NORTHWARD (y decreasing)
    triggers = []
    for f in range(1, 10):
        trk = [{"track_id": 2, "vehicle_class": "Truck", "anchor_pixel": (400.0, 500.0 - f * 5.0)}]
        trig = ww_det.detect_frame_triggers(trk, frame_index=f, timestamp=f * 0.033)
        triggers.extend(trig)

    assert len(triggers) > 0
    t = triggers[0]
    assert t["track_id"] == 2
    assert t["observed_direction"] == "NORTHWARD"
    assert "direction_vector" in t
    assert t["direction_confidence"] > 0.8


def test_speed_provenance():
    """
    Verifies SpeedEstimate provenance parameters (speed_method, distance_m, elapsed_time_s, validity, sample_count).
    """
    calib = HomographyCalibrator()
    estimator = MonocularSpeedEstimator(calibrator=calib)

    traj = VehicleTrajectory(track_id=1, vehicle_class="Sedan")
    for i in range(10):
        pt = TrajectoryPoint(frame_index=i, timestamp=i * 0.1, anchor_pixel=(400.0 + i * 2.0, 300.0 + i * 5.0))
        pt.world_pos = (i * 0.5, i * 2.0)
        traj.add_point(pt)

    est_reg = estimator.estimate_speed_at_frame(traj, target_frame_idx=5)
    assert est_reg is not None
    assert est_reg.speed_method == "REGRESSION"
    assert est_reg.sample_count >= 3
    assert est_reg.validity == "VALID"

    est_path = estimator.estimate_path_average_speed(traj)
    assert est_path is not None
    assert est_path.speed_method == "PATH_AVERAGE"
    assert est_path.sample_count == 10
    assert est_path.distance_m > 0.0


def test_calibration_validation_sanity():
    """
    Verifies calibration sanity check logic.
    """
    # Valid convex quad
    img_pts = [(100.0, 100.0), (500.0, 100.0), (600.0, 400.0), (50.0, 400.0)]
    world_pts = [(0.0, 150.0), (12.0, 150.0), (12.0, 0.0), (0.0, 0.0)]
    calib_valid = HomographyCalibrator(image_points=img_pts, world_points=world_pts)
    val1 = calib_valid.validate_calibration()
    assert val1["is_valid"] is True
    assert val1["reprojection_rmse_m"] <= 15.0

    # Non-convex quad
    img_pts_concave = [(100.0, 100.0), (300.0, 250.0), (500.0, 100.0), (300.0, 400.0)]
    calib_invalid = HomographyCalibrator(image_points=img_pts_concave, world_points=world_pts)
    val2 = calib_invalid.validate_calibration()
    assert val2["is_valid"] is False
    assert "convex" in val2["reason"].lower()


def test_congestion_explainability():
    """
    Verifies CongestionEngine output and explicit classification_reason.
    """
    engine = CongestionEngine(policy_version="v1.0.0")

    # Free Flow evaluation
    res_ff = engine.evaluate_congestion(avg_speed_kmh=65.0, mean_density_veh_km=30.0, queue_length_m=0.0)
    assert res_ff["congestion_state"] == "FREE_FLOW"
    assert "FREE_FLOW because" in res_ff["classification_reason"]
    assert res_ff["policy_version"] == "v1.0.0"

    # Moderate evaluation
    res_mod = engine.evaluate_congestion(avg_speed_kmh=42.0, mean_density_veh_km=85.0, queue_length_m=0.0)
    assert res_mod["congestion_state"] == "MODERATE"
    assert "MODERATE because" in res_mod["classification_reason"]

    # Congested evaluation
    res_cong = engine.evaluate_congestion(avg_speed_kmh=18.0, mean_density_veh_km=100.0, queue_length_m=20.0)
    assert res_cong["congestion_state"] == "CONGESTED"
    assert "CONGESTED because" in res_cong["classification_reason"]
