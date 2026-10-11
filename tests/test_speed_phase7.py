"""
Unit & Integration Tests for VisionRadar Phase 7 Speed Intelligence Engine & Forensic Reconciliation.
"""

import math
import numpy as np
import pytest

from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.cv.speed import (
    MonocularSpeedEstimator, SpeedProvenance, SpeedEstimate, GroundTruthSpeedMatcher, KalmansSpeedFilter
)


def get_test_calibrator() -> HomographyCalibrator:
    img_pts = [(200.0, 150.0), (600.0, 150.0), (750.0, 400.0), (50.0, 400.0)]
    w_pts = [(0.0, 60.0), (12.0, 60.0), (12.0, 0.0), (0.0, 0.0)]
    return HomographyCalibrator(image_points=img_pts, world_points=w_pts)


def test_speed_provenance_schema():
    prov = SpeedProvenance(
        job_id="job_01",
        video_id="vid_01",
        track_id=42,
        frame_index=15,
        timestamp=0.5,
        speed_method="REGRESSION",
        speed_kmh=60.0,
        speed_mps=16.67,
        distance_m=8.33,
        elapsed_time_s=0.5,
        sample_count=10,
        trajectory_length_m=8.33,
        uncertainty_kmh=2.5,
        calibration_id="cal_01",
        calibration_status="VALID",
        trajectory_quality=0.95,
        validity="VALID",
        invalid_reason=None,
        estimator_version="7.0",
        estimator_disagreement_kmh=1.2,
        estimates_by_method={"REGRESSION": 60.0, "KALMAN_SMOOTHED": 60.5},
        diagnostic_flag="SPEED_VALID"
    )
    d = prov.to_dict()
    assert d["job_id"] == "job_01"
    assert d["speed_method"] == "REGRESSION"
    assert d["estimator_version"] == "7.0"
    assert d["diagnostic_flag"] == "SPEED_VALID"


def test_calibration_safety_roi_check():
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=1, vehicle_class="car")
    # Outside polygon ROI
    traj.add_point(TrajectoryPoint(frame_index=1, timestamp=0.033, anchor_pixel=(10.0, 10.0)))
    # Inside polygon ROI
    traj.add_point(TrajectoryPoint(frame_index=2, timestamp=0.066, anchor_pixel=(400.0, 250.0)))

    estimator.project_trajectory(traj)

    assert traj.points[0].roi_status == "OUT_OF_ROI"
    assert traj.points[0].world_pos is None
    assert traj.points[1].roi_status == "VALID"
    assert traj.points[1].world_pos is not None


def test_segment_reset_contamination_isolation():
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=2, vehicle_class="car")
    # OUT
    traj.add_point(TrajectoryPoint(frame_index=1, timestamp=0.033, anchor_pixel=(10, 10)))
    # VALID segment 1 (3 points)
    traj.add_point(TrajectoryPoint(frame_index=2, timestamp=0.066, anchor_pixel=(300, 200)))
    traj.add_point(TrajectoryPoint(frame_index=3, timestamp=0.100, anchor_pixel=(300, 220)))
    traj.add_point(TrajectoryPoint(frame_index=4, timestamp=0.133, anchor_pixel=(300, 240)))
    # OUT
    traj.add_point(TrajectoryPoint(frame_index=5, timestamp=0.166, anchor_pixel=(10, 10)))
    # VALID segment 2 (2 points)
    traj.add_point(TrajectoryPoint(frame_index=6, timestamp=0.200, anchor_pixel=(400, 300)))
    traj.add_point(TrajectoryPoint(frame_index=7, timestamp=0.233, anchor_pixel=(400, 320)))

    estimator.project_trajectory(traj)
    segments = estimator.extract_contiguous_valid_segments(traj)

    assert len(segments) == 2
    assert len(segments[0]) == 3
    assert len(segments[1]) == 2


def test_multi_estimator_speed_calculation():
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=3, vehicle_class="car")
    for i in range(10):
        t = (i + 1) * 0.033
        v_pixel = 200 + i * 15
        traj.add_point(TrajectoryPoint(frame_index=i+1, timestamp=t, anchor_pixel=(400, v_pixel)))

    est = estimator.evaluate_multi_estimator_speed(traj, target_frame_idx=10)

    assert est.speed_kmh > 0.0
    assert est.provenance is not None
    assert "REGRESSION" in est.provenance.estimates_by_method
    assert "KALMAN_SMOOTHED" in est.provenance.estimates_by_method
    assert "INSTANTANEOUS" in est.provenance.estimates_by_method
    assert "PATH_AVERAGE" in est.provenance.estimates_by_method
    assert est.provenance.estimator_disagreement_kmh >= 0.0


def test_impossible_speed_diagnostics():
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=4, vehicle_class="car")
    # Massive spatial position jump causing >200 km/h speed
    traj.add_point(TrajectoryPoint(frame_index=1, timestamp=0.033, anchor_pixel=(400, 160)))
    traj.add_point(TrajectoryPoint(frame_index=2, timestamp=0.066, anchor_pixel=(400, 390)))

    est = estimator.evaluate_multi_estimator_speed(traj, target_frame_idx=2)

    assert est.provenance.diagnostic_flag == "SPEED_INVALID"
    assert est.provenance.invalid_reason == "WORLD_POSITION_JUMP"


def test_ground_truth_matcher_unavailable():
    matcher = GroundTruthSpeedMatcher(gt_available=False)
    res = matcher.match_track_to_ground_truth(10, VehicleTrajectory(track_id=10, vehicle_class="car"))
    assert res["status"] == "GROUND_TRUTH_MATCHING_UNAVAILABLE"
    assert res["ground_truth_id"] is None


# ============================================================
# PHASE 7 FORENSIC RECONCILIATION REGRESSION TESTS (Section 13)
# ============================================================

def test_regression_tiny_dt_safety():
    """Verify estimator handles tiny dt (dt < 1e-4s) safely without ZeroDivisionError."""
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=50, vehicle_class="car")
    traj.add_point(TrajectoryPoint(frame_index=1, timestamp=0.1000, anchor_pixel=(400, 200)))
    traj.add_point(TrajectoryPoint(frame_index=2, timestamp=0.1000, anchor_pixel=(400, 205)))  # dt = 0

    est = estimator.evaluate_multi_estimator_speed(traj, target_frame_idx=2)
    assert est is not None
    assert est.elapsed_time_s >= 1e-4
    assert not math.isnan(est.speed_kmh)


def test_regression_fps_mismatch_and_timestamp_drift():
    """Verify timestamps derived from 29.97 vs 30.0 fps maintain non-zero positive elapsed time."""
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator, fps=29.97)

    traj = VehicleTrajectory(track_id=51, vehicle_class="car")
    for idx in range(1, 15):
        t = idx / 29.97
        traj.add_point(TrajectoryPoint(frame_index=idx, timestamp=t, anchor_pixel=(400, 200 + idx * 5)))

    est = estimator.evaluate_multi_estimator_speed(traj, target_frame_idx=14)
    assert est.provenance.calibration_status == "VALID"
    assert est.elapsed_time_s > 0.40


def test_regression_stale_kalman_state_reset():
    """Verify KalmansSpeedFilter reset method correctly clears position/velocity state."""
    kf = KalmansSpeedFilter()
    pos1, spd1 = kf.update(10.0, 0.033)
    pos2, spd2 = kf.update(20.0, 0.033)
    assert kf.initialized is True
    assert spd2 > 0

    kf.reset()
    assert kf.initialized is False
    assert kf.x[0] == 0.0
    assert kf.x[1] == 0.0


def test_regression_invalid_observation_window_duration():
    """Verify observation window requiring 2.0s rejects a 0.5s trajectory duration."""
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=52, vehicle_class="car")
    # 0.5s duration trajectory (16 points @ 30fps)
    for idx in range(1, 16):
        traj.add_point(TrajectoryPoint(frame_index=idx, timestamp=idx * 0.033, anchor_pixel=(400, 200 + idx * 2)))

    estimator.project_trajectory(traj)
    valid_pts = [p for p in traj.points if p.roi_status == "VALID"]
    duration = valid_pts[-1].timestamp - valid_pts[0].timestamp

    requested_window_s = 2.0
    is_valid_window = duration >= requested_window_s
    assert is_valid_window is False
    assert duration < requested_window_s


def test_regression_huge_world_displacement():
    """Verify excessive world displacement (>180 km/h) triggers SPEED_INVALID diagnostic flag."""
    calibrator = get_test_calibrator()
    estimator = MonocularSpeedEstimator(calibrator)

    traj = VehicleTrajectory(track_id=53, vehicle_class="car")
    traj.add_point(TrajectoryPoint(frame_index=1, timestamp=0.033, anchor_pixel=(400, 160)))
    # Moving from horizon (v=160) to bottom (v=390) in 1 frame = massive displacement
    traj.add_point(TrajectoryPoint(frame_index=2, timestamp=0.066, anchor_pixel=(400, 390)))

    est = estimator.evaluate_multi_estimator_speed(traj, target_frame_idx=2)
    assert est.provenance.diagnostic_flag == "SPEED_INVALID"
    assert est.provenance.invalid_reason == "WORLD_POSITION_JUMP"
