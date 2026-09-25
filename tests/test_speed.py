import pytest
import numpy as np
# pyrefly: ignore [missing-import]
from visionradar.cv.calibration import HomographyCalibrator
# pyrefly: ignore [missing-import]
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.cv.speed import MonocularSpeedEstimator

def test_speed_estimation_on_linear_trajectory():
    """
    Tests speed recovery on a vehicle moving at constant speed v = 72 km/h (20 m/s).
    """
    # Identity calibrator mapping (1 pixel = 1 meter for simplicity)
    image_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    world_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calibrator = HomographyCalibrator(image_points=image_pts, world_points=world_pts)

    estimator = MonocularSpeedEstimator(calibrator=calibrator, window_size_frames=15, fps=30.0)

    # Generate synthetic trajectory moving along Y-axis at 20 m/s (72 km/h)
    true_speed_ms = 20.0
    true_speed_kmh = true_speed_ms * 3.6  # 72.0 km/h

    traj = VehicleTrajectory(track_id=1, vehicle_class="Car")
    fps = 30.0

    for frame in range(30):
        t = frame / fps
        world_y = 10.0 + true_speed_ms * t
        world_x = 5.0
        bbox = (world_x - 1.0, world_y - 2.0, world_x + 1.0, world_y)
        pt = TrajectoryPoint(frame_index=frame, timestamp=t, bbox=bbox, world_pos=(world_x, world_y))
        traj.add_point(pt)

    # Estimate speed at frame 15
    estimate = estimator.estimate_speed_at_frame(traj, target_frame_idx=15)
    assert estimate is not None
    assert abs(estimate.smoothed_kmh - true_speed_kmh) < 0.5, f"Expected ~{true_speed_kmh} km/h, got {estimate.smoothed_kmh}"
    assert estimate.uncertainty_kmh > 0.0
    assert estimate.confidence_low_kmh <= estimate.smoothed_kmh <= estimate.confidence_high_kmh

def test_deterministic_speed_sanity_cases():
    """
    Task 6 — Speed Sanity Validation:
    Deterministic verification of 10m/1s (36 km/h), 20m/2s (36 km/h), and 0m/1s (0 km/h).
    """
    image_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    world_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    calibrator = HomographyCalibrator(image_points=image_pts, world_points=world_pts)
    estimator = MonocularSpeedEstimator(calibrator=calibrator, window_size_frames=30, fps=30.0)

    # Case 1: 10 meters over 1 second (30 frames) -> Expected 36 km/h
    traj_10m = VehicleTrajectory(track_id=1, vehicle_class="Car")
    for f in range(30):
        t = f / 30.0
        wy = 10.0 * t  # 0 to 10 meters
        pt = TrajectoryPoint(frame_index=f, timestamp=t, bbox=(0.0, wy, 2.0, wy + 2.0), world_pos=(0.0, wy))
        traj_10m.add_point(pt)
    
    est1 = estimator.estimate_speed_at_frame(traj_10m, target_frame_idx=15)
    assert est1 is not None
    assert abs(est1.smoothed_kmh - 36.0) < 0.5, f"Expected 36.0 km/h for 10m/1s, got {est1.smoothed_kmh}"

    # Case 2: 20 meters over 2 seconds (60 frames) -> Expected 36 km/h
    traj_20m = VehicleTrajectory(track_id=2, vehicle_class="Car")
    for f in range(60):
        t = f / 30.0
        wy = 10.0 * t  # 0 to 20 meters over 2s
        pt = TrajectoryPoint(frame_index=f, timestamp=t, bbox=(0.0, wy, 2.0, wy + 2.0), world_pos=(0.0, wy))
        traj_20m.add_point(pt)
    
    est2 = estimator.estimate_speed_at_frame(traj_20m, target_frame_idx=30)
    assert est2 is not None
    assert abs(est2.smoothed_kmh - 36.0) < 0.5, f"Expected 36.0 km/h for 20m/2s, got {est2.smoothed_kmh}"

    # Case 3: 0 meters over 1 second (stationary vehicle) -> Expected 0 km/h
    traj_0m = VehicleTrajectory(track_id=3, vehicle_class="Car")
    for f in range(30):
        t = f / 30.0
        pt = TrajectoryPoint(frame_index=f, timestamp=t, bbox=(5.0, 5.0, 7.0, 7.0), world_pos=(5.0, 5.0))
        traj_0m.add_point(pt)
    
    est3 = estimator.estimate_speed_at_frame(traj_0m, target_frame_idx=15)
    assert est3 is not None
    assert abs(est3.smoothed_kmh - 0.0) < 0.1, f"Expected 0.0 km/h for stationary vehicle, got {est3.smoothed_kmh}"

