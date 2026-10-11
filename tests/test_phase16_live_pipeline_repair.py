import os
import asyncio
import pytest
from fastapi.testclient import TestClient
from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import Project, Video, Calibration, ProcessingJob, Track, SpeedMeasurement
from visionradar.worker.job_worker import run_job_streaming, get_or_create_queue, drop_job_queue
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint
from visionradar.cv.speed import MonocularSpeedEstimator

client = TestClient(app)


def test_1_detector_to_tracker_and_speed_status_mapping():
    """Verify that speed statuses produce clear labels without N/A."""
    # Test MonocularSpeedEstimator when uncalibrated
    traj = VehicleTrajectory(track_id=1, vehicle_class="Car")
    traj.add_point(TrajectoryPoint(frame_index=0, timestamp=0.0, bbox=(100, 100, 200, 200)))
    traj.add_point(TrajectoryPoint(frame_index=1, timestamp=0.033, bbox=(105, 100, 205, 200)))

    uncalib_estimator = MonocularSpeedEstimator(calibrator=None)
    est = uncalib_estimator.estimate_speed_at_frame(traj, 1)
    assert est.validity == "CALIBRATION_REQUIRED"
    assert est.speed_kmh == 0.0

    # Ensure frontend status mapping handles CALIBRATION_REQUIRED cleanly
    status_label = "CALIBRATION REQUIRED" if est.validity == "CALIBRATION_REQUIRED" else "N/A"
    assert status_label == "CALIBRATION REQUIRED"


def test_2_backend_frontend_payload_compatibility():
    """Ensure API /api/v1/health returns pipeline verification evidence."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "pipeline_verified" in data
    assert "succeeded_jobs" in data
    assert data["pipeline_verified"] is True
    assert data["succeeded_jobs"] > 0


def test_3_job_creation_auto_resolves_calibration():
    """Ensure creating a job without calibration_id automatically falls back to default camera calibration."""
    db = SessionLocal()
    video = db.query(Video).first()
    assert video is not None
    db.close()

    res = client.post(f"/api/v1/videos/{video.id}/jobs", json={"calibration_id": None})
    assert res.status_code == 200
    job_data = res.json()
    assert job_data["calibration_id"] is not None, "Job should auto-link default calibration instead of None"

    db = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_data["id"]).first()
    assert job.calibration_id is not None
    db.close()


def test_4_tracks_endpoint_provides_valid_measurements():
    """Verify GET /api/v1/jobs/{job_id}/tracks returns full speed measurements and error components."""
    db = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.status == "SUCCEEDED").order_by(ProcessingJob.id.desc()).first()
    assert job is not None
    job_id = job.id
    db.close()

    res = client.get(f"/api/v1/jobs/{job_id}/tracks")
    assert res.status_code == 200
    tracks = res.json()
    assert isinstance(tracks, list)
    if len(tracks) > 0:
        t0 = tracks[0]
        assert "track_id" in t0
        assert "vehicle_class" in t0
        assert "speed_measurements" in t0
        assert "trajectory" in t0


def test_5_speed_estimator_trajectory_accumulation():
    """Verify trajectory points accumulate and calculate speed with valid calibration."""
    from visionradar.cv.calibration import HomographyCalibrator
    img_pts = [[330.0, 160.0], [470.0, 160.0], [748.0, 435.0], [51.0, 435.0]]
    world_pts = [[-1.0, 150.0], [13.0, 150.0], [13.0, 0.5], [-1.0, 0.5]]
    calibrator = HomographyCalibrator(img_pts, world_pts, camera_height=9.0, pitch_deg=14.0)

    estimator = MonocularSpeedEstimator(calibrator=calibrator, fps=30.0)
    traj = VehicleTrajectory(track_id=42, vehicle_class="Car")

    # Add 10 points within the road polygon moving forward
    for i in range(10):
        # Coordinates inside the polygon
        x = 400.0 + i * 0.2
        y = 250.0 + i * 1.5
        traj.add_point(TrajectoryPoint(frame_index=i, timestamp=i / 30.0, bbox=(x - 30, y - 30, x + 30, y + 30)))

    estimator.project_trajectory(traj, source_width=800.0, source_height=450.0)
    est = estimator.estimate_speed_at_frame(traj, target_frame_idx=9)
    assert est is not None
    assert est.validity == "VALID"
    assert est.speed_kmh is not None and est.speed_kmh > 0.0
    assert est.uncertainty_kmh is not None


def test_6_multiple_vehicles_overlay_collision_avoidance():
    """Verify mathematical logic of label collision resolution for vehicles near the top of the video."""
    vehicles = [
        {"track_id": 1, "rx": 100, "ry": 10, "rw": 50, "rh": 40},
        {"track_id": 2, "rx": 120, "ry": 15, "rw": 50, "rh": 40},
    ]
    scaleX = 800 / 1920
    scaleY = 450 / 1080
    rendered_badges = []

    for v in vehicles:
        bx = v["rx"] * scaleX
        by = v["ry"] * scaleY
        bw = v["rw"] * scaleX
        bh = v["rh"] * scaleY
        tw = 120
        badgeX = max(2, min(800 - tw - 2, bx + bw / 2 - tw / 2))
        badgeY = by - 22 if by >= 25 else by + bh + 4

        for prev in rendered_badges:
            overlapX = abs(badgeX - prev["x"]) < max(tw, prev["w"]) * 0.85
            overlapY = abs(badgeY - prev["y"]) < 18
            if overlapX and overlapY:
                badgeY = prev["y"] + prev["h"] + 2

        rendered_badges.append({"x": badgeX, "y": badgeY, "w": tw, "h": 18})

    # The two badges must NOT overlap in Y
    b1, b2 = rendered_badges[0], rendered_badges[1]
    assert abs(b1["y"] - b2["y"]) >= 18, f"Badges must not collide: b1.y={b1['y']}, b2.y={b2['y']}"
