import os
import pytest
from fastapi.testclient import TestClient
from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import Project, Video, Calibration, ProcessingJob, Track, SpeedMeasurement, Violation, Evidence
from visionradar.worker.job_worker import run_job

client = TestClient(app)

def test_e2e_real_video_processing_pipeline():
    """
    End-to-End integration test for Sprint 1 Definition of Done:
    Upload real video -> Validate & Ingest -> Save Calibration -> Run Job ->
    Extract Detections & Tracks -> Compute Homography Speed & Uncertainty ->
    Persist Evidence -> Verify via API & DB.
    """
    db = SessionLocal()

    # 1. Create Project
    proj_res = client.post("/api/v1/projects", json={"name": "Sprint 1 E2E Test Project", "description": "E2E Verification"})
    assert proj_res.status_code == 201
    proj_id = proj_res.json()["id"]

    # 2. Upload Video
    video_path = "data/videos/synthetic_highway.mp4"
    assert os.path.exists(video_path), "Test video file must exist"

    with open(video_path, "rb") as f:
        up_res = client.post(f"/api/v1/projects/{proj_id}/videos", files={"file": ("test_highway.mp4", f, "video/mp4")})
    assert up_res.status_code == 200
    video_id = up_res.json()["id"]

    # 3. Create Calibration
    img_pts = [[330.0, 160.0], [470.0, 160.0], [748.0, 435.0], [51.0, 435.0]]
    world_pts = [[-1.0, 150.0], [13.0, 150.0], [13.0, 0.5], [-1.0, 0.5]]
    calib_res = client.post(f"/api/v1/videos/{video_id}/calibrations", json={
        "image_points": img_pts,
        "world_points": world_pts,
        "camera_height_m": 9.0,
        "pitch_deg": 14.0
    })
    assert calib_res.status_code == 200
    calib_id = calib_res.json()["id"]

    # 4. Trigger Processing Job
    job_res = client.post(f"/api/v1/videos/{video_id}/jobs", json={"calibration_id": calib_id, "config_json": {"detector": "mog2"}})
    assert job_res.status_code == 200
    job_id = job_res.json()["id"]

    # 5. Execute Processing Job Worker Synchronously
    run_job(job_id)

    # 6. Verify Job Completion
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert job is not None
    assert job.status == "SUCCEEDED"
    assert job.progress_pct == 100.0

    # 7. Verify Tracks & Trajectories
    tracks = db.query(Track).filter(Track.job_id == job_id).all()
    assert len(tracks) > 0, "Job must produce real tracked vehicles"

    for trk in tracks:
        assert trk.track_id > 0
        assert trk.vehicle_class in ["Car", "SUV", "Truck", "Motorcycle", "Sedan"]
        assert len(trk.trajectory_json) > 0, "Track trajectory points must be persisted"

    # 8. Verify Speed Measurements & Uncertainty
    speed_measurements = db.query(SpeedMeasurement).filter(SpeedMeasurement.track_id == tracks[0].id).all()
    assert len(speed_measurements) > 0, "Speed measurements must be generated"

    valid_sm = [s for s in speed_measurements if s.smoothed_kmh is not None and s.smoothed_kmh > 0]
    if valid_sm:
        sm = valid_sm[0]
        assert sm.smoothed_kmh > 0.0
        assert sm.uncertainty_kmh >= 0.8
        assert sm.confidence_low_kmh <= sm.smoothed_kmh <= sm.confidence_high_kmh
        assert "homography_perspective_pct" in sm.error_components_json

    # 9. Verify Candidate Violations & Evidence
    viols = db.query(Violation).filter(Violation.job_id == job_id).all()
    if viols:
        ev = db.query(Evidence).filter(Evidence.violation_id == viols[0].id).first()
        assert ev is not None
        assert os.path.exists(ev.full_frame_path)
        assert os.path.exists(ev.crop_frame_path)

    db.close()
