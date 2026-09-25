import os
import sys
import json
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages")))

from apps.api.main import app
from visionradar.models.database import get_db
from visionradar.models.entities import ProcessingJob, Video, Calibration, Track, SpeedMeasurement
from visionradar.cv.calibration import HomographyCalibrator
from visionradar.cv.speed import MonocularSpeedEstimator
from visionradar.cv.trajectory import VehicleTrajectory, TrajectoryPoint

client = TestClient(app)

def run_phase323b_audit():
    db = next(get_db())
    job = db.query(ProcessingJob).filter(ProcessingJob.id == 70).first()
    if not job:
        print("ERROR: Job #70 not found in database!")
        return

    video = db.query(Video).filter(Video.id == job.video_id).first()
    calib = db.query(Calibration).filter(Calibration.id == (job.calibration_id or 1)).first()

    print("=========================================================================================")
    print("VISIONRADAR -- PHASE 3.2.3B CALIBRATION & FRAME INTEGRITY AUDIT")
    print("=========================================================================================")
    print(f"Job ID: {job.id} | Video ID: {job.video_id} | Filename: {video.filename if video else 'N/A'}")
    print(f"Calibration ID: {job.calibration_id} | Calib Version: {calib.version if calib else 1}")
    print(f"Image Points (P1-P4): {calib.image_points_json if calib else []}")
    print(f"World Points (P1-P4): {calib.world_points_json if calib else []}")

    # 1. EXACT FRAME SYNCHRONIZATION AUDIT FOR JOB 70
    print("\n--- 1. EXACT FRAME SYNCHRONIZATION AUDIT (GET /api/v1/jobs/70/frames/200) ---")
    res_200 = client.get(f"/api/v1/jobs/70/frames/200")
    print(f"HTTP Status: {res_200.status_code}")
    data_200 = res_200.json() if res_200.status_code == 200 else {}
    print(f"Requested Frame: 200 | Returned JSON Frame: {data_200.get('frame_index')}")
    print(f"Detections Count at Frame 200: {len(data_200.get('detections', []))}")
    for d in data_200.get('detections', []):
        print(f"  Track #{d['track_id']} ({d['class_name']}): BBox={d['bbox_xyxy']}")

    # 2. CALIBRATION POLYGON MEMBERSHIP TEST
    print("\n--- 2. CALIBRATION POLYGON MEMBERSHIP TEST (P1-P4 ROI) ---")
    calibrator = HomographyCalibrator(image_points=calib.image_points_json, world_points=calib.world_points_json)
    
    test_points = [
        ("Inside ROI (Center)", (400.0, 300.0), True),
        ("Outside ROI Above P1-P2 (Horizon)", (400.0, 100.0), False),
        ("Outside ROI Left", (10.0, 300.0), False),
        ("Outside ROI Right", (780.0, 300.0), False),
        ("Outside ROI Below P3-P4", (400.0, 445.0), False),
    ]

    for label, (u, v), expected in test_points:
        is_in = calibrator.is_point_in_quadrilateral(u, v, 800, 450)
        status_str = "PASS" if is_in == expected else "FAIL"
        print(f"  [{status_str}] {label:<35} Point: ({u:5.1f}, {v:5.1f}) -> In ROI: {is_in} (Expected: {expected})")

    # 3. HOMOGRAPHY SAFETY TEST NEAR SINGULARITY
    print("\n--- 3. HOMOGRAPHY NUMERICAL SAFETY TEST ---")
    singularity_pts = [
        ("At Singularity Line (y=90.88)", (400.0, 90.88)),
        ("Near Horizon (y=100.0)", (400.0, 100.0)),
        ("Valid Road Plane (y=250.0)", (400.0, 250.0))
    ]
    for label, (u, v) in singularity_pts:
        is_stable, reason = calibrator.is_homography_stable(u, v, 800, 450)
        print(f"  {label:<32} Point: ({u:5.1f}, {v:5.1f}) -> Stable: {is_stable} ({reason})")

    # 4. TRACK #1 HORIZON CONTAMINATION AUDIT
    print("\n--- 4. TRACK #1 SPEED CONTAMINATION REGRESSION AUDIT ---")
    trks = db.query(Track).filter(Track.job_id == 70).order_by(Track.id.desc()).all()
    t1 = next((t for t in trks if t.track_id == 1), None)

    if t1:
        traj = VehicleTrajectory(1, "Car")
        [traj.add_point(TrajectoryPoint(p['frame_index'], p['timestamp'], bbox=p['bbox'], anchor_pixel=p['anchor_pixel'])) for p in t1.trajectory_json]
        estimator = MonocularSpeedEstimator(calibrator)
        estimator.project_trajectory(traj, 1920, 1080)

        out_of_roi_cnt = sum(1 for p in traj.points if p.roi_status == "OUT_OF_ROI")
        valid_cnt = sum(1 for p in traj.points if p.roi_status == "VALID")
        print(f"  Track #1 Total Trajectory Points: {len(traj.points)}")
        print(f"  Track #1 OUT_OF_ROI Points: {out_of_roi_cnt}")
        print(f"  Track #1 VALID Inside-ROI Points: {valid_cnt}")

        print("\n  Sample Trajectory Points for Track #1:")
        for idx in [0, 5, 10, 20, 40, 45, 50]:
            if idx < len(traj.points):
                p = traj.points[idx]
                est = estimator.estimate_speed_at_frame(traj, p.frame_index)
                sp_str = f"{est.smoothed_kmh:.1f} km/h" if (est and est.smoothed_kmh) else "N/A"
                print(f"    Frame {p.frame_index:2d}: anchor_1080=({p.anchor_pixel[0]:6.1f}, {p.anchor_pixel[1]:6.1f}) | ROI Status: {p.roi_status:<12} | Est Speed: {sp_str}")
    else:
        print("  Track #1 not found in Job 70")

    print("\n=========================================================================================")

if __name__ == "__main__":
    run_phase323b_audit()
