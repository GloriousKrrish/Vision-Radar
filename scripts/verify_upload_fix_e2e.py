import os
import sys
import json
import time
import hashlib
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import Video, Calibration, ProcessingJob, Track, SpeedMeasurement
from visionradar.worker.job_worker import run_job

def run_e2e_verification():
    client = TestClient(app)
    db = SessionLocal()

    print("=================================================================")
    print("STEP 1: VERIFY DEMO PROCESSING PATH (synthetic_highway.mp4)")
    print("=================================================================")
    demo_video = db.query(Video).filter(Video.filename == "synthetic_highway.mp4").first()
    assert demo_video is not None, "Demo video synthetic_highway.mp4 must exist in DB"
    demo_calib = db.query(Calibration).filter(Calibration.video_id == demo_video.id).first()
    assert demo_calib is not None, "Demo calibration must exist for synthetic_highway.mp4"

    demo_job_res = client.post(
        f"/api/v1/videos/{demo_video.id}/jobs",
        json={"calibration_id": demo_calib.id, "config_json": {"detector": "mog2"}}
    )
    assert demo_job_res.status_code == 200
    demo_job_id = demo_job_res.json()["id"]
    print(f"Created Demo Job #{demo_job_id} for Video #{demo_video.id}")
    run_job(demo_job_id)

    demo_job = db.query(ProcessingJob).filter(ProcessingJob.id == demo_job_id).first()
    assert demo_job.status == "SUCCEEDED", f"Demo job failed: {demo_job.error_message}"
    demo_tracks = db.query(Track).filter(Track.job_id == demo_job_id).all()
    demo_speeds = db.query(SpeedMeasurement).join(Track).filter(Track.job_id == demo_job_id).all()
    demo_valid_speeds = [s for s in demo_speeds if s.smoothed_kmh is not None and s.smoothed_kmh > 0]
    print(f"Demo Job #{demo_job_id} Status: {demo_job.status}")
    print(f"  Processed frames: {demo_job.telemetry_json.get('frames_processed')}")
    print(f"  Tracks: {len(demo_tracks)}")
    print(f"  Total speed measurements: {len(demo_speeds)}")
    print(f"  Valid speed measurements: {len(demo_valid_speeds)}")
    assert len(demo_tracks) > 0, "Demo job must produce tracks"
    assert len(demo_valid_speeds) > 0, "Demo job with calibration must produce valid speeds"
    print("DEMO PATH VERIFIED: PASS")

    print("\n=================================================================")
    print("STEP 2: VERIFY USER UPLOAD PATH WITHOUT CALIBRATION (AICC22-Custom)")
    print("=================================================================")
    upload_video_path = "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_Original-video.mp4"
    assert os.path.exists(upload_video_path), f"Test video must exist at {upload_video_path}"
    with open(upload_video_path, "rb") as f:
        file_bytes = f.read()

    file_size = len(file_bytes)
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    print(f"Uploading file: {os.path.basename(upload_video_path)}")
    print(f"  File size: {file_size} bytes")
    print(f"  File SHA-256: {file_sha256}")

    # Multipart upload
    upload_res = client.post(
        "/api/v1/projects/1/videos",
        files={"file": ("user_upload_aicc22.mp4", file_bytes, "video/mp4")}
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    vid_meta = upload_res.json()
    uploaded_vid_id = vid_meta["id"]
    storage_path = vid_meta["storage_path"]

    print(f"Uploaded Video Record ID: {uploaded_vid_id}")
    print(f"  Storage path: {storage_path}")
    print(f"  Stored SHA-256: {vid_meta['sha256_hash']}")
    print(f"  Extracted resolution: {vid_meta['width']}x{vid_meta['height']}")
    print(f"  Extracted FPS: {vid_meta['fps']}")
    print(f"  Extracted duration: {vid_meta['duration_sec']}s")

    # 1. Stored file hash matches uploaded file
    assert vid_meta["sha256_hash"] == file_sha256, "Stored file hash must match uploaded file!"
    assert os.path.exists(storage_path), "Physical storage file must exist on disk!"
    with open(storage_path, "rb") as f:
        actual_disk_hash = hashlib.sha256(f.read()).hexdigest()
    assert actual_disk_hash == file_sha256, "Physical stored file bytes must match upload!"

    # 2. Verify no calibration exists for this new uploaded video
    calib_list_res = client.get(f"/api/v1/videos/{uploaded_vid_id}/calibrations")
    assert calib_list_res.status_code == 200
    assert len(calib_list_res.json()) == 0, "New video must have no calibrations initially"

    # 3. Create job without calibration (user has not calibrated yet)
    job_create_res = client.post(
        f"/api/v1/videos/{uploaded_vid_id}/jobs",
        json={"calibration_id": None}
    )
    assert job_create_res.status_code == 200
    upload_job_data = job_create_res.json()
    upload_job_id = upload_job_data["id"]

    # 4. Verify job references correct video ID and calibration_id is None (no demo fallback!)
    assert upload_job_data["video_id"] == uploaded_vid_id, "Job must reference the newly uploaded video ID"
    assert upload_job_data["calibration_id"] is None, "Job calibration_id must be None (no silent fallback to demo calibration #1)"
    print(f"Created User Upload Job #{upload_job_id}: video_id={uploaded_vid_id}, calibration_id={upload_job_data['calibration_id']}")

    # 5. Run worker on user uploaded video
    print(f"Executing worker for User Upload Job #{upload_job_id}...")
    t_start = time.time()
    run_job(upload_job_id)
    t_duration = time.time() - t_start
    print(f"Worker execution completed in {t_duration:.2f}s")

    # 6. Verify job reached SUCCEEDED
    upload_job = db.query(ProcessingJob).filter(ProcessingJob.id == upload_job_id).first()
    assert upload_job.status == "SUCCEEDED", f"Job failed: {upload_job.error_message}"
    tel = upload_job.telemetry_json or {}
    print(f"User Upload Job #{upload_job_id} Status: {upload_job.status}")
    print(f"  Frames processed: {tel.get('frames_processed')}")
    print(f"  Video resolution: {tel.get('video_resolution')}")
    print(f"  Video FPS: {tel.get('video_fps')}")
    print(f"  Total tracks: {tel.get('total_tracks')}")
    print(f"  Calibration ID: {tel.get('calibration_id')}")
    print(f"  Calibration Validation: {tel.get('calibration_validation')}")

    # 7. Verify correct video properties decoded by worker
    assert tel.get("frames_processed") == 200, f"Expected 200 frames, got {tel.get('frames_processed')}"
    assert tel.get("video_resolution") == [1280, 720], f"Expected [1280, 720], got {tel.get('video_resolution')}"
    assert tel.get("video_fps") == 10.0, f"Expected 10.0 fps, got {tel.get('video_fps')}"
    assert tel.get("calibration_id") is None, "Telemetry calibration_id must be None"
    assert tel.get("calibration_validation", {}).get("status") == "CALIBRATION_REQUIRED", "Calibration validation must be CALIBRATION_REQUIRED"

    # 8. Verify tracks and speed status
    up_tracks = db.query(Track).filter(Track.job_id == upload_job_id).all()
    assert len(up_tracks) > 0, "User uploaded video must produce tracks"
    up_speeds = db.query(SpeedMeasurement).join(Track).filter(Track.job_id == upload_job_id).all()
    calib_req_speeds = [s for s in up_speeds if s.error_components_json and s.error_components_json.get("validity") == "CALIBRATION_REQUIRED"]
    print(f"  DB Tracks count: {len(up_tracks)}")
    print(f"  DB Speed measurements: {len(up_speeds)}")
    print(f"  Measurements with CALIBRATION_REQUIRED: {len(calib_req_speeds)}")
    assert len(calib_req_speeds) == len(up_speeds), "All speed measurements must be CALIBRATION_REQUIRED when video is uncalibrated (no fake speeds/OUT_OF_ROI)!"
    for s in up_speeds:
        assert s.smoothed_kmh is None, "Metric speed must be None when video has no calibration!"
        assert s.instantaneous_kmh is None, "Metric speed must be None when video has no calibration!"

    print("USER UPLOAD PATH (UNCALIBRATED) VERIFIED: PASS")

    print("\n=================================================================")
    print("STEP 3: VERIFY USER UPLOAD PATH WITH EXPLICIT CALIBRATION (Traffic1.mp4)")
    print("=================================================================")
    traffic1_path = "Traffic1.mp4"
    with open(traffic1_path, "rb") as f:
        t1_bytes = f.read()

    upload_t1_res = client.post(
        "/api/v1/projects/1/videos",
        files={"file": ("Traffic1_Calibrated_Upload.mp4", t1_bytes, "video/mp4")}
    )
    assert upload_t1_res.status_code == 200
    t1_vid_meta = upload_t1_res.json()
    t1_vid_id = t1_vid_meta["id"]

    # Explicitly create calibration for this video
    calib_payload = {
        "image_points": [[330.0, 160.0], [470.0, 160.0], [748.0, 435.0], [51.0, 435.0]],
        "world_points": [[-1.0, 150.0], [13.0, 150.0], [13.0, 0.5], [-1.0, 0.5]],
        "camera_height_m": 5.5,
        "pitch_deg": 18.0
    }
    t1_cal_res = client.post(
        f"/api/v1/videos/{t1_vid_id}/calibrations",
        json=calib_payload
    )
    assert t1_cal_res.status_code == 200
    t1_cal_id = t1_cal_res.json()["id"]
    print(f"Created explicit calibration #{t1_cal_id} for Video #{t1_vid_id}")

    # Start job with this valid calibration
    t1_job_res = client.post(
        f"/api/v1/videos/{t1_vid_id}/jobs",
        json={"calibration_id": t1_cal_id}
    )
    assert t1_job_res.status_code == 200
    t1_job_id = t1_job_res.json()["id"]
    print(f"Created Job #{t1_job_id} for Traffic1 with Calibration #{t1_cal_id}")
    run_job(t1_job_id)

    t1_job = db.query(ProcessingJob).filter(ProcessingJob.id == t1_job_id).first()
    assert t1_job.status == "SUCCEEDED"
    t1_tracks = db.query(Track).filter(Track.job_id == t1_job_id).all()
    t1_speeds = db.query(SpeedMeasurement).join(Track).filter(Track.job_id == t1_job_id).all()
    t1_valid_speeds = [s for s in t1_speeds if s.smoothed_kmh is not None and s.smoothed_kmh > 0]
    print(f"Traffic1 Calibrated Job #{t1_job_id} Status: {t1_job.status}")
    print(f"  Frames processed: {t1_job.telemetry_json.get('frames_processed')}")
    print(f"  Tracks: {len(t1_tracks)}")
    print(f"  Speed measurements: {len(t1_speeds)}")
    print(f"  Valid speeds (> 0 km/h): {len(t1_valid_speeds)}")
    assert len(t1_tracks) > 0, "Must generate tracks"
    assert t1_job.telemetry_json.get("calibration_id") == t1_cal_id, "Telemetry must report the explicit calibration ID"
    print("USER UPLOAD PATH (CALIBRATED) VERIFIED: PASS")

    db.close()
    print("\n=================================================================")
    print("ALL VERIFICATION CHECKS PASSED EMPIRICALLY!")
    print("=================================================================")

if __name__ == '__main__':
    run_e2e_verification()
