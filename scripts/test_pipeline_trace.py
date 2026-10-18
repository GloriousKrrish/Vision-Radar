import os
import sys
import json
import time
import hashlib
import asyncio
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import Video, ProcessingJob, Track, Calibration
from visionradar.worker.job_worker import run_job

def test_pipeline_trace():
    client = TestClient(app)
    
    # Let's test with AICC22-Custom_Original-video.mp4
    test_video_path = "DATASETS/Vehicle_Tracking/AICC22-Custom/AICC22-Custom_Original-video.mp4"
    if not os.path.exists(test_video_path):
        print(f"File not found: {test_video_path}")
        return

    with open(test_video_path, "rb") as f:
        content = f.read()
    
    original_size = len(content)
    original_hash = hashlib.sha256(content).hexdigest()
    print(f"Original file: {test_video_path}")
    print(f"  Size: {original_size} bytes")
    print(f"  SHA256: {original_hash}")

    # Step 1: Upload via FastAPI endpoint
    response = client.post(
        "/api/v1/projects/1/videos",
        files={"file": ("AICC22_UserUpload.mp4", content, "video/mp4")}
    )
    print(f"Upload response status: {response.status_code}")
    if response.status_code != 200:
        print("Upload failed:", response.text)
        return
    
    vid_data = response.json()
    video_id = vid_data["id"]
    print(f"Created Video ID: {video_id}")
    print(f"  Recorded storage path: {vid_data['storage_path']}")
    print(f"  Recorded dimensions: {vid_data['width']}x{vid_data['height']} @ {vid_data['fps']}fps, duration={vid_data['duration_sec']}s")
    print(f"  Recorded sha256: {vid_data['sha256_hash']}")

    stored_path = vid_data["storage_path"]
    assert os.path.exists(stored_path), "Stored file must exist on disk!"
    stored_size = os.path.getsize(stored_path)
    with open(stored_path, "rb") as f:
        stored_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Stored file check:")
    print(f"  Size matches: {stored_size == original_size} ({stored_size} vs {original_size})")
    print(f"  Hash matches: {stored_hash == original_hash}")

    # Step 2: Check calibrations for this new video
    calib_resp = client.get(f"/api/v1/videos/{video_id}/calibrations")
    print(f"Calibrations for Video #{video_id}: {calib_resp.json()}")

    # Step 3: Start job (as frontend does: calibration_id is either None or passed)
    # Case 1: with calibration_id=None (what a clean upload should do)
    job_resp = client.post(
        f"/api/v1/videos/{video_id}/jobs",
        json={"calibration_id": None}
    )
    print(f"Job creation response status: {job_resp.status_code}")
    job_data = job_resp.json()
    job_id = job_data["id"]
    print(f"Created Job ID: {job_id}")
    print(f"  Job.video_id: {job_data['video_id']}")
    print(f"  Job.calibration_id: {job_data['calibration_id']}")

    # Let's run the worker synchronously to catch any output/exception
    print(f"Executing worker for Job #{job_id}...")
    t0 = time.time()
    run_job(job_id)
    t1 = time.time()
    print(f"Worker finished in {t1 - t0:.2f}s")

    # Inspect final job record
    db = SessionLocal()
    job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    print(f"Final job status: {job.status}")
    print(f"Final job stage: {job.stage}")
    print(f"Final error: {job.error_message}")
    if job.telemetry_json:
        tel = job.telemetry_json
        print(f"Telemetry summary:")
        print(f"  frames_processed: {tel.get('frames_processed')}")
        print(f"  total_tracks: {tel.get('total_tracks')}")
        print(f"  post_nms_detection_count: {tel.get('post_nms_detection_count')}")
        print(f"  vehicle_detection_count: {tel.get('vehicle_detection_count')}")
        print(f"  video_resolution: {tel.get('video_resolution')}")
        print(f"  video_fps: {tel.get('video_fps')}")

    tracks = db.query(Track).filter(Track.job_id == job_id).all()
    print(f"Tracks count in DB: {len(tracks)}")
    if tracks:
        for t in tracks[:3]:
            print(f"  Track #{t.track_id} ({t.vehicle_class}): points={len(t.trajectory_points_json) if t.trajectory_points_json else 0}, speeds={len(t.speed_measurements)}")

    db.close()

if __name__ == '__main__':
    test_pipeline_trace()
