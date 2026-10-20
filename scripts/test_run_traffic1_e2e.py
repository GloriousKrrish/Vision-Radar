import requests
import time
import sys

API_BASE = "http://127.0.0.1:8000"

print(f"Uploading Traffic1.mp4 to {API_BASE}...")
with open("Traffic1.mp4", "rb") as f:
    r = requests.post(f"{API_BASE}/api/v1/projects/1/videos", files={"file": ("Traffic1.mp4", f, "video/mp4")})

if r.status_code != 200:
    print(f"Failed upload: {r.status_code} {r.text}")
    sys.exit(1)

video_data = r.json()
video_id = video_data["id"]
print(f"Video uploaded successfully! Video ID: {video_id}, Duration: {video_data.get('duration_sec')}s, FPS: {video_data.get('fps')}")

# Create calibration
calib_payload = {
    "image_points": [[250, 200], [550, 200], [700, 420], [100, 420]],
    "world_points": [[0, 0], [12, 0], [12, 60], [0, 60]],
    "camera_height_m": 8.0,
    "pitch_deg": 15.0
}
r_cal = requests.post(f"{API_BASE}/api/v1/videos/{video_id}/calibrations", json=calib_payload)
calib_id = r_cal.json().get("id") if r_cal.status_code in (200, 201) else None
print(f"Calibration response: {r_cal.status_code}, Calibration ID: {calib_id}")

# Start job
job_payload = {
    "calibration_id": calib_id,
    "config_json": {
        "confidence_threshold": 0.25,
        "input_resolution": [416, 416]
    }
}
r_job = requests.post(f"{API_BASE}/api/v1/videos/{video_id}/jobs", json=job_payload)
if r_job.status_code not in (200, 201):
    print(f"Failed to create job: {r_job.status_code} {r_job.text}")
    sys.exit(1)

job_data = r_job.json()
job_id = job_data["id"]
print(f"Job created! Job ID: {job_id}. Polling progress...")

start_time = time.time()
while True:
    time.sleep(1.0)
    res = requests.get(f"{API_BASE}/api/v1/jobs/{job_id}")
    j = res.json()
    status = j.get("status")
    progress = j.get("progress_pct", 0)
    stage = j.get("stage")
    elapsed = time.time() - start_time
    print(f"[{elapsed:.1f}s] Status: {status} | Stage: {stage} | Progress: {progress}%")
    if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
        break

total_time = time.time() - start_time
print(f"\nFinal Status: {j.get('status')} in {total_time:.2f}s")
if j.get("status") == "SUCCEEDED":
    r_tracks = requests.get(f"{API_BASE}/api/v1/jobs/{job_id}/tracks")
    tracks = r_tracks.json() if r_tracks.status_code == 200 else []
    print(f"Success! Detected & tracked {len(tracks)} vehicles across the video.")
    telemetry = j.get("telemetry") or {}
    print(f"Telemetry summary: {telemetry}")
else:
    print(f"Job failed with error: {j.get('error_message')}")
