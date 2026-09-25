import urllib.request
import urllib.parse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

def test_upload():
    # 1. Get or create project
    req = urllib.request.Request("http://localhost:8000/api/v1/projects")
    with urllib.request.urlopen(req) as resp:
        projects = json.loads(resp.read().decode("utf-8"))
        project_id = projects[0]["id"]

    print(f"Using Project #{project_id}")

    # 2. Upload Traffic1.mp4 via HTTP multipart form
    video_path = "Traffic1.mp4"
    if not os.path.exists(video_path):
        video_path = "data/videos/Traffic1.mp4"

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="Traffic1.mp4"\r\n')
    body.extend(b"Content-Type: video/mp4\r\n\r\n")
    body.extend(video_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    upload_req = urllib.request.Request(
        f"http://localhost:8000/api/v1/projects/{project_id}/videos",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    with urllib.request.urlopen(upload_req) as resp:
        video_res = json.loads(resp.read().decode("utf-8"))
        video_id = video_res["id"]

    print(f"Uploaded Traffic1.mp4 successfully! Video ID: {video_id}, Dims: {video_res['width']}x{video_res['height']}")

    # 3. Create Calibration
    calib_payload = json.dumps({
        "image_points": [[330.0, 160.0], [470.0, 160.0], [748.0, 435.0], [51.0, 435.0]],
        "world_points": [[-1.0, 150.0], [13.0, 150.0], [13.0, 0.5], [-1.0, 0.5]],
        "camera_height_m": 5.5,
        "pitch_deg": 18.0
    }).encode("utf-8")

    calib_req = urllib.request.Request(
        f"http://localhost:8000/api/v1/videos/{video_id}/calibrations",
        data=calib_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(calib_req) as resp:
        calib_res = json.loads(resp.read().decode("utf-8"))
        calib_id = calib_res["id"]

    # 4. Start Processing Job
    job_payload = json.dumps({
        "calibration_id": calib_id,
        "config_json": {"detector": "yolox", "model_path": "data/models/yolox_nano.onnx"}
    }).encode("utf-8")

    job_req = urllib.request.Request(
        f"http://localhost:8000/api/v1/videos/{video_id}/jobs",
        data=job_payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(job_req) as resp:
        job_res = json.loads(resp.read().decode("utf-8"))
        job_id = job_res["id"]

    print(f"Started Processing Job #{job_id} for Traffic1.mp4. Triggering worker...")

    # Directly run worker for job_id
    from visionradar.worker.job_worker import run_job
    run_job(job_id)

    # 5. Fetch Job Status
    status_req = urllib.request.Request(f"http://localhost:8000/api/v1/jobs/{job_id}")
    with urllib.request.urlopen(status_req) as resp:
        st = json.loads(resp.read().decode("utf-8"))
        print(f"\nJob #{job_id} status: {st['status']} ({st['stage']}, {st['progress_pct']:.1f}%)")

    # 6. Fetch Traffic Overview API
    overview_req = urllib.request.Request(f"http://localhost:8000/api/v1/jobs/{job_id}/traffic/overview")
    with urllib.request.urlopen(overview_req) as resp:
        overview = json.loads(resp.read().decode("utf-8"))
        print("\n=== TRAFFIC INTELLIGENCE OVERVIEW FOR Traffic1.mp4 ===")
        print(json.dumps(overview, indent=2))

if __name__ == "__main__":
    test_upload()
