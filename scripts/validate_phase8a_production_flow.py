"""
Phase 8A — Production User Flow Validator
Runs the complete end-to-end backend production test via API calls.
"""
import sys
import os
import time
import json
import hashlib
import requests
import threading
import asyncio
import websockets

BASE = "http://localhost:8000/api/v1"
VIDEO_PATH = "Traffic1.mp4"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_flow():
    results = {}
    t_flow_start = time.time()

    # =========================================================
    # STEP 1: Health check
    # =========================================================
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.ok, f"Health check failed: {r.status_code}"
    health = r.json()
    results["health"] = health
    print(f"[1] Health: {health}")

    # =========================================================
    # STEP 2: Ensure project exists (ID 1)
    # =========================================================
    r = requests.get(f"{BASE}/projects", timeout=5)
    projects = r.json()
    if not projects:
        r = requests.post(f"{BASE}/projects", json={
            "name": "Phase8A Test Project",
            "description": "Production flow validation"
        })
        assert r.ok, f"Project creation failed: {r.status_code} {r.text}"
        project_id = r.json()["id"]
    else:
        project_id = projects[0]["id"]
    results["project_id"] = project_id
    print(f"[2] Project ID: {project_id}")

    # =========================================================
    # STEP 3: Upload real video
    # =========================================================
    if not os.path.exists(VIDEO_PATH):
        alt = os.path.join("data", "videos", "Traffic1.mp4")
        if os.path.exists(alt):
            VIDEO_PATH_USED = alt
        else:
            print("ERROR: Traffic1.mp4 not found")
            sys.exit(1)
    else:
        VIDEO_PATH_USED = VIDEO_PATH

    video_hash = sha256_file(VIDEO_PATH_USED)
    file_size = os.path.getsize(VIDEO_PATH_USED)
    results["video_file"] = VIDEO_PATH_USED
    results["video_hash"] = video_hash
    results["video_size_bytes"] = file_size
    print(f"[3] Video: {VIDEO_PATH_USED} | Size: {file_size:,} bytes | SHA256: {video_hash[:12]}...")

    t_upload_start = time.time()
    with open(VIDEO_PATH_USED, "rb") as f:
        upload_resp = requests.post(
            f"{BASE}/projects/{project_id}/videos",
            files={"file": (os.path.basename(VIDEO_PATH_USED), f, "video/mp4")},
            timeout=120
        )
    upload_ms = round((time.time() - t_upload_start) * 1000)
    assert upload_resp.ok, f"Upload failed: {upload_resp.status_code} {upload_resp.text}"
    video_data = upload_resp.json()
    video_id = video_data["id"]
    results["video_id"] = video_id
    results["upload_ms"] = upload_ms
    print(f"[3] Video uploaded: ID={video_id} in {upload_ms}ms")

    # =========================================================
    # STEP 4: Create processing job
    # =========================================================
    t_job_create_start = time.time()
    job_resp = requests.post(
        f"{BASE}/videos/{video_id}/jobs",
        json={"calibration_id": None},
        timeout=15
    )
    job_create_ms = round((time.time() - t_job_create_start) * 1000)
    assert job_resp.ok, f"Job creation failed: {job_resp.status_code} {job_resp.text}"
    job_data = job_resp.json()
    job_id = job_data["id"]
    results["job_id"] = job_id
    results["job_create_ms"] = job_create_ms
    print(f"[4] Job created: ID={job_id} in {job_create_ms}ms | Status: {job_data['status']}")

    # =========================================================
    # STEP 5: Poll for completion (REST fallback path)
    # =========================================================
    print(f"[5] Polling job #{job_id} for completion...")
    t_poll_start = time.time()
    first_detection_reported_at = None
    max_poll_sec = 300  # 5 min timeout

    while True:
        elapsed = time.time() - t_poll_start
        if elapsed > max_poll_sec:
            print("ERROR: Job timed out after 5 minutes")
            results["job_timeout"] = True
            break

        time.sleep(3)
        poll_resp = requests.get(f"{BASE}/jobs/{job_id}", timeout=10)
        if not poll_resp.ok:
            continue
        poll_data = poll_resp.json()
        status = poll_data.get("status")
        progress = poll_data.get("progress_pct", 0)
        stage = poll_data.get("stage", "")
        print(f"  [{elapsed:.0f}s] Job #{job_id}: {status} | {stage} | {progress:.0f}%")

        if status == "SUCCEEDED":
            t_complete = time.time() - t_poll_start
            results["job_completed"] = True
            results["job_completion_sec"] = round(t_complete, 1)
            telemetry = poll_data.get("telemetry") or {}
            results["processing_fps"] = telemetry.get("average_fps")
            results["real_time_factor"] = telemetry.get("real_time_factor")
            results["first_detection_latency_s"] = telemetry.get("first_detection_latency_s")
            results["total_tracks"] = telemetry.get("total_tracks")
            results["frames_processed"] = telemetry.get("frames_processed")
            results["video_fps"] = telemetry.get("video_fps")
            results["video_resolution"] = telemetry.get("video_resolution")
            results["traffic_intelligence"] = telemetry.get("traffic_intelligence")
            print(f"  JOB COMPLETED in {t_complete:.1f}s | FPS={results['processing_fps']} | RTF={results['real_time_factor']}")
            break
        elif status == "FAILED":
            results["job_failed"] = True
            results["job_error"] = poll_data.get("error_message", "Unknown error")
            print(f"  JOB FAILED: {results['job_error'][:200]}")
            break

    # =========================================================
    # STEP 6: Load final tracks
    # =========================================================
    trk_resp = requests.get(f"{BASE}/jobs/{job_id}/tracks", timeout=30)
    if trk_resp.ok:
        tracks = trk_resp.json()
        results["track_count"] = len(tracks)
        valid_speed_tracks = sum(1 for t in tracks if any(
            s.get("smoothed_kmh") and s.get("smoothed_kmh", 0) > 0
            for s in (t.get("speed_measurements") or [])
        ))
        results["valid_speed_tracks"] = valid_speed_tracks
        results["invalid_speed_tracks"] = len(tracks) - valid_speed_tracks

        all_speeds = [s.get("smoothed_kmh", 0) for t in tracks
                      for s in (t.get("speed_measurements") or [])
                      if s.get("smoothed_kmh") and s.get("smoothed_kmh", 0) > 0]
        if all_speeds:
            import numpy as np
            results["mean_speed_kmh"] = round(float(np.mean(all_speeds)), 2)
            results["p85_speed_kmh"] = round(float(np.percentile(all_speeds, 85)), 2)
            results["max_speed_kmh"] = round(float(np.max(all_speeds)), 2)
        print(f"[6] Tracks: {len(tracks)} total | {valid_speed_tracks} with valid speed")
    else:
        results["tracks_error"] = f"HTTP {trk_resp.status_code}"
        print(f"[6] ERROR loading tracks: {trk_resp.status_code}")

    # =========================================================
    # STEP 7: Traffic analytics
    # =========================================================
    traf_resp = requests.get(f"{BASE}/jobs/{job_id}/traffic/overview", timeout=15)
    if traf_resp.ok:
        traf = traf_resp.json()
        results["traffic_overview"] = {
            "total_vehicles": traf.get("total_vehicles"),
            "traffic_intel_keys": list((traf.get("traffic_intelligence") or {}).keys())
        }
        print(f"[7] Traffic overview: {results['traffic_overview']}")

    # =========================================================
    # STEP 8: Video streaming endpoint
    # =========================================================
    stream_resp = requests.get(f"{BASE}/videos/{video_id}/stream",
                               headers={"Range": "bytes=0-65535"}, timeout=10)
    results["video_stream_206"] = (stream_resp.status_code == 206)
    results["video_stream_status"] = stream_resp.status_code
    print(f"[8] Video stream HTTP 206: {results['video_stream_206']} (got {stream_resp.status_code})")

    # =========================================================
    # STEP 9: Persistence check (reload job by ID)
    # =========================================================
    reload_resp = requests.get(f"{BASE}/jobs/{job_id}", timeout=5)
    results["persistence_pass"] = reload_resp.ok and reload_resp.json().get("status") == "SUCCEEDED"
    print(f"[9] Persistence check: {results['persistence_pass']}")

    # =========================================================
    # STEP 10: Second upload isolation check
    # =========================================================
    print("[10] Testing second upload isolation...")
    with open(VIDEO_PATH_USED, "rb") as f:
        upload2_resp = requests.post(
            f"{BASE}/projects/{project_id}/videos",
            files={"file": (os.path.basename(VIDEO_PATH_USED), f, "video/mp4")},
            timeout=120
        )
    if upload2_resp.ok:
        video2_id = upload2_resp.json()["id"]
        job2_resp = requests.post(f"{BASE}/videos/{video2_id}/jobs",
                                  json={"calibration_id": None}, timeout=15)
        if job2_resp.ok:
            job2_id = job2_resp.json()["id"]
            results["second_upload_job_id"] = job2_id
            results["second_upload_isolated"] = (job2_id != job_id and video2_id != video_id)
            print(f"  Second job ID={job2_id} | Isolated: {results['second_upload_isolated']}")
        else:
            results["second_upload_isolated"] = False
    else:
        results["second_upload_isolated"] = False

    # =========================================================
    # SUMMARY
    # =========================================================
    total_flow_sec = round(time.time() - t_flow_start, 1)
    results["total_flow_sec"] = total_flow_sec
    print(f"\n{'='*60}")
    print("PHASE 8A PRODUCTION FLOW VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(results, indent=2, default=str))
    return results

if __name__ == "__main__":
    results = run_flow()
    # Save results
    os.makedirs("reports/phase8", exist_ok=True)
    with open("reports/phase8/production_flow_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to reports/phase8/production_flow_results.json")
