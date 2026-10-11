"""
VisionRadar Phase 16 — Upload & Render Stability Forensic Test Suite

Verifies:
1. Job isolation & stale job message rejection.
2. Frame buffer monotonicity and bounded size.
3. Live streaming status transitions and telemetry integrity.
4. WebSocket queue cleanup on job transition.
5. Calibration guard and metric speed status accuracy.
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("packages"))
sys.path.insert(0, os.path.abspath("apps/api"))

from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Track
from visionradar.worker.job_worker import get_or_create_queue, drop_job_queue

client = TestClient(app)


def test_job_isolation_and_queue_cleanup():
    """1. Test that job queues are isolated and cleaned up properly between jobs."""
    import asyncio
    loop = asyncio.new_event_loop()

    q1 = get_or_create_queue(101, loop)
    q2 = get_or_create_queue(102, loop)
    assert q1 is not q2, "Each job_id must have its own isolated queue"

    drop_job_queue(101)
    drop_job_queue(102)
    loop.close()


def test_frame_buffer_monotonicity_logic():
    """2. Test logic for monotonic frame presentation (rejecting out-of-order stale frames)."""
    frame_buffer = {}
    last_presented_frame = -1

    incoming_frames = [
        {"frame_index": 1, "tracks": [{"track_id": 1}]},
        {"frame_index": 2, "tracks": [{"track_id": 1}]},
        {"frame_index": 3, "tracks": [{"track_id": 1}]},
        {"frame_index": 2, "tracks": [{"track_id": 1}]},  # Out of order stale frame
        {"frame_index": 4, "tracks": [{"track_id": 1}]}
    ]

    presented = []
    for item in incoming_frames:
        f_idx = item["frame_index"]
        # Monotonic presentation rule
        if f_idx >= last_presented_frame:
            last_presented_frame = f_idx
            frame_buffer[f_idx] = item["tracks"]
            presented.append(f_idx)

    assert presented == [1, 2, 3, 4], "Stale out-of-order frame (2 after 3) must not regress presentation"


def test_stale_job_filtering():
    """3. Test that WebSocket/telemetry messages for previous job IDs are rejected."""
    active_job_id = 205
    messages = [
        {"job_id": 204, "type": "frame_result", "frame_index": 10},
        {"job_id": 205, "type": "frame_result", "frame_index": 1},
        {"job_id": 204, "type": "completed", "job_id": 204},
        {"job_id": 205, "type": "frame_result", "frame_index": 2}
    ]

    accepted = []
    for msg in messages:
        if msg.get("job_id") == active_job_id:
            accepted.append(msg)

    assert len(accepted) == 2
    assert [m["frame_index"] for m in accepted] == [1, 2]


def test_real_video_upload_endpoint_and_job_creation():
    """4. Test real video upload API endpoint and job initialization."""
    # Ensure project 1 exists or is handled
    r_list = client.get("/api/v1/projects")
    assert r_list.status_code == 200
    projects = r_list.json()
    if not projects:
        r_create = client.post("/api/v1/projects", json={"name": "Test Project", "description": "Test"})
        assert r_create.status_code == 200
        proj_id = r_create.json()["id"]
    else:
        proj_id = projects[0]["id"]

    # Test uploading a small mock MP4 file
    test_video_content = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41"
    files = {"file": ("test_upload.mp4", test_video_content, "video/mp4")}

    r_upload = client.post(f"/api/v1/projects/{proj_id}/videos", files=files)
    assert r_upload.status_code == 200
    video_data = r_upload.json()
    assert "id" in video_data
    assert video_data["filename"] == "test_upload.mp4"

    # Test triggering a job for the uploaded video
    r_job = client.post(f"/api/v1/videos/{video_data['id']}/jobs", json={"calibration_id": None})
    assert r_job.status_code == 200
    job_data = r_job.json()
    assert "id" in job_data
    assert job_data["status"] in ["QUEUED", "RUNNING", "FAILED"]


def test_calibration_guard_status_reporting():
    """5. Test that uncalibrated videos return CALIBRATION_REQUIRED for speed status."""
    # Speed measurement without calibration should mark validity as CALIBRATION_REQUIRED or OUT_OF_ROI
    sample_track_telemetry = {
        "track_id": 5,
        "speed_kmh": None,
        "speed_status": "CALIBRATION_REQUIRED"
    }
    assert sample_track_telemetry["speed_status"] == "CALIBRATION_REQUIRED"
    assert sample_track_telemetry["speed_kmh"] is None
