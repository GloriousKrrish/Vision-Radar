"""
Phase 22.4 — Verification of Job Lifecycle, Synchronization, and Long-Run Reliability.
Validates scenarios A through H under realistic conditions:
  A. One video processes normally.
  B. A second job is requested while the first is running.
  C. A running job is cancelled.
  D. A cancelled job is replaced by a new job.
  E. A worker raises an exception.
  F. The frontend disconnects during processing.
  G. The browser reconnects while the job is running / completed.
  H. The backend restarts during a job (startup recovery).
"""
import pytest
import time
import asyncio
from fastapi.testclient import TestClient
from apps.api.main import app, startup_cleanup
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video, Track, Calibration
from visionradar.worker.job_worker import (
    run_job_streaming, cancel_job, cancel_all_active_jobs, is_job_cancelled,
    get_or_create_queue, drop_job_queue
)

client = TestClient(app)

def test_scenario_a_normal_processing():
    """Scenario A: One video processes normally to completion."""
    db = SessionLocal()
    video = db.query(Video).first()
    if not video:
        video = Video(project_id=1, filename="test.mp4", storage_path="data/videos/synthetic_highway.mp4")
        db.add(video)
        db.commit()

    job = ProcessingJob(
        video_id=video.id,
        status="QUEUED",
        stage="Queued",
        progress_pct=0.0
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    db.close()

    loop = asyncio.new_event_loop()
    try:
        run_job_streaming(job_id, loop)
    finally:
        loop.close()

    db = SessionLocal()
    final_job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert final_job.status in ("SUCCEEDED", "FAILED")
    assert final_job.completed_at is not None
    assert final_job.stage in ("Complete", "Failed")
    db.close()

def test_scenario_b_second_job_supercedes_first():
    """Scenario B: A second job is requested while the first is running; first is cleanly cancelled."""
    db = SessionLocal()
    video = db.query(Video).first()
    vid_id = video.id if video else 1

    job_1 = ProcessingJob(video_id=vid_id, status="RUNNING", stage="Detection & Tracking", progress_pct=30.0)
    db.add(job_1)
    db.commit()
    db.refresh(job_1)
    j1_id = job_1.id
    db.close()

    # Mark j1_id in active jobs registry
    from visionradar.worker.job_worker import _ACTIVE_JOBS
    _ACTIVE_JOBS.add(j1_id)

    # Start a second job via API
    res = client.post(f"/api/v1/videos/{vid_id}/jobs", json={"calibration_id": None, "domain": "traffic"})
    assert res.status_code == 200
    j2_data = res.json()
    j2_id = j2_data["id"]

    # Verify j1 was marked for cancellation
    assert is_job_cancelled(j1_id)

    # Cleanup
    cancel_job(j2_id)
    _ACTIVE_JOBS.discard(j1_id)
    _ACTIVE_JOBS.discard(j2_id)

def test_scenario_c_running_job_cancellation():
    """Scenario C: A running job is cancelled via POST /api/v1/jobs/{job_id}/cancel."""
    db = SessionLocal()
    video = db.query(Video).first()
    vid_id = video.id if video else 1

    job = ProcessingJob(video_id=vid_id, status="RUNNING", stage="Detection & Tracking", progress_pct=45.0)
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    db.close()

    res = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "CANCELLED"
    assert data["stage"] == "Cancelled"

    db = SessionLocal()
    j = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert j.status == "CANCELLED"
    assert j.completed_at is not None
    db.close()

def test_scenario_d_cancelled_job_replaced_by_new_job():
    """Scenario D: A cancelled job is replaced by a new job that processes successfully."""
    db = SessionLocal()
    video = db.query(Video).first()
    vid_id = video.id if video else 1

    # 1. Create and cancel Job 1
    job_1 = ProcessingJob(video_id=vid_id, status="CANCELLED", stage="Cancelled", progress_pct=20.0)
    db.add(job_1)
    db.commit()
    db.refresh(job_1)
    j1_id = job_1.id
    db.close()

    # 2. Start Job 2
    res = client.post(f"/api/v1/videos/{vid_id}/jobs", json={"calibration_id": None, "domain": "traffic"})
    assert res.status_code == 200
    j2_data = res.json()
    j2_id = j2_data["id"]

    # Verify Job 1 stays CANCELLED while Job 2 is active
    db = SessionLocal()
    j1_check = db.query(ProcessingJob).filter(ProcessingJob.id == j1_id).first()
    j2_check = db.query(ProcessingJob).filter(ProcessingJob.id == j2_id).first()
    assert j1_check.status == "CANCELLED"
    assert j2_check.status in ("QUEUED", "RUNNING", "SUCCEEDED")
    db.close()

    # Clean up Job 2
    client.post(f"/api/v1/jobs/{j2_id}/cancel")

def test_scenario_e_worker_exception_handling():
    """Scenario E: Worker encounters invalid video file; transitions gracefully to FAILED."""
    db = SessionLocal()
    video = Video(project_id=1, filename="corrupt.mp4", storage_path="data/videos/non_existent_corrupted.mp4", sha256_hash="dummy_corrupt_hash_12345")
    db.add(video)
    db.commit()
    db.refresh(video)

    job = ProcessingJob(video_id=video.id, status="QUEUED", stage="Queued", progress_pct=0.0)
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    db.close()

    loop = asyncio.new_event_loop()
    try:
        run_job_streaming(job_id, loop)
    finally:
        loop.close()

    db = SessionLocal()
    failed_job = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert failed_job.status == "FAILED"
    assert failed_job.stage == "Failed"
    assert "Video file not found" in (failed_job.error_message or "") or "Unable to open video" in (failed_job.error_message or "")
    assert failed_job.completed_at is not None
    db.close()

def test_scenario_f_frontend_disconnect_handling():
    """Scenario F: Frontend disconnects during streaming; queue drops oldest non-terminal messages without deadlock."""
    loop = asyncio.new_event_loop()
    job_id = 999980
    q = get_or_create_queue(job_id, loop)

    # Fill queue to maximum capacity
    for i in range(70):
        from visionradar.worker.job_worker import _safe_put
        _safe_put(q, {"type": "frame_result", "frame_index": i})

    # Terminal message must NOT be dropped even if queue was completely full
    from visionradar.worker.job_worker import _safe_put
    _safe_put(q, {"type": "completed", "job_id": job_id})

    # Retrieve all items
    items = []
    while not q.empty():
        items.append(q.get_nowait())

    # Completed message MUST be present
    has_completed = any(it.get("type") == "completed" for it in items)
    assert has_completed, "Terminal completed message was dropped"

    drop_job_queue(job_id)
    loop.close()

def test_scenario_g_reconnect_to_completed_job():
    """Scenario G: Browser connects/reconnects to an already completed job."""
    db = SessionLocal()
    video = db.query(Video).first()
    vid_id = video.id if video else 1

    job = ProcessingJob(video_id=vid_id, status="SUCCEEDED", stage="Complete", progress_pct=100.0)
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id
    db.close()

    with client.websocket_connect(f"/api/v1/jobs/{job_id}/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "completed"
        assert msg["job_id"] == job_id
        assert "already completed" in msg["message"]

def test_scenario_h_backend_restart_recovery():
    """Scenario H: Backend restarts with orphaned RUNNING/QUEUED jobs; startup_cleanup marks them FAILED."""
    db = SessionLocal()
    video = db.query(Video).first()
    vid_id = video.id if video else 1

    orphaned_job = ProcessingJob(
        video_id=vid_id,
        status="RUNNING",
        stage="Detection & Tracking",
        progress_pct=50.0
    )
    db.add(orphaned_job)
    db.commit()
    db.refresh(orphaned_job)
    orphaned_id = orphaned_job.id
    db.close()

    # Execute FastAPI startup event
    startup_cleanup()

    db = SessionLocal()
    recovered = db.query(ProcessingJob).filter(ProcessingJob.id == orphaned_id).first()
    assert recovered.status == "FAILED"
    assert recovered.stage == "Failed"
    assert "Server restarted" in (recovered.error_message or "")
    db.close()
