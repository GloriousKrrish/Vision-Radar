import pytest
import time
import threading
from fastapi.testclient import TestClient
from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video
from visionradar.worker.job_worker import (
    cancel_job, cancel_all_active_jobs, is_job_cancelled,
    _ACTIVE_JOBS, _CANCELLED_JOBS
)

client = TestClient(app)

def test_job_cancellation_registry_api():
    """Verify thread-safe cancellation tracking functions."""
    job_id = 999991
    assert not is_job_cancelled(job_id)
    
    cancel_job(job_id)
    assert is_job_cancelled(job_id)

    # Clean up
    _CANCELLED_JOBS.discard(job_id)
    assert not is_job_cancelled(job_id)

def test_cancel_all_active_jobs():
    """Verify bulk cancellation cancels all active job IDs."""
    job_ids = [999992, 999993]
    for jid in job_ids:
        _ACTIVE_JOBS.add(jid)

    cancel_all_active_jobs()
    for jid in job_ids:
        assert is_job_cancelled(job_id=jid)

    # Clean up
    for jid in job_ids:
        _ACTIVE_JOBS.discard(jid)
        _CANCELLED_JOBS.discard(jid)

def test_rest_cancel_endpoint():
    """Verify POST /api/v1/jobs/{job_id}/cancel endpoint sets status to CANCELLED."""
    db = SessionLocal()
    video = db.query(Video).first()
    vid_id = video.id if video else 1

    job = ProcessingJob(
        video_id=vid_id,
        status="RUNNING",
        stage="Detection & Tracking",
        progress_pct=25.0
    )
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

    # Verify in DB
    db = SessionLocal()
    updated = db.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
    assert updated.status == "CANCELLED"
    db.close()
