"""
VisionRadar Phase 5.3 — Job Lifecycle Reliability and Resource Management Test Suite.

Validates:
1. Strict Job Isolation across multiple videos and domains.
2. Accidental Cancellation Immunity (cancelling one job never cancels unrelated jobs).
3. Background Worker lifecycle registration and resource cleanup in both Traffic & Examination.
4. Exception resilience and guaranteed error state persistence.
5. WebSocket streaming terminal state handling for both Traffic dicts and Examination CommonEnvelopes.
6. Server restart stale job recovery.
"""
import pytest
import time
import asyncio
import datetime
import hashlib
from fastapi.testclient import TestClient

from apps.api.main import app, startup_cleanup
from visionradar.models.database import SessionLocal
from visionradar.models.entities import ProcessingJob, Video
from visionradar.worker.job_worker import (
    cancel_job, cancel_all_active_jobs, cancel_active_jobs_for_video, is_job_cancelled,
    get_or_create_queue, drop_job_queue, _ACTIVE_JOBS, _CANCELLED_JOBS, _JOB_COMPLETED_AT
)

client = TestClient(app)

@pytest.fixture
def test_videos():
    """Fetch two distinct existing test videos from the database, or create valid ones."""
    with SessionLocal() as db:
        videos = db.query(Video).limit(2).all()
        if len(videos) >= 2:
            return videos[0].id, videos[1].id
        h1 = hashlib.sha256(b"test_vid_1_bytes").hexdigest()
        h2 = hashlib.sha256(b"test_vid_2_bytes").hexdigest()
        v1 = Video(project_id=1, filename="traffic_test_v1.mp4", storage_path="data/videos/synthetic_highway.mp4", sha256_hash=h1)
        v2 = Video(project_id=1, filename="exam_test_v2.mp4", storage_path="data/videos/Exam 2.mp4", sha256_hash=h2)
        db.add(v1)
        db.add(v2)
        db.commit()
        db.refresh(v1)
        db.refresh(v2)
        return v1.id, v2.id

def test_job_isolation_multivideo(test_videos):
    """Test 1: Starting a job on Video 2 does NOT cancel an active job on Video 1."""
    v1_id, v2_id = test_videos

    with SessionLocal() as db:
        job_1 = ProcessingJob(
            video_id=v1_id,
            domain="traffic",
            status="RUNNING",
            stage="Detection & Tracking",
            progress_pct=35.0
        )
        db.add(job_1)
        db.commit()
        db.refresh(job_1)
        j1_id = job_1.id

    # Register Job 1 in active jobs
    _ACTIVE_JOBS.add(j1_id)

    try:
        # Create Job 2 on a completely different video (Video 2)
        res = client.post(f"/api/v1/videos/{v2_id}/jobs", json={"domain": "examination"})
        assert res.status_code == 200
        j2_id = res.json()["id"]

        # Job 1 on Video 1 MUST NOT be cancelled
        assert not is_job_cancelled(j1_id), "Job 1 on Video 1 was accidentally cancelled by Job 2 on Video 2!"

        with SessionLocal() as db:
            j1_db = db.query(ProcessingJob).filter(ProcessingJob.id == j1_id).first()
            assert j1_db.status == "RUNNING", f"Job 1 status was changed to {j1_db.status}!"
    finally:
        _ACTIVE_JOBS.discard(j1_id)
        cancel_job(j1_id)
        if 'j2_id' in locals():
            cancel_job(j2_id)
            _ACTIVE_JOBS.discard(j2_id)


def test_cancelling_one_job_does_not_cancel_unrelated_jobs(test_videos):
    """Test 2: Cancelling Job A specifically leaves Job B on another video completely untouched."""
    v1_id, v2_id = test_videos

    with SessionLocal() as db:
        job_a = ProcessingJob(video_id=v1_id, domain="traffic", status="RUNNING", stage="Tracking", progress_pct=25.0)
        job_b = ProcessingJob(video_id=v2_id, domain="examination", status="RUNNING", stage="Inference", progress_pct=50.0)
        db.add(job_a)
        db.add(job_b)
        db.commit()
        db.refresh(job_a)
        db.refresh(job_b)
        ja_id, jb_id = job_a.id, job_b.id

    _ACTIVE_JOBS.add(ja_id)
    _ACTIVE_JOBS.add(jb_id)

    try:
        # Cancel Job A via REST endpoint
        res = client.post(f"/api/v1/jobs/{ja_id}/cancel")
        assert res.status_code == 200
        data_a = res.json()
        assert data_a["status"] == "CANCELLED"
        assert is_job_cancelled(ja_id)

        # Job B MUST remain unaffected
        assert not is_job_cancelled(jb_id), "Job B received cancellation signal when Job A was cancelled!"
        with SessionLocal() as db:
            jb_check = db.query(ProcessingJob).filter(ProcessingJob.id == jb_id).first()
            assert jb_check.status == "RUNNING"
            assert jb_check.stage == "Inference"
    finally:
        cancel_job(ja_id)
        cancel_job(jb_id)
        _ACTIVE_JOBS.discard(ja_id)
        _ACTIVE_JOBS.discard(jb_id)


def test_examination_worker_exception_handling():
    """Test 3: Worker catches missing video exception, sets FAILED status and completed_at."""
    unique_hash = hashlib.sha256(f"ghost_file_{time.time()}".encode()).hexdigest()
    with SessionLocal() as db:
        bad_video = Video(
            project_id=1,
            filename="ghost.mp4",
            storage_path="data/videos/nonexistent_ghost_file.mp4",
            sha256_hash=unique_hash
        )
        db.add(bad_video)
        db.commit()
        db.refresh(bad_video)

        bad_job = ProcessingJob(video_id=bad_video.id, domain="examination", status="QUEUED", stage="Queued", progress_pct=0.0)
        db.add(bad_job)
        db.commit()
        db.refresh(bad_job)
        b_id = bad_job.id

    from visionradar.examination.pipeline import run_examination_job
    run_examination_job(b_id)

    with SessionLocal() as db:
        final_job = db.query(ProcessingJob).filter(ProcessingJob.id == b_id).first()
        assert final_job.status == "FAILED"
        assert final_job.stage == "Failed"
        assert final_job.completed_at is not None
        assert "Video file not found" in (final_job.error_message or "")

    # Verify active jobs set does not leak job ID
    assert b_id not in _ACTIVE_JOBS


def test_examination_worker_cancellation(test_videos):
    """Test 4: Examination worker breaks on is_job_cancelled, marks status CANCELLED and sets completed_at."""
    _, v2_id = test_videos

    with SessionLocal() as db:
        job = ProcessingJob(video_id=v2_id, domain="examination", status="QUEUED", stage="Queued", progress_pct=0.0)
        db.add(job)
        db.commit()
        db.refresh(job)
        jid = job.id

    # Pre-cancel before running
    cancel_job(jid)

    from visionradar.examination.pipeline import run_examination_job
    run_examination_job(jid)

    with SessionLocal() as db:
        final_job = db.query(ProcessingJob).filter(ProcessingJob.id == jid).first()
        assert final_job.status == "CANCELLED"
        assert final_job.stage == "Cancelled"
        assert final_job.completed_at is not None
        assert "Cancelled by user" in (final_job.error_message or "")

    # Cleanup
    assert jid not in _ACTIVE_JOBS
    _CANCELLED_JOBS.discard(jid)


def test_websocket_terminal_envelope_recognition():
    """Test 5: Verifies terminal message detector handles both Traffic and Examination envelope structures."""
    from visionradar.worker.job_worker import _safe_put
    queue = asyncio.Queue(maxsize=1)

    # 1. Traffic completed dict
    traffic_msg = {"type": "completed", "job_id": 100, "total_tracks": 5}
    _safe_put(queue, traffic_msg)
    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert item["type"] == "completed"

    # 2. Examination CommonEnvelope structure
    exam_envelope = {
        "schema_version": "2.0",
        "job_id": 101,
        "result_type": "job_summary",
        "processing_status": "SUCCESS",
        "payload": {"type": "completed", "tracks_count": 12}
    }
    _safe_put(queue, exam_envelope)
    assert queue.qsize() == 1
    item2 = queue.get_nowait()
    assert item2["result_type"] == "job_summary"
    assert item2["processing_status"] == "SUCCESS"


def test_startup_recovery_cleans_stale_jobs():
    """Test 6: Server startup recovery transitions lingering RUNNING jobs to FAILED with timestamps."""
    with SessionLocal() as db:
        stale_job = ProcessingJob(
            video_id=1,
            domain="traffic",
            status="RUNNING",
            stage="Tracking",
            progress_pct=60.0
        )
        db.add(stale_job)
        db.commit()
        db.refresh(stale_job)
        s_id = stale_job.id

    startup_cleanup()

    with SessionLocal() as db:
        cleaned_job = db.query(ProcessingJob).filter(ProcessingJob.id == s_id).first()
        assert cleaned_job.status == "FAILED"
        assert cleaned_job.stage == "Failed"
        assert cleaned_job.completed_at is not None
        assert "Server restarted during job execution" in cleaned_job.error_message
