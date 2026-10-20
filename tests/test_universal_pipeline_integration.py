import os
import pytest
from fastapi.testclient import TestClient

from apps.api.main import app
from visionradar.models.database import SessionLocal
from visionradar.models.entities import (
    Project, Video, ProcessingJob, Track, SpeedMeasurement,
    ExaminationTrackRecord, ExaminationEventRecord, SceneClassificationRecord
)

client = TestClient(app)

def test_pipeline_isolation_and_envelope_contract():
    """
    Empirical integration test verifying:
    1. Common Envelope schema conforms to Layer D specifications.
    2. Traffic jobs and Examination jobs maintain strict data isolation in database.
    """
    db = SessionLocal()
    try:
        # Create test project & video if none exist
        proj = db.query(Project).first()
        if not proj:
            proj = Project(name="Test Universal Project")
            db.add(proj)
            db.commit()
            db.refresh(proj)

        video = db.query(Video).first()
        if not video:
            video = Video(
                project_id=proj.id,
                filename="test_video.mp4",
                storage_path="data/videos/Traffic1.mp4",
                sha256_hash="test_hash_12345",
                duration_sec=10.0,
                fps=30.0,
                width=1920,
                height=1080
            )
            db.add(video)
            db.commit()
            db.refresh(video)

        # 1. Create a Traffic Job
        job_traffic = ProcessingJob(
            video_id=video.id,
            status="SUCCEEDED",
            stage="Completed",
            progress_pct=100.0,
            config_json={"module": "traffic"},
            telemetry_json={"average_fps": 30.0}
        )
        db.add(job_traffic)
        db.commit()
        db.refresh(job_traffic)

        # Attach a traffic track
        trk = Track(
            job_id=job_traffic.id,
            track_id=101,
            vehicle_class="Car",
            first_frame=0,
            last_frame=30
        )
        db.add(trk)
        db.commit()

        # 2. Create an Examination Job
        job_exam = ProcessingJob(
            video_id=video.id,
            status="SUCCEEDED",
            stage="Completed",
            progress_pct=100.0,
            config_json={"module": "examination"},
            telemetry_json={"average_fps": 28.5}
        )
        db.add(job_exam)
        db.commit()
        db.refresh(job_exam)

        # Attach an examination event with unique ID
        unique_event_id = f"test_evt_iso_{job_exam.id}"
        exam_evt = ExaminationEventRecord(
            event_id=unique_event_id,
            job_id=job_exam.id,
            video_id=video.id,
            event_type="POSSIBLE_PHONE_INTERACTION",
            start_frame=10,
            end_frame=20,
            start_time=0.33,
            end_time=0.66,
            confidence=0.88,
            uncertainty=0.10,
            description="Observable phone proximity test",
            review_status="PENDING"
        )
        db.add(exam_evt)
        db.commit()

        # 3. Verify Isolation Gate
        # Traffic job MUST NOT have examination events or examination tracks
        traffic_exam_tracks = db.query(ExaminationTrackRecord).filter(ExaminationTrackRecord.job_id == job_traffic.id).all()
        traffic_exam_events = db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job_traffic.id).all()
        assert len(traffic_exam_tracks) == 0
        assert len(traffic_exam_events) == 0

        # Examination job MUST NOT have traffic vehicle tracks or speed measurements
        exam_traffic_tracks = db.query(Track).filter(Track.job_id == job_exam.id).all()
        assert len(exam_traffic_tracks) == 0

        # 4. Verify Common Envelope API Contract for Traffic
        res_t = client.get(f"/api/v1/jobs/{job_traffic.id}/results")
        assert res_t.status_code == 200
        data_t = res_t.json()
        assert data_t["schema_version"] == "2.0"
        assert data_t["analysis_module"] == "traffic"
        assert data_t["job_id"] == job_traffic.id
        assert "tracks_count" in data_t["payload"]

        # 5. Verify Common Envelope API Contract for Examination
        res_e = client.get(f"/api/v1/jobs/{job_exam.id}/results")
        assert res_e.status_code == 200
        data_e = res_e.json()
        assert data_e["schema_version"] == "2.0"
        assert data_e["analysis_module"] == "examination"
        assert data_e["job_id"] == job_exam.id
        assert len(data_e["payload"]["candidate_events"]) == 1
        assert data_e["payload"]["candidate_events"][0]["event_id"] == unique_event_id

        # 6. Verify Human Proctor Review Flow
        patch_res = client.patch(
            f"/api/v1/examination/events/{unique_event_id}/review",
            json={"review_status": "ACCEPTED", "reviewer_notes": "Reviewed and confirmed by chief proctor"}
        )
        assert patch_res.status_code == 200
        patch_data = patch_res.json()
        assert patch_data["review_status"] == "ACCEPTED"
        assert "confirmed" in patch_data["reviewer_notes"]

    finally:
        # Clean up test artifacts
        try:
            db.query(ExaminationEventRecord).filter(ExaminationEventRecord.job_id == job_exam.id).delete()
            db.query(Track).filter(Track.job_id == job_traffic.id).delete()
            db.query(ProcessingJob).filter(ProcessingJob.id.in_([job_traffic.id, job_exam.id])).delete()
            db.commit()
        except Exception:
            db.rollback()
        db.close()
